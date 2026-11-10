/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Binary front (crates/tpu/src/dis.rs, binlift.rs): decoder agreement with the listings on
//! the differential corpora and the captured Kilo/Pallas kernels (the numbers of
//! tools/tpudis.py --test and tools/kcov.py), and three-front simulation agreement
//! (original / final_bundles / machine code) on every probe-corpus kernel with an executable.
//! Data lives under fb/targets/tpu/probes (large parts untracked); absent sets are skipped.

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use ptxv_tpu::binlift;
use ptxv_tpu::binlift::Score;
use ptxv_tpu::corpus::KernelDir;
use ptxv_tpu::machine::DmaPolicy;
use ptxv_tpu::num::PLACEHOLDER;
use ptxv_tpu_ir::Front;
use ptxv_tpu_ir::Kernel;

fn probes() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fb/targets/tpu/probes")
}

fn show(name: &str, s: &Score) {
    eprintln!(
        "{name:70} bundles {:6}/{:6} ops {:7}/{:7} {:7.3}%  {:?}",
        s.bundles_exact,
        s.bundles,
        s.ops_matched,
        s.ops,
        s.pct(),
        s.top_miss(4)
    );
}

/// tools/tpudis.py --test numbers (matched / listing ops), 2026-11-09.
const CORPUS: &[(&str, usize, usize)] = &[
    ("binary/corpus/data/train_c1", 1019, 1020),
    ("binary/corpus/data/train_c2", 1248, 1249),
    ("binary/corpus/data/train_c3", 306, 307),
    ("binary/corpus/data/holdout1", 1009, 1010),
    ("binary/corpus/data/holdout2", 1021, 1023),
];

#[test]
fn decoder_agrees_with_differential_corpora() {
    let mut any = false;
    for (d, ok, n) in CORPUS {
        let p = probes().join(d);
        if !p.join("rt").exists() {
            eprintln!("skip {d}: absent");
            continue;
        }
        any = true;
        let s = binlift::score_corpus_dir(&p).expect("score");
        show(d, &s);
        assert_eq!(
            (s.ops_matched, s.ops),
            (*ok, *n),
            "{d}: Python tpudis.py --test gives {ok}/{n}"
        );
    }
    for d in ["wheel", "wheel_cmp", "nightly", "nightly_cmp"] {
        let p = probes().join("binary/kcorpus/data").join(d);
        if !p.join("rt").exists() {
            eprintln!("skip kcorpus/{d}: absent");
            continue;
        }
        any = true;
        let s = binlift::score_corpus_dir(&p).expect("score");
        show(&format!("kcorpus/{d}"), &s);
        assert!(s.pct() >= 99.5, "kcorpus/{d}: {:.3}%", s.pct());
    }
    if !any {
        eprintln!("no differential corpus present; nothing checked");
    }
}

fn capture_dirs(root: &Path, out: &mut Vec<PathBuf>) {
    let Ok(rd) = std::fs::read_dir(root) else {
        return;
    };
    let mut v: Vec<PathBuf> = rd
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.is_dir())
        .collect();
    v.sort();
    for p in v {
        if (p.join("exe.rt.bin").exists() || p.join("exe0.rt.bin").exists())
            && p.join("llo").exists()
        {
            out.push(p);
        } else {
            capture_dirs(&p, out);
        }
    }
}

/// Every captured Kilo/Pallas kernel (probes/kilo): tools/kcov.py gives >= 99.86% of listing
/// ops per variant and 99.96% / 99.97% over the train / held-out sets.
#[test]
fn decoder_agrees_with_kilo_kernels() {
    let mut dirs = Vec::new();
    capture_dirs(&probes().join("kilo"), &mut dirs);
    if dirs.is_empty() {
        eprintln!("skip: no Kilo/Pallas captures under probes/kilo");
        return;
    }
    let mut total = Score::default();
    let mut worst = 100.0f64;
    for d in &dirs {
        let s = binlift::score_capture_dir(d).expect("score");
        let rel = d
            .strip_prefix(probes().join("kilo"))
            .unwrap_or(d)
            .display()
            .to_string();
        show(&rel, &s);
        if s.ops > 0 {
            worst = worst.min(s.pct());
        }
        total.add(&s);
    }
    show("TOTAL", &total);
    assert!(worst >= 99.8, "worst variant {worst:.3}%");
    assert!(total.pct() >= 99.9, "total {:.3}%", total.pct());
}

fn run(kd: &KernelDir, k: Kernel) -> (String, Vec<(usize, Vec<u32>)>) {
    let mut m = kd.machine(Arc::new(k), &PLACEHOLDER).expect("machine");
    let r = m.run(DmaPolicy::Eager);
    let status = match &r {
        Ok(_) => "ok".to_string(),
        Err(e) => format!("{}: {}", e.status(), e),
    };
    let outs = if r.is_ok() {
        kd.outputs(&m)
            .expect("outputs")
            .into_iter()
            .map(|(c, a)| (c, a.data))
            .collect()
    } else {
        Vec::new()
    };
    (status, outs)
}

/// Three fronts on every probe-corpus kernel with an executable: the binary front simulates
/// to the same outputs (bit for bit) and status as the final_bundles front.
#[test]
fn three_fronts_agree_on_corpus() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tpu_corpus/k");
    let rd = std::fs::read_dir(&root).expect("fixtures/tpu_corpus");
    let mut names: Vec<String> = rd
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .collect();
    names.sort();
    let mut checked = 0;
    let mut bad = Vec::new();
    for name in names {
        let kd = KernelDir::open(&root.join(&name)).expect("kernel dir");
        if kd.exe.is_none() {
            continue;
        }
        let fin = kd.lift(Front::Final);
        let bin = binlift::lift_kernel_dir(&kd).expect("binary lift");
        let orig = kd.lift(Front::Original);
        let unl = |k: &Kernel| k.unlowered.len();
        let (sf, of) = run(&kd, fin.clone());
        let (sb, ob) = run(&kd, bin.clone());
        let (so, oo) = run(&kd, orig);
        let hw: Vec<bool> = of
            .iter()
            .enumerate()
            .map(|(i, (c, a))| {
                let per = of[..i].iter().filter(|x| x.0 == *c).count();
                kd.expected(*c, per)
                    .map(|(w, _)| w.data == *a)
                    .unwrap_or(false)
            })
            .collect();
        eprintln!(
            "{name:14} final {sf:6} binary {sb:6} original {so:6} | unlowered f/b {}/{} | delays f/b {}/{} | bin==fin {} orig==fin {} fin==hw {:?}",
            unl(&fin),
            unl(&bin),
            fin.delay_source,
            bin.delay_source,
            ob == of,
            oo == of,
            hw
        );
        for n in &bin.notes {
            eprintln!("   {n}");
        }
        checked += 1;
        if sb != sf || ob != of {
            bad.push(format!(
                "{name}: binary {sb} vs final {sf}, outputs equal {}",
                ob == of
            ));
        }
        if unl(&bin) > unl(&fin) {
            bad.push(format!(
                "{name}: binary front leaves {} unlowered ops, final {}",
                unl(&bin),
                unl(&fin)
            ));
        }
    }
    eprintln!("{checked} kernels with executables");
    assert!(bad.is_empty(), "{bad:#?}");
}

/// Negative check: a one-bit change in the machine code (the destination register of
/// vec_loop's accumulate) is visible to the binary front as a listing disagreement and changes
/// the simulated output; the final_bundles front cannot see it.
#[test]
fn binary_front_sees_patched_machine_code() {
    let kdir =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tpu_corpus/k/vec_loop");
    let kd = KernelDir::open(&kdir).expect("kernel dir");
    let exe_path = kd.exe.clone().expect("vec_loop has an executable");
    let mut bytes = std::fs::read(&exe_path).expect("exe");
    let img = ptxv_tpu::exe::code_image(&bytes).expect("image");
    let name = ptxv_tpu::exe::listing_name(&kd.final_path).expect("name");
    let placed = binlift::place_kernel(&img, &kd.llo_dir, &name).expect("placed");
    let (raw, base) = placed
        .iter()
        .find_map(|p| {
            let d = ptxv_tpu::dis::decode(&p.raw)
                .into_iter()
                .find(|d| d.mn == "vadd.f32")?;
            let base = match d.slot {
                "valu0" => 270,
                "valu1" => 237,
                "valu2" => 204,
                _ => 171,
            };
            Some((p.raw, base))
        })
        .expect("a vadd.f32 bundle");
    let at = bytes
        .windows(64)
        .position(|w| w == raw)
        .expect("bundle bytes in the executable");
    bytes[at + base / 8] ^= 1 << (base % 8); // lowest bit of the destination vreg
    let ctx = binlift::ListingCtx {
        llo_dir: &kd.llo_dir,
        kernel: &name,
        final_text: &kd.final_text,
        original_text: &kd.original_text,
    };
    let (_, _, st) = binlift::binary_listing(&bytes, &ctx).expect("render");
    assert!(
        !st.disagree.is_empty(),
        "the patched destination should disagree with the listing"
    );
    let k = binlift::lift_binary(&bytes, &ctx).expect("lift");
    let (_, patched) = run(&kd, k);
    let (_, fin) = run(&kd, kd.lift(Front::Final));
    assert_ne!(
        patched, fin,
        "the patched accumulate should change the output"
    );
}
