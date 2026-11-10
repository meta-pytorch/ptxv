//! Rust mirror of tools/test_mxu_machine.py: the MXU op semantics (notes/numerics.md (e)) run
//! through TpuMachine with the calibrated numerics, every output bit compared with hardware.
//! Default test: the committed subset fixtures/tpu_mxu (xposeB push1/push3 .xpose, HIGHEST,
//! review_mxu_ops3 lay and bat with binary delays). The full sets are `#[ignore]`d: they need the
//! hardware data root `PTXV_TPU_DATA` (`<probe>/out/*.npz`, `<probe>/out/k/`) and the probes'
//! listings (`PTXV_TPU_PROBES`, default fb/targets/tpu/probes; `out/` is not committed). Cases: gridepi (1,N) bias layout
//! (ASSUME(TPU-L-ROW-TILE)), push1/push3 .xpose einsums, the bf16-accumulator grid with binary
//! branch delays, a K-loop grid, M2048 K2048 MRB accumulation, HIGHEST, int8 + nt einsum, and the
//! corpus mm kernels.

use std::collections::BTreeSet;
use std::path::Path;
use std::path::PathBuf;

use ptxv_tpu::corpus::KernelDir;
use ptxv_tpu::corpus::run_kernel;
use ptxv_tpu::data;
use ptxv_tpu::inflate::zip_member;
use ptxv_tpu::inflate::zip_names;
use ptxv_tpu::lift::operand_decls;
use ptxv_tpu::machine::DmaPolicy;
use ptxv_tpu_ir::Front;

/// Hardware data root (`PTXV_TPU_DATA`); the full-set tests are ignored by default.
fn data_root() -> PathBuf {
    let d = std::env::var("PTXV_TPU_DATA")
        .map(PathBuf::from)
        .expect("PTXV_TPU_DATA (hardware data root) is required for this ignored test");
    assert!(d.exists(), "PTXV_TPU_DATA={} absent", d.display());
    d
}

fn probes() -> PathBuf {
    std::env::var("PTXV_TPU_PROBES")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fb/targets/tpu/probes")
        })
}

fn fixtures() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tpu_mxu")
}

#[derive(Default)]
struct Case {
    name: &'static str,
    npz: &'static str,
    probe: &'static str,
    kpre: &'static str,
    only: &'static [&'static str],
    parts: usize,
    subs: &'static [(&'static str, &'static str)],
    c_in_d: bool,
    /// (input index, tag substring): saved as the logical operand, the kernel takes its transpose
    tin: Option<(usize, &'static str)>,
    rt: bool,
}

fn cases() -> Vec<Case> {
    vec![
        Case {
            name: "gridepi bias (1,N)",
            npz: "review_mxu_ops/out/rmo_8201.npz",
            probe: "review_mxu_ops",
            kpre: "rmo",
            only: &["gridepi_normal"],
            parts: 1,
            c_in_d: true,
            ..Default::default()
        },
        Case {
            name: "xpose push1+push3",
            npz: "review_mxu_ops/out/rmo_8201.npz",
            probe: "review_mxu_ops",
            kpre: "rmo",
            only: &["xposeA_normal", "xposeB_normal"],
            parts: 1,
            c_in_d: true,
            tin: Some((1, "xpose")),
            ..Default::default()
        },
        Case {
            name: "bf16 acc grid, binary delays",
            npz: "mxu_rt/out/rrt_8211.npz",
            probe: "mxu_rt",
            kpre: "rrt",
            only: &["gridbfacc_normal"],
            parts: 1,
            rt: true,
            ..Default::default()
        },
        Case {
            name: "K-loop grid",
            npz: "mxu_loop/out/loop_7811.npz",
            probe: "mxu_loop",
            kpre: "loop",
            only: &["grid_*_cancel", "fori_*_cancel"],
            c_in_d: true,
            ..Default::default()
        },
        Case {
            name: "MRB accumulation M2048 K2048",
            npz: "review_mxu/out/rev_7001.npz",
            probe: "review_mxu",
            kpre: "rev",
            only: &["m2048k2048n128_ovf"],
            ..Default::default()
        },
        Case {
            name: "HIGHEST",
            npz: "review_mxu_f32/out/rf32_7101.npz",
            probe: "review_mxu_f32",
            kpre: "rf32",
            only: &["HI_M256_K128_N128_normal"],
            parts: 4,
            subs: &[("DEF", "DEFAULT"), ("HI", "HIGHEST")],
            ..Default::default()
        },
        Case {
            name: "int8 + nt einsum",
            npz: "review_sched/out/rs_7901.npz",
            probe: "review_sched",
            kpre: "rs",
            only: &["int8_*", "nt_*"],
            parts: 9,
            c_in_d: true,
            ..Default::default()
        },
    ]
}

/// Prefix match with one optional `*` (any run of characters) — the regexes of the Python cases.
fn matches(tag: &str, pats: &[&str]) -> bool {
    pats.iter().any(|p| match p.split_once('*') {
        None => tag.starts_with(p),
        Some((a, b)) => tag.starts_with(a) && tag[a.len()..].contains(b),
    })
}

fn transpose(a: &data::Array) -> data::Array {
    let (r, c) = (a.shape[0] as usize, a.shape[1] as usize);
    let mut t = a.clone();
    t.shape = vec![a.shape[1], a.shape[0]];
    for i in 0..r {
        for j in 0..c {
            t.data[j * r + i] = a.data[i * c + j];
        }
    }
    t
}

fn npz_array(z: &[u8], key: &str, tmp: &Path) -> data::Array {
    let p = tmp.join("member.npy");
    std::fs::write(&p, zip_member(z, &format!("{key}.npy")).unwrap()).unwrap();
    data::read_npy(&p).unwrap()
}

fn link(src: &Path, dir: &Path) {
    let dst = dir.join(src.file_name().unwrap());
    if !dst.exists() {
        std::os::unix::fs::symlink(src, dst).unwrap();
    }
}

/// Runs one case: (mismatching elements, compared elements, failed runs).
fn run_case(c: &Case, tmp: &Path) -> (usize, usize, Vec<String>) {
    let z = std::fs::read(data_root().join(c.npz)).unwrap();
    let names = zip_names(&z).unwrap();
    let keys: BTreeSet<&str> = names
        .iter()
        .filter_map(|n| n.strip_suffix(".npy"))
        .collect();
    let tags: Vec<&str> = keys.iter().filter_map(|k| k.strip_suffix("_D")).collect();
    let llo = probes().join(c.probe).join("out/llo");
    let llo_files: Vec<PathBuf> = std::fs::read_dir(&llo)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    let (mut bad, mut n, mut fails) = (0, 0, Vec::new());
    let num = ptxv_tpu::num::calibrated();
    for t in tags.iter().filter(|t| matches(t, c.only)) {
        let parts: Vec<&str> = t.split('_').collect();
        let mut shape = if c.parts > 0 {
            parts[..c.parts.min(parts.len())].join("_")
        } else {
            parts[..parts.len() - 1].join("_")
        };
        for (o, nw) in c.subs {
            shape = shape.replace(o, nw);
        }
        for (kind, kname) in [
            ("D", format!("{}_{shape}", c.kpre)),
            ("DC", format!("{}c_{shape}", c.kpre)),
        ] {
            if !keys.contains(format!("{t}_{kind}").as_str()) {
                continue;
            }
            let suffix = format!("-{kname}.1-02-final_bundles.txt");
            let fins: Vec<&PathBuf> = llo_files
                .iter()
                .filter(|p| p.to_string_lossy().ends_with(&suffix))
                .collect();
            if fins.is_empty() {
                fails.push(format!("{t} {kind}: no listing"));
                continue;
            }
            let kd = tmp.join("k").join(&kname);
            let ld = tmp.join("llo").join(&kname);
            let _ = std::fs::remove_dir_all(&kd);
            std::fs::create_dir_all(&kd).unwrap();
            std::fs::create_dir_all(&ld).unwrap();
            let rtf = probes()
                .join(c.probe)
                .join("out/rt")
                .join(format!("{kname}.bin"));
            if c.rt && rtf.exists() {
                std::fs::create_dir_all(tmp.join("rt")).unwrap();
                link(&rtf, &tmp.join("rt"));
                for f in &llo_files {
                    link(f, &ld);
                }
            } else {
                // the HLO dumps carry the operand layouts (ASSUME(TPU-L-OPERAND-LAYOUT))
                for f in llo_files
                    .iter()
                    .filter(|p| p.to_string_lossy().ends_with("-hlo.txt"))
                {
                    link(f, &ld);
                }
                for f in &fins {
                    link(f, &ld);
                    let o = f
                        .to_string_lossy()
                        .replace("-02-final_bundles.txt", "-00-original.txt");
                    link(Path::new(&o), &ld);
                }
            }
            let withc = (kind == "DC" || c.c_in_d) && keys.contains(format!("{t}_C").as_str());
            let mut ins = vec![
                npz_array(&z, &format!("{t}_A"), tmp),
                npz_array(&z, &format!("{t}_B"), tmp),
            ];
            if withc {
                ins.push(npz_array(&z, &format!("{t}_C"), tmp));
            }
            let decls = operand_decls(&std::fs::read_to_string(fins[fins.len() - 1]).unwrap());
            let mut ok = true;
            for (i, mut a) in ins.into_iter().enumerate() {
                let want: Vec<u64> = decls[i].shape.clone();
                if c.tin.is_some_and(|(ti, s)| ti == i && t.contains(s)) {
                    a = transpose(&a);
                }
                if a.shape != want && a.shape.len() == 2 && a.shape.iter().rev().eq(want.iter()) {
                    a = transpose(&a);
                }
                if a.shape != want {
                    fails.push(format!(
                        "{t} {kind}: input {i} shape {:?} vs {want:?}",
                        a.shape
                    ));
                    ok = false;
                    break;
                }
                data::write_npy(&kd.join(format!("in{i}.npy")), &a).unwrap();
            }
            if !ok {
                continue;
            }
            let out = npz_array(&z, &format!("{t}_{kind}"), tmp);
            data::write_npy(&kd.join("out0.npy"), &out).unwrap();
            std::fs::write(kd.join("meta.json"), format!("{{\"kernel\": \"{kname}\"}}")).unwrap();
            let r = KernelDir::open(&kd)
                .and_then(|k| run_kernel(&k, Front::Final, DmaPolicy::Eager, num));
            match r {
                Ok(r) if r.status == "ok" => {
                    for o in &r.outputs {
                        bad += o.mismatch;
                        n += o.n;
                    }
                    eprintln!(
                        "  {t:36} {kind:2} {}/{}",
                        r.outputs.iter().map(|o| o.mismatch).sum::<usize>(),
                        r.outputs.iter().map(|o| o.n).sum::<usize>()
                    );
                }
                Ok(r) => fails.push(format!("{t} {kind}: {} {}", r.status, r.error)),
                Err(e) => fails.push(format!("{t} {kind}: {e:#}")),
            }
        }
    }
    (bad, n, fails)
}

#[test]
#[ignore = "full npz sets: needs PTXV_TPU_DATA and probes/*/out listings"]
fn mxu_regression_cases() {
    let tmp = std::env::temp_dir().join(format!("ptxv_mxu_cases_{}", std::process::id()));
    let mut failed = Vec::new();
    for c in cases() {
        let _ = std::fs::remove_dir_all(&tmp);
        let (bad, n, fails) = run_case(&c, &tmp);
        let ok = bad == 0 && fails.is_empty() && n > 0;
        eprintln!(
            "{} {:32} {bad}/{n} failed runs {} {:?}",
            if ok { "ok  " } else { "FAIL" },
            c.name,
            fails.len(),
            fails
        );
        if !ok {
            failed.push(c.name);
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
    assert!(failed.is_empty(), "{failed:?}");
}

/// Round-4b counterexample (probes/review_mxu_ops2 bcast: grid epilogue with broadcast operands
/// f32[1,256] T(1,128), f32[4,256] T(4,128), f32[1,128] T(1,128), f32[16,256] T(8,128)) plus the
/// K-looped 'mk,nk->mn' einsum and the bf16-accumulator grid; delays from the executables. Data in
/// corpus layout (`k/<kernel>_<dist>/`). Layouts come from the HLO (ASSUME(TPU-L-OPERAND-LAYOUT)).
#[test]
#[ignore = "needs PTXV_TPU_DATA (review_mxu_ops2/out/k) and probes/review_mxu_ops2/out"]
fn review_mxu_ops2_kernels() {
    let d = data_root().join("review_mxu_ops2/out/k");
    corpus_layout_probe(
        "review_mxu_ops2",
        &d,
        &probes().join("review_mxu_ops2/out"),
        9,
    );
}

/// Round-4c layout stress set (probes/review_mxu_ops3): epilogue operands f32 (2|3|5|6|7, 200),
/// bf16 (3, 200) T(4,128)(2,1), int8 (1, 200) T(4,128)(4,1), rank-3/4 f32 T(4,128), batched
/// einsums; needs `vperm.slane` sublane broadcasts and `vunpack.c.0.s8.s32`.
#[test]
#[ignore = "needs PTXV_TPU_DATA (review_mxu_ops3/out/k) and probes/review_mxu_ops3/out"]
fn review_mxu_ops3_kernels() {
    let d = data_root().join("review_mxu_ops3/out/k");
    corpus_layout_probe(
        "review_mxu_ops3",
        &d,
        &probes().join("review_mxu_ops3/out"),
        8,
    );
}

/// Committed subset (fixtures/tpu_mxu, built by fb/targets/tpu/tools/make_tpu_fixtures.py):
/// xposeB_normal (vmatpush1 and vmatpush3 .xpose, review_mxu_ops seed 8201), HIGHEST M256 K128
/// N128 (f32 LHS passes, review_mxu_f32 seed 7101), rmo3_lay_normal and rmo3_bat_normal (HLO
/// operand layouts T(2|4|8,128), bf16 T(4,128)(2,1), s8 T(4,128)(4,1), rank-3/4, `vperm.slane`,
/// `vunpack.c.0.s8.s32`, binary branch delays; review_mxu_ops3 seed 8401). Bitwise vs hardware.
#[test]
fn mxu_fixture_kernels() {
    corpus_layout_probe("tpu_mxu", &fixtures().join("k"), &fixtures(), 4);
}

/// Every saved run `data/<tag>/` (meta.json names the kernel) against `root/llo` (one flat
/// listing directory, or `llo/<kernel>/`) and `root/rt`: status ok and 0 mismatching elements,
/// `runs` runs in total.
fn corpus_layout_probe(name: &str, data: &Path, probe: &Path, runs: usize) {
    let tmp = std::env::temp_dir().join(format!("ptxv_{name}_{}", std::process::id()));
    let num = ptxv_tpu::num::calibrated();
    let mut tags: Vec<String> = std::fs::read_dir(data)
        .unwrap()
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .collect();
    tags.sort();
    let mut bad = Vec::new();
    let mut total = (0, 0);
    for t in &tags {
        let meta: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(data.join(t).join("meta.json")).unwrap())
                .unwrap();
        let kname = meta["kernel"].as_str().unwrap().to_string();
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("k")).unwrap();
        // a real directory (KernelDir canonicalizes the path), files linked
        std::fs::create_dir_all(tmp.join("k").join(&kname)).unwrap();
        for e in std::fs::read_dir(data.join(t)).unwrap() {
            link(&e.unwrap().path(), &tmp.join("k").join(&kname));
        }
        std::fs::create_dir_all(tmp.join("llo")).unwrap();
        let per_kernel = probe.join("llo").join(&kname);
        let llo = if per_kernel.is_dir() {
            per_kernel
        } else {
            probe.join("llo")
        };
        std::os::unix::fs::symlink(llo, tmp.join("llo").join(&kname)).unwrap();
        if probe.join("rt").is_dir() {
            std::os::unix::fs::symlink(probe.join("rt"), tmp.join("rt")).unwrap();
        }
        let r = KernelDir::open(&tmp.join("k").join(&kname))
            .and_then(|k| run_kernel(&k, Front::Final, DmaPolicy::Eager, num))
            .unwrap();
        let (m, n): (usize, usize) = (
            r.outputs.iter().map(|o| o.mismatch).sum(),
            r.outputs.iter().map(|o| o.n).sum(),
        );
        eprintln!("  {t:28} {} {m}/{n} {}", r.status, r.error);
        total.0 += m;
        total.1 += n;
        if r.status != "ok" || m > 0 {
            bad.push(format!("{t}: {} {m}/{n} {}", r.status, r.error));
        }
    }
    let _ = std::fs::remove_dir_all(&tmp);
    eprintln!("{name}: {}/{} mismatches", total.0, total.1);
    assert!(bad.is_empty() && total.1 > 0, "{bad:?}");
    assert_eq!(tags.len(), runs, "{name}: saved runs");
}
