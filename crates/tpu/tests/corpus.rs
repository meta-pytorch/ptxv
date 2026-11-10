/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The probe corpus (fixtures/tpu_corpus, a copy of fb/targets/tpu/probes/corpus/out): every kernel, both fronts, against the
//! hardware outputs and against each other.

use std::path::PathBuf;

use ptxv_tpu::corpus::KernelDir;
use ptxv_tpu::corpus::run_kernel;
use ptxv_tpu::machine::DmaPolicy;
use ptxv_tpu::num::PLACEHOLDER;
use ptxv_tpu_ir::Front;

fn root() -> PathBuf {
    // committed copy of probes/corpus/out (listings, HLO, executables, I/O); regenerate with
    // fb/targets/tpu/tools/make_tpu_fixtures.py corpus
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tpu_corpus")
}

/// Kernels whose outputs differ from hardware only by MXU accumulation / EUP numerics
/// (placeholder models; tools/llosim.py agrees).
const NUMERICS_ONLY: &[&str] = &["div_f32", "mm_bf16", "mm_f32", "trans_f32"];

#[test]
fn corpus_bitwise() {
    let mut names: Vec<String> = std::fs::read_dir(root().join("k"))
        .expect("corpus out/k")
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .collect();
    names.sort();
    let (mut exact, mut agree, mut n) = ([0usize; 2], 0usize, 0usize);
    let mut bad = Vec::new();
    // Outputs of tools/llosim.py `check --dump DIR` (both passes) to compare against bitwise.
    let oracle = std::env::var("PTXV_LLOSIM_DUMP").ok().map(PathBuf::from);
    let mut oracle_checked = 0;
    for name in &names {
        let kd = KernelDir::open(&root().join("k").join(name)).expect("kernel dir");
        let fin = kd.lift(Front::Final);
        if fin.delay_source != "none" {
            eprintln!(
                "{name:14} branch delays: {} {:?}",
                fin.delay_source, fin.notes
            );
            if kd.exe.is_some() && fin.delay_source != "binary" {
                bad.push(format!(
                    "{name}: executable present but delays {}",
                    fin.delay_source
                ));
            }
        }
        let mut res = Vec::new();
        for (fi, front) in [Front::Final, Front::Original].into_iter().enumerate() {
            let r = run_kernel(&kd, front, DmaPolicy::Eager, &PLACEHOLDER).expect("run");
            let mm: Vec<String> = r
                .outputs
                .iter()
                .map(|o| format!("{}{}", o.mismatch, if o.nondet { "?" } else { "" }))
                .collect();
            eprintln!(
                "{name:14} {:8} {:8} exact={:?} mism=[{}] steps={} {}",
                r.front,
                r.status,
                r.exact,
                mm.join(","),
                r.steps,
                r.error
            );
            if let Some(dir) = &oracle {
                let mut per = vec![0usize; kd.ncores()];
                for (c, a) in &r.arrays {
                    let p = dir.join(format!("{name}.{}.d{c}_out{}.npy", r.front, per[*c]));
                    per[*c] += 1;
                    let want = ptxv_tpu::data::read_npy(&p).expect("oracle npy");
                    oracle_checked += 1;
                    if want.data != a.data {
                        bad.push(format!(
                            "{name}/{}: differs from llosim.py ({})",
                            r.front,
                            p.display()
                        ));
                    }
                }
            }
            if r.exact == Some(true) {
                exact[fi] += 1;
            } else if !NUMERICS_ONLY.contains(&name.as_str()) {
                bad.push(format!("{name}/{}", r.front));
            }
            res.push(r);
        }
        n += 1;
        let same = res[0].status == "ok"
            && res[1].status == "ok"
            && res[0].arrays.len() == res[1].arrays.len()
            && res[0]
                .arrays
                .iter()
                .zip(&res[1].arrays)
                .all(|(a, b)| a.1.data == b.1.data);
        if same {
            agree += 1;
        } else {
            bad.push(format!("{name}: fronts disagree"));
        }
    }
    eprintln!(
        "exact vs hardware: final {}/{n}, original {}/{n}; fronts agree {agree}/{n}; llosim.py outputs compared: {oracle_checked}",
        exact[0], exact[1]
    );
    assert_eq!((exact, agree), ([26, 26], 30), "corpus totals");
    assert!(bad.is_empty(), "{bad:?}");
}

/// sbr.rel encodings decoded from the serialized executables equal tools/tpubranch.py's
/// (probes/binary FINDINGS "Branch delay"), including remote_sig, whose kernels are called in
/// the opposite of dump order.
#[test]
fn branch_encodings_from_executables() {
    use ptxv_tpu::exe::BranchEnc;
    let want: &[(&str, &[(u32, u8, i32)])] = &[
        ("vec_loop", &[(0x5, 6, -4)]),
        ("smem_loop", &[(0x18, 5, -11)]),
        ("remote_sig", &[(0x14, 5, 11), (0x20, 0, 9)]),
        (
            "grid_mm",
            &[
                (0x21, 0, 30),
                (0x23, 6, 14),
                (0x31, 6, 14),
                (0x42, 4, 99),
                (0x48, 6, 9),
                (0x7b, 6, 16),
                (0xb1, 6, -167),
            ],
        ),
        (
            "pred_when",
            &[
                (0x5, 0, 49),
                (0x8, 2, 4),
                (0x15, 5, 7),
                (0x1e, 6, 7),
                (0x38, 6, -55),
            ],
        ),
    ];
    for (name, brs) in want {
        let kd = KernelDir::open(&root().join("k").join(name)).unwrap();
        let exe = std::fs::read(kd.exe.as_ref().expect("executable")).unwrap();
        let listing = ptxv_tpu::exe::listing_name(&kd.final_path).unwrap();
        let got = ptxv_tpu::exe::branch_delays(&exe, &kd.llo_dir, &listing).unwrap();
        let want: std::collections::BTreeMap<u32, BranchEnc> = brs
            .iter()
            .map(|&(a, y, imm)| (a, BranchEnc { y, imm }))
            .collect();
        assert_eq!(got, want, "{name}");
    }
}

/// The calibrated numerics (tools/tpunum.py models, notes/numerics.md): the MXU and VPU kernels
/// become exact; kernels using EUP f32 functions without a bitwise model stop as unmodeled.
#[test]
fn corpus_calibrated() {
    let num = ptxv_tpu::num::calibrated();
    let mut names: Vec<String> = std::fs::read_dir(root().join("k"))
        .unwrap()
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .collect();
    names.sort();
    let (mut exact, mut unmodeled, mut n) = (0, Vec::new(), 0);
    let mut bad = Vec::new();
    for name in &names {
        let kd = KernelDir::open(&root().join("k").join(name)).unwrap();
        let mut outs = Vec::new();
        for front in [Front::Final, Front::Original] {
            let r = run_kernel(&kd, front, DmaPolicy::Eager, num).unwrap();
            let mm: Vec<String> = r.outputs.iter().map(|o| o.mismatch.to_string()).collect();
            eprintln!(
                "{name:16} {:8} {:8} exact={:?} mism=[{}] {}",
                r.front,
                r.status,
                r.exact,
                mm.join(","),
                r.error
            );
            n += 1;
            match (r.status.as_str(), r.exact) {
                ("ok", Some(true)) => exact += 1,
                ("Unimpl", _) => unmodeled.push(format!("{name}/{}", r.front)),
                _ => bad.push(format!("{name}/{}: {} {:?}", r.front, r.status, r.exact)),
            }
            outs.push(r.arrays);
        }
    }
    eprintln!("calibrated: exact {exact}/{n} (both fronts), unmodeled {unmodeled:?}");
    assert!(bad.is_empty(), "{bad:?}");
    // every kernel exact except trans_f32 (EUP f32 exp2/log2/tanh have no bitwise model)
    assert_eq!((exact, unmodeled.len()), (58, 2));
}
