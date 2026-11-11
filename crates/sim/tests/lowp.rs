/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! N-CVT-LP: the narrow-format cvt model against GB300, without a GPU at test time.
//! - probes/cvt_lp/sweep.json: hashes of GB300's outputs of every probe op over the cvt-sweep
//!   sets (all 2^16 16-bit sources / code pairs; 2^20 edge and shaped random f32 operands with
//!   random bits), recomputed from the model;
//! - probes/cvt_lp through both front-ends (the SASS one on ptxas's binary) against
//!   gb300_out.bin, bit for bit;
//! - the s2f6 dequant: the hash of GB200's `cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2`
//!   outputs over every (code, scale) pair in every lane position, recomputed from the model.

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use ptxv_case::Case;
use ptxv_sim::build_launch;
use ptxv_sim::lowp;
use ptxv_sim::lowp::sweep;
use ptxv_sim::machine::Machine;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn cvt_model_replays_gb300_sweep() {
    let dir = root().join("probes/cvt_lp");
    let ops = lowp::probe_ops(&std::fs::read_to_string(dir.join("kernel.ptx")).unwrap());
    assert_eq!(ops.len(), 51);
    let rec: BTreeMap<String, BTreeMap<String, (usize, String)>> =
        serde_json::from_str(&std::fs::read_to_string(dir.join("sweep.json")).unwrap()).unwrap();
    for s in sweep::sets() {
        let r = &rec[s.name];
        assert_eq!(r.len(), ops.len());
        for name in &ops {
            let (n, h) = &r[name];
            let m = sweep::model(&s, name);
            assert_eq!(*n, m.len());
            assert_eq!(
                *h,
                format!("{:016x}", sweep::hash(m.into_iter())),
                "set {} op {name}: model differs from GB300",
                s.name
            );
        }
    }
}

/// N-CVT-LP: the s2f6 dequant's model against GB200 (sm_100a, CUDA 13.1, 2026-11-08): 65,536
/// threads of `cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2`, two cvts each; thread i,
/// p = i % 256, q = i / 256, converts the code words x = {q ^ 0x96, p ^ 0x5a, q, p} (bytes, high
/// first) with the scale words y = {p ^ 0x2d, q ^ 0xc3, p, q}, so every lane position sees every
/// (code, scale) pair once with independently varying neighbours. The FNV-1a hash of the 131,072
/// results in thread order must match the GPU's.
#[test]
fn s2f6_model_replays_gb200_sweep() {
    let out = (0..65536u32).flat_map(|i| {
        let (p, q) = (i & 0xff, i >> 8);
        let x = (q ^ 0x96) << 24 | (p ^ 0x5a) << 16 | q << 8 | p;
        let y = (p ^ 0x2d) << 24 | (q ^ 0xc3) << 16 | p << 8 | q;
        [
            lowp::cvt_s2f6x2_scaled(x as u16, y as u16),
            lowp::cvt_s2f6x2_scaled((x >> 16) as u16, (y >> 16) as u16),
        ]
    });
    assert_eq!(
        format!("{:016x}", sweep::hash(out)),
        "b03fb8bf9d8e435d",
        "s2f6 model differs from GB200"
    );
}

fn run(case: &Case, k: ptxv_ir::Kernel) -> Vec<u8> {
    let launch = build_launch(case, &k).unwrap();
    let m = Machine::new(Arc::new(k), Arc::new(launch));
    let (m, _, ok) = ptxv_sim::explore::run_once(m);
    assert!(ok);
    m.outputs()
}

#[test]
fn cvt_lp_both_fronts_match_gb300() {
    let case = Case::load(root().join("probes/cvt_lp")).unwrap();
    let hw = std::fs::read(case.dir.join("gb300_out.bin")).unwrap();
    let ptx = case.variant_ptx("good").unwrap();
    let dir = std::env::temp_dir().join("ptxv-test-cvt_lp");
    std::fs::create_dir_all(&dir).unwrap();
    let (p, cubin) = (dir.join("k.ptx"), dir.join("k.cubin"));
    std::fs::write(&p, &ptx).unwrap();
    ptxv_sass::ptxas(&p, &cubin, &case.m.arch).unwrap();
    let w = |x: &[u8], i: usize| u32::from_le_bytes(x[4 * i..4 * i + 4].try_into().unwrap());
    for k in [
        ptxv_ptx::parse_kernel_for(
            &ptx,
            Some(&case.m.entry),
            case.m
                .ptxas_toolkit
                .as_deref()
                .and_then(ptxv_ptx::toolkit_version),
        )
        .unwrap(),
        ptxv_sass::load_cubin(&cubin, &case.m.entry).unwrap(),
    ] {
        let front = format!("{:?}", k.front);
        let got = run(&case, k);
        assert_eq!(got.len(), hw.len());
        let bad: Vec<(usize, u32, u32)> = (0..hw.len() / 4)
            .filter(|&i| w(&got, i) != w(&hw, i))
            .map(|i| (i, w(&got, i), w(&hw, i)))
            .take(8)
            .collect();
        assert!(bad.is_empty(), "{front} (word, sim, GB300): {bad:x?}");
    }
}
