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
//!   gb300_out.bin, bit for bit.

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
        ptxv_ptx::parse_kernel(&ptx, Some(&case.m.entry)).unwrap(),
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
