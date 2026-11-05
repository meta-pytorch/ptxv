/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! N-MUFU: the MUFU / PTX approx model against GB300, without a GPU at test time.
//! - the replay sample of the exhaustive sweep (fixtures/mufu/samples.json, written by
//!   `ptxv mufu-sweep`): special inputs and 2^20 random inputs per f32 op, all inputs of the
//!   16-bit forms; the full 2^32 replay is an ignored test;
//! - fixtures/s_mufu (softmax / RMSNorm rows and raw-bit MUFUs, SASS front-end) and
//!   probes/mufu/approx_ptx (every PTX approx form, PTX front-end) against their recorded
//!   GB300 outputs, bit for bit.

use std::path::PathBuf;
use std::sync::Arc;

use ptxv_case::Case;
use ptxv_ir::MufuFmt;
use ptxv_sim::build_launch;
use ptxv_sim::machine::Machine;
use ptxv_sim::mufu_sample::*;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn samples() -> Vec<Sample> {
    let p = root().join("fixtures/mufu/samples.json");
    serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap()
}

#[test]
fn mufu_model_replays_gb300_sweep_sample() {
    let all = samples();
    for (name, ..) in OPS {
        assert!(all.iter().any(|s| s.op == *name), "no sample for {name}");
    }
    for s in &all {
        let (_, fmt, _) = op(&s.op).unwrap();
        if fmt == MufuFmt::F32 {
            assert!(s.specials.len() > 500 && s.random_hashes.len() == RANDOM_N / RANDOM_BLOCK);
            s.check_with(false).unwrap();
        } else {
            s.check_full(0..1).unwrap();
        }
    }
}

#[test]
#[ignore = "slow (minutes): replays all 2^32 inputs of every f32 op"]
fn mufu_model_replays_full_gb300_sweep() {
    for s in samples() {
        let (_, fmt, _) = op(&s.op).unwrap();
        assert_eq!(s.chunk_hashes.len() as u64, total(fmt).div_ceil(CHUNK));
        s.check_full(0..s.chunk_hashes.len()).unwrap();
    }
}

/// One run of `kernel` on the case's launch: its output bytes.
fn run(case: &Case, k: ptxv_ir::Kernel) -> Vec<u8> {
    let launch = build_launch(case, &k).unwrap();
    let m = Machine::new(Arc::new(k), Arc::new(launch));
    let (m, _, ok) = ptxv_sim::explore::run_once(m);
    assert!(ok);
    m.outputs()
}

fn diff(a: &[u8], b: &[u8]) -> Vec<(usize, u32, u32)> {
    let w = |x: &[u8], i: usize| u32::from_le_bytes(x[4 * i..4 * i + 4].try_into().unwrap());
    (0..a.len().min(b.len()) / 4)
        .filter(|&i| w(a, i) != w(b, i))
        .map(|i| (i, w(a, i), w(b, i)))
        .take(8)
        .collect()
}

#[test]
fn s_mufu_sass_matches_gb300() {
    let case = Case::load(root().join("fixtures/s_mufu")).unwrap();
    let k = ptxv_sass::load_cubin(&case.dir.join("kernel.cubin"), &case.m.entry).unwrap();
    let got = run(&case, k);
    let hw = std::fs::read(case.dir.join("gb300_out.bin")).unwrap();
    assert_eq!(got.len(), hw.len());
    assert!(got == hw, "(word, sim, GB300): {:x?}", diff(&got, &hw));
}

#[test]
fn approx_ptx_matches_gb300() {
    let case = Case::load(root().join("probes/mufu/approx_ptx")).unwrap();
    let k =
        ptxv_ptx::parse_kernel(&case.variant_ptx("good").unwrap(), Some(&case.m.entry)).unwrap();
    let got = run(&case, k);
    let hw = std::fs::read(case.dir.join("gb300_out.bin")).unwrap();
    assert_eq!(got.len(), hw.len());
    assert!(got == hw, "(word, sim, GB300): {:x?}", diff(&got, &hw));
}

/// MUFU.RCP64H / RSQ64H: `rcp/rsqrt.approx.ftz.f64` through both front-ends (the SASS one
/// on ptxas's binary) against GB300's output.
#[test]
fn f64_approx_matches_gb300() {
    let case = Case::load(root().join("probes/mufu/f64_ptx")).unwrap();
    let hw = std::fs::read(case.dir.join("gb300_out.bin")).unwrap();
    let ptx = case.variant_ptx("good").unwrap();
    let dir = std::env::temp_dir().join("ptxv-test-f64_ptx");
    std::fs::create_dir_all(&dir).unwrap();
    let (p, cubin) = (dir.join("k.ptx"), dir.join("k.cubin"));
    std::fs::write(&p, &ptx).unwrap();
    ptxv_sass::ptxas(&p, &cubin, &case.m.arch).unwrap();
    for k in [
        ptxv_ptx::parse_kernel(&ptx, Some(&case.m.entry)).unwrap(),
        ptxv_sass::load_cubin(&cubin, &case.m.entry).unwrap(),
    ] {
        let front = format!("{:?}", k.front);
        let got = run(&case, k);
        assert_eq!(got.len(), hw.len());
        assert!(
            got == hw,
            "{front} (word, sim, GB300): {:x?}",
            diff(&got, &hw)
        );
    }
}
