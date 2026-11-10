/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Explorer over the TPU machine: fixtures/tpu_remote_signal races (the receiver reads the
//! buffer after the flag signal, which may overtake the remote DMA); the fixed variant (waits on
//! the receive semaphore first) is clean on every schedule and reproduces the expected outputs.

use std::path::PathBuf;
use std::sync::Arc;

use ptxv_sim::model::explore;
use ptxv_tpu::corpus::KernelDir;
use ptxv_tpu::model::TpuModel;
use ptxv_tpu::num::PLACEHOLDER;
use ptxv_tpu_ir::Front;

fn fixture(name: &str) -> KernelDir {
    let k = if name.ends_with("_fixed") {
        "k/remote_sig_fixed"
    } else {
        "k/remote_sig"
    };
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures")
        .join(name)
        .join(k);
    KernelDir::open(&d).expect("fixture")
}

fn model(kd: &KernelDir) -> TpuModel {
    model_front(kd, Front::Final)
}

fn model_front(kd: &KernelDir, f: Front) -> TpuModel {
    let k = Arc::new(kd.lift(f));
    TpuModel::new(kd.machine(k, &PLACEHOLDER).expect("machine"))
}

#[test]
fn remote_signal_races() {
    let kd = fixture("tpu_remote_signal");
    // reduced search: the race is reported
    for f in [Front::Final, Front::Original] {
        let m = model_front(&kd, f);
        assert!(m.reduced());
        let v = explore(m, 200_000, 4);
        assert!(
            v.complete && v.findings.iter().any(|x| x.kind == "race"),
            "{f:?}"
        );
    }
    // every completion order: also both early-read values
    for f in [Front::Final, Front::Original] {
        let v = explore(model_front(&kd, f).without_reduction(), 200_000, 4);
        for f in &v.findings {
            eprintln!("{} {}: {}\n  trace: {:?}", f.kind, f.sig, f.msg, f.trace);
        }
        eprintln!(
            "states {} terminal {} outputs {} complete {}",
            v.states, v.terminal_states, v.distinct_outputs, v.complete
        );
        assert!(v.complete);
        let race = v
            .findings
            .iter()
            .find(|f| f.kind == "race")
            .expect("race reported");
        assert!(race.msg.contains("core 1"), "{}", race.msg);
        assert!(
            v.distinct_outputs >= 2,
            "the early read sees either the sentinel or the DMA'd rows"
        );
        assert!(
            v.findings.iter().all(|f| f.kind == "race"),
            "{:?}",
            v.findings.iter().map(|f| &f.sig).collect::<Vec<_>>()
        );
    }
}

#[test]
fn remote_signal_fixed_is_clean() {
    let kd = fixture("tpu_remote_signal_fixed");
    assert_eq!(kd.lift(Front::Final).delay_source, "binary");
    for f in [Front::Final, Front::Original] {
        let v = explore(model_front(&kd, f), 200_000, 4);
        for f in &v.findings {
            eprintln!("{} {}: {}\n  trace: {:?}", f.kind, f.sig, f.msg, f.trace);
        }
        eprintln!(
            "states {} terminal {} outputs {} complete {}",
            v.states, v.terminal_states, v.distinct_outputs, v.complete
        );
        assert!(v.complete && v.findings.is_empty());
        assert_eq!(v.distinct_outputs, 1);
        // the one outcome equals the expected outputs
        let mut want = Vec::new();
        for c in 0..2 {
            for oi in 0..2 {
                let (a, _) = kd.expected(c, oi).unwrap();
                for x in &a.data {
                    want.extend_from_slice(&x.to_le_bytes());
                }
            }
        }
        assert_eq!(v.outputs[0], want);
    }
}

/// Every probe-corpus kernel except remote_sig is race- and deadlock-free on every schedule,
/// with one outcome.
#[test]
fn corpus_kernels_clean() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tpu_corpus/k");
    let mut names: Vec<String> = std::fs::read_dir(&root)
        .unwrap()
        .filter_map(|e| e.ok()?.file_name().into_string().ok())
        .collect();
    names.sort();
    let mut bad = Vec::new();
    for n in &names {
        let kd = KernelDir::open(&root.join(n)).unwrap();
        let v = explore(model(&kd), 200_000, 4);
        let sigs: Vec<&str> = v.findings.iter().map(|f| f.sig.as_str()).collect();
        eprintln!(
            "{n:14} states {:6} terminal {:3} outputs {} complete {} {:.2}s findings {sigs:?}",
            v.states, v.terminal_states, v.distinct_outputs, v.complete, v.seconds
        );
        let clean = v.complete && v.findings.is_empty() && v.distinct_outputs == 1;
        if clean == (n == "remote_sig") {
            bad.push(n.clone());
        }
    }
    assert!(bad.is_empty(), "{bad:?}");
}

/// Mutation: dropping the DMA waits of the pipelined kernels must surface a race (the vector
/// loads read buffers a DMA may still be writing) or the kernel's exit check.
#[test]
fn dropped_waits_are_reported() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/tpu_corpus/k");
    for n in ["grid_ew", "grid_mm", "prefetch", "hbm_dma", "remote_ring"] {
        let kd = KernelDir::open(&root.join(n)).unwrap();
        let mut k = kd.lift(Front::Final);
        let mut dropped = 0;
        for b in &mut k.bundles {
            for i in &mut b.insts {
                if i.mn == "dma.done.wait" {
                    i.op = ptxv_tpu_ir::Op::Nop;
                    dropped += 1;
                }
            }
        }
        let m = TpuModel::new(kd.machine(Arc::new(k), &PLACEHOLDER).unwrap());
        let v = explore(m, 20_000, 4);
        let kinds: Vec<String> = v
            .findings
            .iter()
            .map(|f| format!("{} {}", f.kind, f.sig))
            .collect();
        eprintln!(
            "{n:12} dropped {dropped} waits: {} states {:.1}s {kinds:?}",
            v.states, v.seconds
        );
        assert!(
            dropped > 0
                && v.findings
                    .iter()
                    .any(|f| f.kind == "race" || f.sig.starts_with("exit-check")),
            "{n}"
        );
    }
}

/// Semaphore checks (notes/sync.md): a software add that overflows int32 halts
/// (TPU-S-SEM-OVF-TRAP); a flag left nonzero at exit is reported even with the kernel's exit
/// check compiled out (TPU-S-SEM-PERSIST); an index outside every sync-flag allocation is
/// reported (TPU-S-SEM-INDEX).
#[test]
fn semaphore_findings() {
    use ptxv_tpu_ir::Op;
    use ptxv_tpu_ir::Val;
    let run = |k: ptxv_tpu_ir::Kernel, kd: &KernelDir| {
        let v = explore(
            TpuModel::new(kd.machine(Arc::new(k), &PLACEHOLDER).unwrap()),
            50_000,
            4,
        );
        let sigs: Vec<String> = v
            .findings
            .iter()
            .map(|f| format!("{}: {}", f.sig, f.msg))
            .collect();
        eprintln!("{sigs:#?}");
        v
    };
    // overflow: the barrier decrement (vsyncadd [#allocation8], -1) becomes +INT32_MAX; with the
    // peer's +1 either before or after, some add overflows on every schedule.
    let kd = fixture("tpu_remote_signal_fixed");
    let mut k = kd.lift(Front::Final);
    let b = k.bundles.iter_mut().find(|b| b.addr == 0x11).unwrap();
    let Op::SyncAdd { val, .. } = &mut b.insts[0].op else {
        panic!("0x11 is vsyncadd")
    };
    *val = Val::Imm(i32::MAX as u32);
    let v = run(k, &kd);
    assert!(
        v.findings
            .iter()
            .any(|f| f.sig.starts_with("sem-overflow:core 0@0x11"))
    );
    assert!(
        v.findings
            .iter()
            .any(|f| f.sig.starts_with("sem-overflow:remote"))
    );
    // unbalanced at exit with the exit checks removed (disable_semaphore_checks): the
    // receiver's recv-semaphore decrement (fixed listing 0x25) and every shalt.err dropped.
    let kd = fixture("tpu_remote_signal_fixed");
    let mut k = kd.lift(Front::Final);
    for b in &mut k.bundles {
        for i in &mut b.insts {
            if matches!(i.op, Op::HaltErr { .. }) || (b.addr == 0x25 && i.mn == "vsyncadd") {
                i.op = Op::Nop;
            }
        }
    }
    let v = run(k, &kd);
    assert!(
        v.findings
            .iter()
            .any(|f| f.sig.starts_with("sem-unbalanced"))
    );
    assert!(v.findings.iter().all(|f| !f.sig.starts_with("exit-check")));
    // out-of-range semaphore index
    let kd = fixture("tpu_remote_signal_fixed");
    let mut k = kd.lift(Front::Final);
    let b = k.bundles.iter_mut().find(|b| b.addr == 0x11).unwrap();
    let Op::SyncAdd { flag, .. } = &mut b.insts[0].op else {
        panic!()
    };
    *flag = Val::Imm(5000);
    let v = run(k, &kd);
    assert!(v.findings.iter().any(|f| f.sig.starts_with("sem-index")));
}

/// "Wait for more than can arrive": the receiver waits for 4096 granules on a semaphore that
/// gets 2048 (fixed listing 0x24 edited): reported as a deadlock as soon as nothing that could
/// still raise the flag is in flight.
#[test]
fn wait_for_more_than_can_arrive() {
    use ptxv_tpu_ir::Op;
    use ptxv_tpu_ir::Val;
    let kd = fixture("tpu_remote_signal_fixed");
    let mut k = kd.lift(Front::Final);
    let b = k.bundles.iter_mut().find(|b| b.addr == 0x24).unwrap();
    let Op::Wait { val, .. } = &mut b.insts[0].op else {
        panic!("0x24 is dma.done.wait")
    };
    *val = Val::Imm(4096);
    let m = TpuModel::new(kd.machine(Arc::new(k), &PLACEHOLDER).unwrap());
    let v = explore(m, 50_000, 4);
    let msgs: Vec<String> = v
        .findings
        .iter()
        .map(|f| format!("{} {}", f.sig, f.msg))
        .collect();
    eprintln!("{msgs:#?}");
    assert!(
        v.findings
            .iter()
            .any(|f| f.kind == "deadlock" && f.msg.contains("more than can arrive"))
    );
}
