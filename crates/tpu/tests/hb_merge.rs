/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Soundness of state merging: two schedules reach the same data state with different
//! happens-before, and only one of them races. The explorer must not merge them.
//!
//! One core: issue a DMA (HBM -> VMEM row 64, flag 16); `vsyncmov 16` (acquires whatever the
//! flag's clock holds); pop the value and overwrite it (the data state no longer shows whether
//! the DMA had landed); a visible no-op `vsyncadd 17, 0`; then `vld` row 64 without a wait.
//! - DMA completes before the vsyncmov: the core acquired the DMA's clock -> the vld is ordered.
//! - DMA completes between the vsyncmov and the no-op: same data state at the no-op, but no
//!   happens-before -> the vld races with the DMA's write.
//! Without the clocks and the race shadow in the fingerprint the two states are equal (this
//! test fails on that assertion when they are left out).

use std::collections::HashMap;
use std::sync::Arc;

use ptxv_sim::model::Model;
use ptxv_sim::model::explore;
use ptxv_tpu::machine::TpuMachine;
use ptxv_tpu::model::Act;
use ptxv_tpu::model::TpuModel;
use ptxv_tpu::num::PLACEHOLDER;
use ptxv_tpu_ir::*;

fn inst(op: Op, def: Option<Reg>, mn: &str) -> Inst {
    Inst {
        guard: None,
        def,
        op,
        mn: mn.into(),
        text: mn.into(),
    }
}

fn bundle(addr: u32, insts: Vec<Inst>) -> Bundle {
    Bundle {
        insts,
        addr,
        depth: 0,
    }
}

#[test]
fn equal_data_different_happens_before_not_merged() {
    let s = |i| Reg {
        file: File::S,
        idx: i,
    };
    let v = |i| Reg {
        file: File::V,
        idx: i,
    };
    let decl = OperandDecl {
        index: 0,
        space: "hbm".into(),
        dtype: "f32".into(),
        shape: vec![8, 128],
        output: false,
    };
    let k = Kernel {
        name: "hb_merge".into(),
        front: Front::Final,
        bundles: vec![
            bundle(
                0,
                vec![inst(
                    Op::Operand { index: 0 },
                    Some(s(0)),
                    "inlined_call_operand.hbm",
                )],
            ),
            bundle(
                1,
                vec![inst(
                    Op::Dma {
                        kind: DmaKind {
                            src: Space::Hbm,
                            dst: Space::Vmem,
                        },
                        src: Val::Reg(s(0)),
                        dst: Val::Imm(64 << 4),
                        size: Val::Imm(128),
                        flag: Val::Imm(16),
                        stride: None,
                    },
                    None,
                    "dma.hbm_to_vmem",
                )],
            ),
            bundle(
                2,
                vec![inst(Op::SyncMov { flag: Val::Imm(16) }, None, "vsyncmov")],
            ),
            bundle(3, vec![inst(Op::Pop(Fifo::Sfrf), Some(s(1)), "vpop.sfrf")]),
            bundle(4, vec![inst(Op::SMov(Val::Imm(0)), Some(s(1)), "smov")]),
            bundle(
                5,
                vec![inst(
                    Op::SyncAdd {
                        flag: Val::Imm(17),
                        val: Val::Imm(0),
                    },
                    None,
                    "vsyncadd",
                )],
            ),
            bundle(
                6,
                vec![inst(
                    Op::VLd {
                        addr: Addr {
                            terms: vec![Term::Imm(64)],
                        },
                        sm: 0xff,
                        stride: None,
                    },
                    Some(v(0)),
                    "vld",
                )],
            ),
        ],
        allocs: Vec::new(),
        operands: vec![decl.clone()],
        unlowered: Vec::new(),
        delay_source: "none".into(),
        assumptions: Vec::new(),
        notes: Vec::new(),
    };
    let mut m = TpuMachine::new(Arc::new(k), 1, HashMap::from([((0, 0), 0)]), &PLACEHOLDER);
    m.place(0, &decl, &[7u32; 1024], None).unwrap();
    let m0 = TpuModel::new(m);
    // the two histories, action by action (each Core action runs to the next visible bundle)
    let run = |acts: &[Act]| {
        let mut x = m0.clone();
        for a in acts {
            assert!(x.enabled().contains(a), "{a:?} enabled");
            let f = x.step(a);
            assert!(f.is_empty(), "{f:?}");
        }
        x
    };
    let a = run(&[Act::Core(0), Act::Dma(0), Act::Core(0), Act::Core(0)]);
    let b = run(&[Act::Core(0), Act::Core(0), Act::Dma(0), Act::Core(0)]);
    // same data state (pc, registers, memories, flags, nothing in flight) ...
    assert_eq!(a.m.cores[0].pc, b.m.cores[0].pc);
    assert_eq!(a.m.cores[0].flags, b.m.cores[0].flags);
    assert!(a.m.pending.is_empty() && b.m.pending.is_empty());
    assert_eq!(a.outputs(), b.outputs());
    // ... different happens-before, so different states
    assert_ne!(
        a.fingerprint(),
        b.fingerprint(),
        "states with different clocks must not merge"
    );
    // only the second history races on the vld
    let race = |mut x: TpuModel| x.step(&Act::Core(0)).iter().any(|f| f.kind == "race");
    assert!(!race(a.clone()));
    assert!(race(b.clone()));
    let v = explore(m0, 10_000, 1);
    let sigs: Vec<&str> = v.findings.iter().map(|f| f.sig.as_str()).collect();
    eprintln!(
        "states {} terminal {} findings {sigs:?}",
        v.states, v.terminal_states
    );
    assert!(v.complete);
    assert!(
        sigs.iter()
            .any(|s| s.contains("core0@0x6") && s.contains("dma(core0@0x1)")),
        "{sigs:?}"
    );
}
