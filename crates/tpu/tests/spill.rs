/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Regression: register-allocator spill slots (`#allocationN_spill`, absent from the original
//! header) must not share VMEM with the kernel's operands. tools/llosim.py handed spill slots out
//! lazily from the first free row after the allocations, which is where the operands go; a spill
//! store then overwrote input rows (HIGHEST f32 matmul, A rows 72..79).

use std::collections::HashMap;
use std::sync::Arc;

use ptxv_tpu::machine::DmaPolicy;
use ptxv_tpu::machine::TpuMachine;
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

#[test]
fn spill_slots_do_not_overlap_operands() {
    let s = |i| Reg {
        file: File::S,
        idx: i,
    };
    let v = |i| Reg {
        file: File::V,
        idx: i,
    };
    let decl = |index, output| OperandDecl {
        index,
        space: "vmem".into(),
        dtype: "f32".into(),
        shape: vec![8, 128],
        output,
    };
    let at = |t: Term| Addr { terms: vec![t] };
    let bundles = vec![
        vec![
            inst(
                Op::Operand { index: 0 },
                Some(s(0)),
                "inlined_call_operand.vmem",
            ),
            inst(
                Op::Operand { index: 1 },
                Some(s(1)),
                "inlined_call_operand.vmem",
            ),
            inst(Op::VStv(Val::Imm(0xDEAD)), Some(v(0)), "vstv"),
        ],
        vec![inst(
            Op::VSt {
                addr: at(Term::Alloc(1)),
                sm: 0xff,
                vm: None,
                src: Val::Reg(v(0)),
            },
            None,
            "vst",
        )],
        vec![inst(
            Op::VLd {
                addr: at(Term::Reg(s(0))),
                sm: 0xff,
                stride: None,
            },
            Some(v(1)),
            "vld",
        )],
        vec![inst(
            Op::VSt {
                addr: at(Term::Reg(s(1))),
                sm: 0xff,
                vm: None,
                src: Val::Reg(v(1)),
            },
            None,
            "vst",
        )],
    ];
    let k = Kernel {
        name: "spill".into(),
        front: Front::Final,
        bundles: bundles
            .into_iter()
            .enumerate()
            .map(|(a, insts)| Bundle {
                insts,
                addr: a as u32,
                depth: 0,
            })
            .collect(),
        allocs: vec![
            Alloc {
                name: "0".into(),
                space: Space::Vmem,
                size: 4096,
                fixed: None,
                tag: String::new(),
            },
            Alloc {
                name: "3_spill".into(),
                space: Space::Vmem,
                size: 512,
                fixed: None,
                tag: String::new(),
            },
        ],
        operands: vec![decl(0, false), decl(1, true)],
        unlowered: Vec::new(),
        delay_source: "none".into(),
        assumptions: Vec::new(),
        notes: Vec::new(),
    };
    let mut m = TpuMachine::new(Arc::new(k), 1, HashMap::from([((0, 0), 0)]), &PLACEHOLDER);
    let input: Vec<u32> = (0..1024).collect();
    m.place(0, &decl(0, false), &input, None).unwrap();
    m.place(0, &decl(1, true), &[0; 1024], None).unwrap();
    m.run(DmaPolicy::Eager).unwrap();
    let out = m.placed.iter().find(|p| p.decl.output).unwrap().clone();
    assert_eq!(
        m.operand_words(&out).unwrap(),
        input,
        "the spill store must not reach the input operand"
    );
}
