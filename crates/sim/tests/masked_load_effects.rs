/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use ptxv_case::Case;
use ptxv_ir::File;
use ptxv_ir::Op;
use ptxv_ir::Reg;
use ptxv_ir::Val;
use ptxv_sim::build_launch;
use ptxv_sim::explore::run_once;
use ptxv_sim::machine::Machine;

#[test]
fn parsed_masked_load_packs_each_half_and_preserves_unselected_registers() {
    for mask in [0x3cu8]
        .into_iter()
        .chain((1..=255u8).filter(|mask| *mask != 0x3c))
    {
        let mut kernel = ptxv_ptx::parse_kernel(
            ".version 8.8\n.target sm_103a\n.address_size 64\n.visible .entry k() { ret; }",
            Some("k"),
        )
        .unwrap();
        let template = kernel.code[0].clone();
        kernel.front = ptxv_ir::Front::Sass;
        kernel.nregs[File::R as usize] = 256;
        let low = if mask & 15 == 0 { "RZ" } else { "R8" };
        let high = if mask >> 4 == 0 { "RZ" } else { "R16" };
        let text = format!("LDG.E.ENL2.256 {high}, {low}, [R0.64], 0x{mask:x};");
        kernel.code = vec![
            ptxv_ir::Inst {
                op: ptxv_sass::lower_text(&text, None).unwrap(),
                ..template.clone()
            },
            ptxv_ir::Inst {
                op: Op::Exit,
                ..template
            },
        ];
        let case = Case {
            dir: Default::default(),
            m: serde_json::from_value(serde_json::json!({
                "name":"masked-load", "description":"masked load destination and word selection",
                "entry":"k", "grid":[1,1,1], "block":[1,1,1], "outputs":[],
                "buffers":[{"name":"input","dtype":"u32","shape":[8],"init":"zeros"}],
                "params":[], "variants":[]
            }))
            .unwrap(),
        };
        let launch = build_launch(&case, &kernel).unwrap();
        let mut machine = Machine::new(Arc::new(kernel), Arc::new(launch));
        machine.wr(
            0,
            Reg {
                file: File::R,
                idx: 0,
            },
            machine.launch.allocs[0].base,
            2,
        );
        for index in 0..8usize {
            let value = 0xa500_0000u32 + index as u32;
            for (byte, value) in value.to_le_bytes().into_iter().enumerate() {
                *machine.gmem[0].bytes.get_mut(index * 4 + byte) = value;
            }
        }
        for base in [8, 16] {
            for offset in 0..4 {
                machine.wr(
                    0,
                    Reg {
                        file: File::R,
                        idx: base + offset,
                    },
                    0xcafe_babe,
                    1,
                );
            }
        }
        let (machine, findings, complete) = run_once(machine);
        assert!(complete && findings.is_empty(), "{text}: {findings:?}");
        for (half, base) in [(0, 8), (1, 16)] {
            let selected: Vec<_> = (0..4)
                .filter(|index| mask & (1 << (4 * half + index)) != 0)
                .collect();
            for offset in 0..4 {
                let expected = selected.get(offset).map_or(0xcafe_babeu32, |word| {
                    0xa500_0000 + (4 * half + word) as u32
                });
                let actual = machine
                    .rd(
                        0,
                        &Val::Reg(
                            Reg {
                                file: File::R,
                                idx: base + offset as u16,
                            },
                            1,
                        ),
                    )
                    .unwrap();
                assert_eq!(actual, expected as u64, "{text}: R{}", base + offset as u16);
            }
        }
    }
}
