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
use ptxv_sim::build_launch;
use ptxv_sim::explore::run_once;
use ptxv_sim::machine::Machine;

#[test]
fn parsed_st256_payload_preserves_two_bases_and_source_word_order() {
    for high in [8, 10, 16] {
        let mut kernel = ptxv_ptx::parse_kernel(
            ".version 8.8\n.target sm_103a\n.address_size 64\n.visible .entry k() { ret; }",
            Some("k"),
        )
        .unwrap();
        let template = kernel.code[0].clone();
        kernel.front = ptxv_ir::Front::Sass;
        kernel.nregs[File::R as usize] = 64;
        kernel.code = vec![
            ptxv_ir::Inst {
                op: ptxv_sass::lower_text(&format!("STG.E.ENL2.256 [R0.64], R8, R{high};"), None)
                    .unwrap(),
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
                "name":"st256-payload", "description":"two-base store word order",
                "entry":"k", "grid":[1,1,1], "block":[1,1,1], "outputs":[],
                "buffers":[{"name":"output","dtype":"u32","shape":[8],"init":"zeros"}],
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
        for base in [8, high] {
            for offset in 0..4 {
                machine.wr(
                    0,
                    Reg {
                        file: File::R,
                        idx: base + offset,
                    },
                    0xa501_0000 + (base + offset) as u64,
                    1,
                );
            }
        }
        let (machine, findings, complete) = run_once(machine);
        assert!(complete && findings.is_empty(), "{high}: {findings:?}");
        for (word, register) in [8, high]
            .into_iter()
            .flat_map(|base| (0..4).map(move |offset| base + offset))
            .enumerate()
        {
            let actual = u32::from_le_bytes(std::array::from_fn(|byte| {
                *machine.gmem[0].bytes.get(4 * word + byte)
            }));
            assert_eq!(actual, 0xa501_0000 + register as u32, "{high}: word {word}");
        }
    }
}
