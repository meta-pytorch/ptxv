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

fn execute(kernel: ptxv_ir::Kernel) -> Machine {
    let case = Case {
        dir: std::path::PathBuf::new(),
        m: serde_json::from_value(serde_json::json!({
            "name": "memory-payload", "description": "instruction payload composition",
            "entry": "k", "grid": [1,1,1], "block": [1,1,1],
            "buffers": [], "outputs": [], "params": [], "variants": []
        }))
        .unwrap(),
    };
    let launch = build_launch(&case, &kernel).unwrap();
    let (machine, findings, completed) =
        run_once(Machine::new(Arc::new(kernel), Arc::new(launch)));
    assert!(completed, "{findings:?}");
    assert!(findings.is_empty(), "{findings:?}");
    machine
}

#[test]
fn parsed_exchange_payload_roundtrips_selected_words() {
    for (bits, value) in [(32, 0x89ab_cdefu64), (64, 0x0123_4567_89ab_cdef)] {
        let ptx = format!(
            ".version 8.8\n.target sm_103a\n.address_size 64\n\
             .visible .entry k() {{\n\
             .reg .b{bits} %value, %old, %out;\n\
             .shared .align 8 .b8 data[8];\n\
             mov.b{bits} %value, {value};\n\
             st.shared.b{bits} [data], 0;\n\
             atom.shared.exch.b{bits} %old, [data], %value;\n\
             ld.shared.b{bits} %out, [data];\nret;\n}}"
        );
        let kernel = ptxv_ptx::parse_kernel(&ptx, Some("k")).unwrap();
        let result = kernel
            .code
            .iter()
            .find_map(|instruction| match instruction.op {
                Op::Ld { d, .. } => Some(d),
                _ => None,
            })
            .unwrap();
        let previous = kernel
            .code
            .iter()
            .find_map(|instruction| match instruction.op {
                Op::Atom { d, .. } => d,
                _ => None,
            })
            .unwrap();
        let machine = execute(kernel);
        assert_eq!(machine.rd(0, &Val::Reg(result, 1)).unwrap(), value);
        assert_eq!(machine.rd(0, &Val::Reg(previous, 1)).unwrap(), 0);
    }
}

#[test]
fn parsed_physical_store_payload_roundtrips_a_register_pair() {
    let mut kernel = ptxv_ptx::parse_kernel(
        ".version 8.8\n.target sm_103a\n.address_size 64\n\
         .visible .entry k() { .shared .align 8 .b8 data[8]; ret; }",
        Some("k"),
    )
    .unwrap();
    let template = kernel.code[0].clone();
    kernel.front = ptxv_ir::Front::Sass;
    kernel.nregs[File::R as usize] = 8;
    kernel.code = [
        "MOV R0, 0x0;",
        "MOV R2, 0x89abcdef;",
        "MOV R3, 0x01234567;",
        "STS.64 [R0], RZ;",
        "STS.64 [R0], R2;",
        "LDS.64 R6, [R0];",
    ]
    .iter()
    .map(|source| ptxv_ir::Inst {
        op: ptxv_sass::lower_text(source, None).unwrap(),
        ..template.clone()
    })
    .collect();
    kernel.code.push(ptxv_ir::Inst {
        op: Op::Exit,
        ..template
    });
    let machine = execute(kernel);
    assert_eq!(
        machine
            .rd(
                0,
                &Val::Reg(
                    Reg {
                        file: File::R,
                        idx: 6
                    },
                    2
                )
            )
            .unwrap(),
        0x0123_4567_89ab_cdef
    );
}
