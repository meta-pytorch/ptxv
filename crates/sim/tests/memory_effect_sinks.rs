/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use ptxv_case::Case;
use ptxv_sim::build_launch;
use ptxv_sim::explore::explore;
use ptxv_sim::machine::Machine;

fn check(body: &str) -> ptxv_sim::explore::Verdict {
    let ptx = format!(
        ".version 8.8\n.target sm_103a\n.address_size 64\n\
         .visible .entry k() {{\n\
         .reg .b32 %r;\n.reg .b32 %a;\n.shared .align 4 .b8 data[4];\n\
         {body}\nret;\n}}"
    );
    let kernel = ptxv_ptx::parse_kernel(&ptx, Some("k")).unwrap();
    run_kernel(kernel)
}

fn make_machine(kernel: ptxv_ir::Kernel) -> Machine {
    let case = Case {
        dir: std::path::PathBuf::new(),
        m: serde_json::from_value(serde_json::json!({
            "name": "memory-effect-sinks", "description": "store operand roles",
            "entry": "k", "grid": [1,1,1], "block": [1,1,1],
            "buffers": [], "outputs": [], "params": [], "variants": []
        }))
        .unwrap(),
    };
    let launch = build_launch(&case, &kernel).unwrap();
    Machine::new(Arc::new(kernel), Arc::new(launch))
}

fn run_kernel(kernel: ptxv_ir::Kernel) -> ptxv_sim::explore::Verdict {
    explore(make_machine(kernel), 100)
}

#[test]
fn undefined_store_address_is_a_sink_when_it_aliases_deferred_data() {
    let verdict = check("st.shared.b32 [%r], %r;");
    assert!(verdict.complete, "{:?}", verdict.findings);
    assert!(
        verdict
            .findings
            .iter()
            .any(|finding| finding.sig.starts_with("uninit-reg:")),
        "undefined address escaped: {:?}",
        verdict.findings
    );
}

#[test]
fn unobserved_undefined_store_data_can_be_deferred() {
    let verdict = check("mov.b32 %a, 0; st.shared.b32 [%a], %r;");
    assert!(verdict.complete, "{:?}", verdict.findings);
    assert_eq!(verdict.classes(), ["clean"]);
}

fn sass_kernel(instructions: &[&str]) -> ptxv_ir::Kernel {
    let mut kernel = ptxv_ptx::parse_kernel(
        ".version 8.8\n.target sm_103a\n.address_size 64\n.visible .entry k() { .shared .align 4 .b8 data[4]; ret; }",
        Some("k"),
    ).unwrap();
    let template = kernel.code[0].clone();
    kernel.front = ptxv_ir::Front::Sass;
    kernel.nregs[ptxv_ir::File::R as usize] = 2;
    kernel.code = instructions
        .iter()
        .map(|source| ptxv_ir::Inst {
            op: ptxv_sass::lower_text(source, None).unwrap(),
            ..template.clone()
        })
        .collect();
    kernel.code.push(ptxv_ir::Inst {
        op: ptxv_ir::Op::Exit,
        ..template
    });
    kernel
}

fn check_sass(instructions: &[&str]) -> ptxv_sim::explore::Verdict {
    run_kernel(sass_kernel(instructions))
}

#[test]
fn physical_store_address_is_not_deferred_as_data() {
    let verdict = check_sass(&["STS [R0], R0;"]);
    assert!(verdict.complete, "{:?}", verdict.findings);
    assert!(
        verdict
            .findings
            .iter()
            .any(|finding| finding.sig.starts_with("uninit-reg:")),
        "{:?}",
        verdict.findings
    );
}

#[test]
fn narrow_data_does_not_narrow_the_aliased_address_read() {
    let prefix = sass_kernel(&["SHF.L.U32 R0, R0, 0x8, RZ;"]);
    let (machine, _, completed) = ptxv_sim::explore::run_once(make_machine(prefix));
    assert!(completed);
    assert_eq!(
        machine.reg_umask(
            0,
            ptxv_ir::Reg {
                file: ptxv_ir::File::R,
                idx: 0
            }
        ),
        0xffff_ff00
    );
    let verdict = check_sass(&["SHF.L.U32 R0, R0, 0x8, RZ;", "STS.U8 [R0], R0;"]);
    assert!(verdict.complete, "{:?}", verdict.findings);
    assert!(
        verdict
            .findings
            .iter()
            .any(|finding| finding.sig.starts_with("uninit-reg:")),
        "{:?}",
        verdict.findings
    );
    let defined_address = check_sass(&[
        "MOV R1, 0x0;",
        "SHF.L.U32 R0, R0, 0x8, RZ;",
        "STS.U8 [R1], R0;",
    ]);
    assert_eq!(
        defined_address.classes(),
        ["clean"],
        "{:?}",
        defined_address.findings
    );
}
