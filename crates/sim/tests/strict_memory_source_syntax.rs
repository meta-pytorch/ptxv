/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ptxv_sim::strict_memory_source;
use serde_json::json;

fn module(body: &str) -> String {
    format!(
        ".version 9.0\n.target sm_103a\n.address_size 64\n.global .align 8 .u64 x[2] = {{0,0}};\n.visible .entry k() {{\n.reg .b64 %a<2>; .reg .b32 %r<2>; .reg .pred %p<2>;\n{body}\n}}\n"
    )
}

fn checked(body: &str) -> serde_json::Value {
    checked_source(&module(body))
}

fn checked_source(source: &str) -> serde_json::Value {
    let manifest = serde_json::from_value(json!({
        "name": "source-syntax", "description": "raw PTX syntax must be admitted exactly",
        "entry": "k", "grid": [1,1,1], "block": [1,1,1],
        "outputs": [], "buffers": [], "params": [], "variants": []
    }))
    .unwrap();
    serde_json::to_value(strict_memory_source::check(
        source.as_bytes(),
        b"k",
        &manifest,
        64,
        strict_memory_source::native_candidate,
    ))
    .unwrap()
}

#[test]
fn rejects_non_ptx_nop_with_ret_companion() {
    let rejected = checked("nop; ret;");
    assert_eq!(rejected["status"], "inconclusive", "{rejected}");
    assert_eq!(rejected["reason"]["kind"], "source_refused", "{rejected}");
    assert_eq!(checked("ret;")["status"], "safe");
}

#[test]
fn rejects_bare_address_subtraction_with_positive_offset_companion() {
    let rejected = checked("mov.u64 %a0,x; add.u64 %a0,%a0,8; ld.global.u32 %r0,[%a0-4]; ret;");
    assert_eq!(rejected["status"], "inconclusive", "{rejected}");
    assert_eq!(rejected["reason"]["kind"], "source_refused", "{rejected}");
    assert_eq!(
        checked("mov.u64 %a0,x; ld.global.u32 %r0,[%a0+4]; ret;")["status"],
        "safe"
    );
}

#[test]
fn operand_forms_are_validated_even_when_predicated_off() {
    for instruction in [
        "add.u64 %a0,x,4;",
        "add.u64 %a0,4,x;",
        "sub.u64 %a0,x,4;",
        "and.b64 %a0,x,7;",
        "or.b64 %a0,x,7;",
        "xor.b64 %a0,x,7;",
        "mul.lo.u64 %a0,x,2;",
        "cvt.u32.u64 %r0,x;",
        "setp.eq.u64 %p1,x,x;",
        "st.global.u64 [x],x;",
        "ld.global.u32 %r0,[0];",
        "st.global.u32 [0],0;",
        "add.u32 %r0,%tid.x,1;",
        "and.b32 %r0,%tid.x,1;",
        "mul.wide.u32 %a0,%tid.x,8;",
        "setp.eq.u32 %p1,%tid.x,0;",
        "st.global.u32 [x],%tid.x;",
        "ld.global.u32 %r0,[x+%a0];",
    ] {
        let body = format!("setp.eq.u32 %p0,0,1; @%p0 {instruction} ret;");
        let report = checked(&body);
        assert_eq!(report["status"], "inconclusive", "{instruction}: {report}");
        assert_eq!(
            report["reason"]["kind"], "source_refused",
            "{instruction}: {report}"
        );
    }
    assert_eq!(
        checked(
            "mov.u64 %a0,x; mov.u32 %r0,4; cvt.u64.u32 %a1,%r0; add.u64 %a0,%a0,%a1; ld.global.u32 %r1,[%a0]; ret;"
        )["status"],
        "safe"
    );
}

#[test]
fn header_pairs_are_validated_before_safe_even_for_ret_only_code() {
    for version in ["8.0", "8.7", "9.0"] {
        for target in ["sm_80", "sm_90", "sm_90a", "sm_100", "sm_100a", "sm_103a"] {
            let source = module("ret;")
                .replace(".version 9.0", &format!(".version {version}"))
                .replace(".target sm_103a", &format!(".target {target}"));
            let report = checked_source(&source);
            let supported = match version {
                "8.0" => matches!(target, "sm_80" | "sm_90" | "sm_90a"),
                "8.7" => target != "sm_103a",
                _ => true,
            };
            assert_eq!(
                report["status"],
                if supported { "safe" } else { "inconclusive" },
                "{version}/{target}: {report}"
            );
            if !supported {
                assert_eq!(report["reason"]["kind"], "source_refused");
            }
        }
    }
}

#[test]
fn entry_and_labels_do_not_change_global_symbol_resolution() {
    let entry_collision = module("ret;").replace(".u64 x[2]", ".u64 k[2]");
    assert_eq!(
        checked_source(&entry_collision)["reason"]["kind"],
        "source_refused"
    );
    assert_eq!(
        checked("x: ld.global.u32 %r0,[x]; ret;")["reason"]["kind"],
        "source_refused"
    );
    assert_eq!(checked("k: ret;")["status"], "safe");
}

#[test]
fn scalar_declarations_cannot_shadow_builtin_vector_names() {
    for builtin in ["tid", "ctaid", "ntid", "nctaid"] {
        let source = module(&format!(
            "mov.u32 %r0,%{builtin}.x; st.global.u32 [x],%r0; ret;"
        ))
        .replace(
            ".reg .b32 %r<2>;",
            &format!(".reg .b32 %r<2>; .reg .b32 %{builtin};"),
        );
        let rejected = checked_source(&source);
        assert_eq!(
            rejected["reason"]["kind"], "source_refused",
            "{builtin}: {rejected}"
        );
    }
    assert_eq!(
        checked("mov.u32 %r0,%tid.x; st.global.u32 [x],%r0; ret;")["status"],
        "safe"
    );
    let scalar_lane = module("mov.u32 %laneid,7; st.global.u32 [x],%laneid; ret;")
        .replace(".reg .b32 %r<2>;", ".reg .b32 %r<2>; .reg .b32 %laneid;");
    assert_eq!(
        checked_source(&scalar_lane)["reason"]["kind"],
        "source_refused"
    );
}

#[test]
fn reserved_nonregister_names_cannot_be_admitted() {
    for name in ["_", "$", "WARP_SZ"] {
        for source in [
            module("ret;").replace(".u64 x[2]", &format!(".u64 {name}[2]")),
            module(&format!("{name}: ret;")),
        ] {
            assert_eq!(
                checked_source(&source)["reason"]["kind"],
                "source_refused",
                "{source}"
            );
        }
        let source = module(&format!(
            "mov.u32 %{name},7; st.global.u32 [x],%{name}; ret;"
        ))
        .replace(
            ".reg .b32 %r<2>;",
            &format!(".reg .b32 %r<2>; .reg .b32 %{name};"),
        );
        assert_eq!(checked_source(&source)["status"], "safe", "{source}");
    }
    assert_eq!(checked("ret;")["status"], "safe");
}

#[test]
fn oversized_global_alignment_is_refused_before_safe() {
    for exponent in 32..64 {
        let alignment = 1u64 << exponent;
        let source = module("ret;").replace(".align 8", &format!(".align {alignment}"));
        let rejected = checked_source(&source);
        assert_eq!(
            rejected["status"], "inconclusive",
            "alignment {alignment}: {rejected}"
        );
        assert_eq!(
            rejected["reason"]["kind"], "source_refused",
            "alignment {alignment}: {rejected}"
        );
    }
    for alignment in [8, 1u64 << 30, 1u64 << 31] {
        let source = module("ret;").replace(".align 8", &format!(".align {alignment}"));
        let accepted = checked_source(&source);
        assert_eq!(
            accepted["status"], "safe",
            "alignment {alignment}: {accepted}"
        );
    }
}
