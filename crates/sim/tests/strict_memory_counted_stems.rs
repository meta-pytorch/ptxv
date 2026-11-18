/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ptxv_case::Manifest;
use ptxv_sim::strict_memory_frontend_launch;
use ptxv_sim::strict_memory_source;
use serde_json::json;

fn manifest() -> Manifest {
    serde_json::from_value(json!({
        "name": "counted-stems", "description": "counted declarations use unnumbered stems",
        "entry": "k", "grid": [1,1,1], "block": [1,1,1],
        "outputs": [], "buffers": [], "params": [], "variants": []
    }))
    .unwrap()
}

fn source(declarations: &str, body: &str) -> String {
    format!(
        ".version 9.0\n.target sm_103a\n.address_size 64\n.global .align 8 .u64 out[1]={{0}};\n.global .align 8 .u64 trap[1]={{0}};\n.visible .entry k(){{\n{declarations}\n{body}\nret;\n}}\n"
    )
}

fn assert_refused(source: &str) {
    let manifest = manifest();
    assert!(
        strict_memory_frontend_launch::build_program(source.as_bytes(), b"k", &manifest).is_none()
    );
    let report = strict_memory_source::check(source.as_bytes(), b"k", &manifest, 64, |_, _| {
        panic!("candidate producer called for unsupported counted declaration")
    });
    let report = serde_json::to_value(report).unwrap();
    assert_eq!(report["status"], "inconclusive", "{source}\n{report}");
    assert_eq!(report["reason"]["kind"], "source_refused", "{report}");
}

#[test]
fn rejects_numbered_counted_stems_even_when_unused_or_false_guarded() {
    for declaration in [
        ".reg .b32 %r0<2>;",
        ".reg .b32 %r7<2>;",
        ".reg .b64 %rd3<2>;",
        ".reg .pred %p7<2>;",
        ".reg .pred %p00<2>;",
    ] {
        assert_refused(&source(declaration, ""));
        assert_refused(&source(
            &format!("{declaration}\n.reg .pred %p;"),
            "setp.eq.u32 %p,0,1;\n@%p st.global.u64 [out],123;",
        ));
    }
}

#[test]
fn rejects_concatenated_counted_reference_repros_before_candidate() {
    assert_refused(&source(
        ".reg .pred %p7<2>;",
        "setp.eq.u32 %p70,1,1;\nsetp.ne.u32 %p71,1,1;\n@%p70 st.global.u64 [out],123;\n@%p71 st.global.u64 [trap+8],123;",
    ));
    assert_refused(&source(
        ".reg .b64 %r7<2>;",
        "mov.u64 %r70,123;\nmov.u64 %r71,17;\nadd.u64 %r70,%r70,%r71;\nst.global.u64 [out],%r70;",
    ));
}

#[test]
fn preserves_ordinary_digit_ending_scalar_and_predicate_names() {
    let manifest = manifest();
    for (declarations, body) in [
        (
            ".reg .b32 %r7;",
            "mov.u32 %r7,123;\nst.global.u32 [out],%r7;",
        ),
        (
            ".reg .b64 %rd3;",
            "mov.u64 %rd3,123;\nst.global.u64 [out],%rd3;",
        ),
        (
            ".reg .pred %p7;",
            "setp.eq.u32 %p7,1,1;\n@%p7 st.global.u64 [out],123;",
        ),
        (
            ".reg .pred %p00;",
            "setp.eq.u32 %p00,1,1;\n@%p00 st.global.u64 [out],123;",
        ),
    ] {
        let text = source(declarations, body);
        let report = strict_memory_source::check(
            text.as_bytes(),
            b"k",
            &manifest,
            64,
            strict_memory_source::native_candidate,
        );
        assert!(report.is_safe(), "{text}\n{report:?}");
    }
}

#[test]
fn preserves_canonical_counted_scalar_and_predicate_declarations() {
    let manifest = manifest();
    let text = source(
        ".reg .b32 %r<2>;\n.reg .b64 %rd<2>;\n.reg .pred %p<8>;",
        "mov.u32 %r0,123;\nmov.u32 %r1,%r0;\nmov.u64 %rd0,out;\nmov.u64 %rd1,%rd0;\nsetp.eq.u32 %p7,%r1,123;\n@%p7 st.global.u64 [%rd1],123;",
    );
    let report = strict_memory_source::check(
        text.as_bytes(),
        b"k",
        &manifest,
        64,
        strict_memory_source::native_candidate,
    );
    assert!(report.is_safe(), "{text}\n{report:?}");
}
