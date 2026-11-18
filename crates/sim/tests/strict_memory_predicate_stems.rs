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
        "name": "predicate-stems", "description": "predicate declarations use ordinary virtual names",
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
        panic!("candidate producer called for unsupported predicate declaration")
    });
    let report = serde_json::to_value(report).unwrap();
    assert_eq!(report["status"], "inconclusive", "{source}\n{report}");
    assert_eq!(report["reason"]["kind"], "source_refused", "{report}");
}

#[test]
fn refuses_predefined_and_other_noncanonical_predicate_stems_even_when_unused() {
    for stem in [
        "clock",
        "clock64",
        "globaltimer",
        "clusterid",
        "nclusterid",
        "cluster_ctarank",
        "envreg0",
        "smid",
        "lanemask_eq",
        "pm0",
        "q",
        "pred",
        "p_bad",
        "P0",
    ] {
        let declarations = format!(".reg .pred %{stem};");
        assert_refused(&source(&declarations, ""));
        let body = format!("setp.eq.u32 %{stem},0,1;\n@%{stem} st.global.u64 [out],123;");
        assert_refused(&source(&declarations, &body));
    }
}

#[test]
fn refuses_builtin_guard_repro_before_checking_the_candidate() {
    for stem in ["clusterid", "nclusterid"] {
        let declarations = format!(".reg .pred %{stem};");
        let body = format!(
            "setp.eq.u32 %{stem},1,1;\n@%{stem} st.global.u64 [out],123;\n@!%{stem} st.global.u64 [trap+8],123;"
        );
        assert_refused(&source(&declarations, &body));
    }
}

#[test]
fn accepts_decimal_predicate_stems_counted_declarations_and_ordinary_p7() {
    let mut examples: Vec<(String, String)> = ["p", "p0", "p7", "p00", "p12345"]
        .into_iter()
        .map(|stem| {
            (
                format!(".reg .pred %{stem};"),
                format!("setp.eq.u32 %{stem},1,1;\n@%{stem} st.global.u64 [out],123;"),
            )
        })
        .collect();
    examples.push((
        ".reg .pred %p<8>;".to_owned(),
        "setp.eq.u32 %p7,1,1;\n@%p7 st.global.u64 [out],123;".to_owned(),
    ));
    let manifest = manifest();
    for (declarations, body) in examples {
        let source = source(&declarations, &body);
        let report = strict_memory_source::check(
            source.as_bytes(),
            b"k",
            &manifest,
            64,
            strict_memory_source::native_candidate,
        );
        assert!(report.is_safe(), "{source}\n{report:?}");
    }
}

#[test]
fn preserves_scalar_declarations_with_the_same_spellings() {
    let manifest = manifest();
    for stem in ["clock", "clock64", "clusterid", "nclusterid", "q", "pred"] {
        let source = source(
            &format!(".reg .b64 %{stem};"),
            &format!("mov.u64 %{stem},123;\nst.global.u64 [out],%{stem};"),
        );
        let report = strict_memory_source::check(
            source.as_bytes(),
            b"k",
            &manifest,
            64,
            strict_memory_source::native_candidate,
        );
        assert!(report.is_safe(), "{source}\n{report:?}");
    }
}
