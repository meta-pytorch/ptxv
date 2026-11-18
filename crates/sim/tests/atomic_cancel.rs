/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;
use std::sync::Arc;

use ptxv_ir::Front;
use ptxv_sim::machine::Machine;

fn machine(observer: bool) -> Machine {
    let observation = if observer {
        "ld.global.relaxed.cta.u32 %r2, [%rd0];"
    } else {
        "mov.u32 %r2, %r1;"
    };
    let source = format!(
        r#"
.version 8.8
.target sm_103a
.address_size 64
.visible .entry k(.param .u64 input) {{
  .reg .b32 %r<4>;
  .reg .b64 %rd<3>;
  ld.param.u64 %rd0, [input];
  mov.u32 %r0, %tid.x;
  atom.global.cta.add.u32 %r1, [%rd0], 1;
  sub.u32 %r1, %r1, %r1;
  {observation}
  mul.wide.u32 %rd1, %r0, 4;
  add.u64 %rd2, %rd0, %rd1;
  st.global.u32 [%rd2+4], %r2;
  ret;
}}
"#
    );
    let kernel = Arc::new(ptxv_ptx::parse_kernel(&source, Some("k")).unwrap());
    let manifest = serde_json::from_value(serde_json::json!({
        "name":"atomic-cancel-certificate", "description":"cancelled return with optional observer",
        "entry":"k", "grid":[1,1,1], "block":[64,1,1], "outputs":["x"],
        "buffers":[{"name":"x","dtype":"u32","shape":[65],"init":"zeros"}],
        "params":[{"kind":"ptr","buffer":"x"}], "variants":[]
    }))
    .unwrap();
    let case = ptxv_case::Case {
        dir: PathBuf::new(),
        m: manifest,
    };
    let launch = Arc::new(ptxv_sim::build_launch(&case, &kernel).unwrap());
    Machine::new(kernel, launch)
}

#[test]
fn cancelled_private_returns_can_complete_only_through_a_clean_certificate() {
    let initial = machine(false);
    assert!(ptxv_ir::atomic_cancel::certificate_candidate(
        Front::Ptx,
        &initial.k.code
    ));
    let certificate = ptxv_sim::wellsync::certify(initial.clone());
    assert!(certificate.reasons.is_empty(), "{:?}", certificate.reasons);
    let expected = certificate.verdict.unwrap();
    assert!(expected.complete);
    assert_eq!(expected.classes(), ["clean"]);
    let actual = ptxv_sim::explore::explore(initial, 1);
    assert!(actual.complete, "{actual:#?}");
    assert_eq!(actual.classes(), ["clean"]);
    assert_eq!(actual.outputs, expected.outputs);
    assert!(
        actual
            .notes
            .iter()
            .any(|note| note.contains("fresh initial-state certificate")),
        "{:?}",
        actual.notes
    );
}

#[test]
fn an_unordered_memory_observer_prevents_cancelled_return_completion() {
    let initial = machine(true);
    assert!(ptxv_ir::atomic_cancel::certificate_candidate(
        Front::Ptx,
        &initial.k.code
    ));
    let certificate = ptxv_sim::wellsync::certify(initial.clone());
    assert!(certificate.verdict.is_none());
    assert!(
        certificate
            .reasons
            .iter()
            .any(|reason| reason.starts_with("C-ORDER:")),
        "{:?}",
        certificate.reasons
    );
    let actual = ptxv_sim::explore::explore(initial, 1);
    assert!(!actual.complete);
    assert!(
        !actual
            .notes
            .iter()
            .any(|note| note.contains("fresh initial-state certificate"))
    );
}
