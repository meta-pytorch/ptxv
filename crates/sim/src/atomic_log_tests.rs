/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;
use std::sync::Arc;

use serde_json::json;

use crate::machine::Machine;

fn upper_component_machine(components: usize) -> Machine {
    let target = components - 1;
    let operands = if components == 2 {
        "{%f0,%f1}"
    } else {
        "{%f0,%f1,%f2,%f3}"
    };
    let text = format!(
        r#"
.version 8.8
.target sm_103a
.address_size 64
.visible .entry k(.param .u64 input) {{
  .reg .b32 %r0;
  .reg .b64 %rd0;
  .reg .f32 %f<4>;
  .reg .pred %p0;
  ld.param.u64 %rd0, [input];
  mov.u32 %r0, %ctaid.x;
  mov.f32 %f0, 0f00000000;
  mov.f32 %f1, 0f00000000;
  mov.f32 %f2, 0f00000000;
  mov.f32 %f3, 0f00000000;
  mov.f32 %f{target}, 0f60ad78ec;
  setp.eq.u32 %p0, %r0, 1;
  @%p0 mov.f32 %f{target}, 0f3f800000;
  setp.eq.u32 %p0, %r0, 2;
  @%p0 mov.f32 %f{target}, 0fe0ad78ec;
  red.global.add.v{components}.f32 [%rd0], {operands};
  ret;
}}
"#
    );
    let kernel = Arc::new(ptxv_ptx::parse_kernel(&text, Some("k")).unwrap());
    let manifest = serde_json::from_value(json!({
        "name":"red-log-component", "description":"upper vector component has distinct operands",
        "entry":"k", "grid":[3,1,1], "block":[1,1,1], "outputs":["x"],
        "buffers":[{"name":"x","dtype":"f32","shape":[components],"init":"zeros"}],
        "params":[{"kind":"ptr","buffer":"x"}], "variants":[]
    }))
    .unwrap();
    let case = ptxv_case::Case {
        dir: PathBuf::new(),
        m: manifest,
    };
    let launch = Arc::new(crate::build_launch(&case, &kernel).unwrap());
    Machine::new(kernel, launch)
}

fn check(components: usize) {
    let machine = upper_component_machine(components);
    let stateful = crate::explore::explore(machine.clone(), 100_000);
    let dpor = crate::dpor::explore_dpor(machine.clone(), 100_000);
    for (name, result) in [("stateful", &stateful), ("DPOR", &dpor)] {
        eprintln!(
            "v{components} upper component {name}: states={}, transitions={}, complete={}, outputs={}, classes={:?}",
            result.states,
            result.transitions,
            result.complete,
            result.outputs.len(),
            result.classes()
        );
        assert!(result.complete);
        assert_eq!(result.classes(), ["nondeterministic"]);
        assert_eq!(result.outputs.len(), 2);
    }
    let certificate = crate::wellsync::certify(machine);
    eprintln!(
        "v{components} upper component certificate: classes={:?}, reasons={:?}",
        certificate.verdict.as_ref().map(|v| v.classes()),
        certificate.reasons
    );
    let certified = certificate
        .verdict
        .expect("small unordered group is fully enumerated");
    assert_eq!(certified.classes(), stateful.classes());
    let mut actual = certified.outputs;
    let mut expected = stateful.outputs;
    actual.sort();
    expected.sort();
    assert_eq!(actual, expected);
}

#[test]
fn v2_certificate_logs_the_second_components_actual_operand() {
    check(2);
}

#[test]
fn v4_certificate_logs_the_fourth_components_actual_operand() {
    check(4);
}
