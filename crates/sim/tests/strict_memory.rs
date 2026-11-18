/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;

use ptxv_ir::Kernel;
use ptxv_memory_write::memory_safety;
use ptxv_sim::machine::Launch;
use ptxv_sim::strict_memory;
use ptxv_sim::strict_memory_admission;
use ptxv_sim::strict_memory_status::Status;
use ptxv_sim::strict_memory_status::status_for_certificate;
use serde_json::json;

fn input(body: &str, threads: u32, undefined: bool) -> (Kernel, Launch) {
    let text = format!(
        ".version 8.8\n.target sm_103a\n.address_size 64\n.visible .entry k() {{\n.reg .b64 %r<2>;\n{body}\nret;\n}}"
    );
    let kernel = ptxv_ptx::parse_kernel(&text, Some("k")).unwrap();
    let case = ptxv_case::Case {
        dir: PathBuf::new(),
        m: serde_json::from_value(json!({
            "name":"strict-memory", "description":"certificate acceptance controls",
            "entry":"k", "grid":[1,1,1], "block":[threads,1,1], "outputs":[],
            "buffers":[{"name":"x","dtype":"u64","shape":[2],"init":"zeros"}],
            "params":[], "variants":[]
        }))
        .unwrap(),
    };
    let mut launch = ptxv_sim::build_launch(&case, &kernel).unwrap();
    launch.allocs[0].base = 4096;
    launch.allocs[0].undef = undefined;
    launch.allocs[0].dead = undefined;
    (kernel, launch)
}

fn report(body: &str, threads: u32, undefined: bool, cap: usize) -> serde_json::Value {
    let (kernel, launch) = input(body, threads, undefined);
    serde_json::to_value(strict_memory::check(&kernel, &launch, cap)).unwrap()
}

#[test]
fn safe_results_require_complete_checked_certificates() {
    let (kernel, launch) = input(
        "st.global.u64 [4096], 0;\nld.global.u64 %r0, [4096];",
        1,
        true,
    );
    let result = strict_memory::check(&kernel, &launch, 100);
    assert!(result.is_safe(), "{result:?}");
    let value = serde_json::to_value(result).unwrap();
    assert_eq!(value["status"], "safe");
    assert_eq!(value["scope"], "typed_ir_ptx_constant_global_memory");
    assert_eq!(value["states"], 4);
    let parallel_reads = report("ld.global.u64 %r0, [4096];", 2, false, 100);
    assert_eq!(parallel_reads["status"], "safe", "{parallel_reads}");
    assert!(parallel_reads["states"].as_u64().unwrap() > 4);
}

#[test]
fn dead_uninitialized_reads_and_same_value_races_are_not_safe() {
    for value in [
        report("ld.global.u64 %r0, [4096];", 1, true, 100),
        report("st.global.u64 [4096], 0;", 2, false, 100),
    ] {
        assert_eq!(value["status"], "unsafe", "{value}");
        assert_eq!(value["reason"]["kind"], "bad_state");
    }
}

#[test]
fn out_of_bounds_and_misalignment_are_not_safe() {
    for address in [4112, 4097] {
        let value = report(&format!("ld.global.u64 %r0, [{address}];"), 1, false, 100);
        assert_eq!(value["status"], "unsafe", "{value}");
    }
}

#[test]
fn exhausted_search_and_unsupported_operations_are_inconclusive() {
    for cap in [0, 1] {
        let value = report("ld.global.u64 %r0, [4096];", 1, false, cap);
        assert_eq!(value["status"], "inconclusive", "{value}");
        assert_eq!(value["reason"]["kind"], "state_cap");
    }
    let value = report("add.u64 %r0, 1, 2;", 1, false, 100);
    assert_eq!(value["status"], "inconclusive", "{value}");
    assert_eq!(value["reason"]["kind"], "admission");
}

#[test]
fn accepting_boundary_rechecks_original_admission_and_certificate_closure() {
    let (kernel, launch) = input("ld.global.u64 %r0, [4096];", 1, false);
    let program = strict_memory_admission::project(&kernel, &launch).unwrap();
    let mut certificate = vec![memory_safety::initial(&program).unwrap()];
    assert_eq!(
        status_for_certificate(&kernel, &launch, &certificate),
        Status::Inconclusive
    );
    assert!(!strict_memory::verify_certificate(
        &kernel,
        &launch,
        &certificate
    ));
    while let Some(next) = memory_safety::successors(&program, certificate.last().unwrap()) {
        if next.is_empty() {
            break;
        }
        assert_eq!(next.len(), 1);
        certificate.push(next[0].clone());
    }
    assert!(strict_memory::verify_certificate(
        &kernel,
        &launch,
        &certificate
    ));
    assert_eq!(
        status_for_certificate(&kernel, &launch, &certificate),
        Status::Safe
    );
    assert_eq!(
        status_for_certificate(&kernel, &launch, &[]),
        Status::Inconclusive
    );
    let (changed_kernel, _) = input("ld.global.u64 %r0, [4112];", 1, false);
    assert_eq!(
        status_for_certificate(&changed_kernel, &launch, &certificate),
        Status::Inconclusive
    );
    assert!(!strict_memory::verify_certificate(
        &changed_kernel,
        &launch,
        &certificate
    ));
    let (unsupported, _) = input("add.u64 %r0, 1, 2;", 1, false);
    assert!(!strict_memory::verify_certificate(
        &unsupported,
        &launch,
        &certificate
    ));
    assert!(!strict_memory::verify_certificate(&kernel, &launch, &[]));
}
