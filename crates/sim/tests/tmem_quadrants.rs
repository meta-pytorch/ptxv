/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::Path;
use std::sync::Arc;

use ptxv_case::Case;
use ptxv_sim::build_launch;
use ptxv_sim::explore::run_once;
use ptxv_sim::machine::Machine;

#[test]
fn tmem_quadrants_follow_linear_thread_rank() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut case = Case::load(root.join("fixtures/e_tmem_quadrants")).unwrap();
    let ptx = case.variant_ptx("good").unwrap();
    let dir = std::env::temp_dir().join(format!("ptxv-tmem-quadrants-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let source = dir.join("kernel.ptx");
    let cubin = dir.join("kernel.cubin");
    std::fs::write(&source, &ptx).unwrap();
    ptxv_sass::ptxas(&source, &cubin, &case.m.arch).unwrap();
    for k in [
        ptxv_ptx::parse_kernel_for(
            &ptx,
            Some(&case.m.entry),
            case.m
                .ptxas_toolkit
                .as_deref()
                .and_then(ptxv_ptx::toolkit_version),
        )
        .unwrap(),
        ptxv_sass::load_cubin(&cubin, &case.m.entry).unwrap(),
    ] {
        for block in [[128, 1, 1], [32, 4, 1], [32, 1, 4], [16, 2, 4]] {
            case.m.block = block;
            let launch = build_launch(&case, &k).unwrap();
            let (m, findings, done) = run_once(Machine::new(Arc::new(k.clone()), Arc::new(launch)));
            assert!(
                done && findings.is_empty(),
                "{:?} {block:?}: {findings:?}",
                k.front
            );
            assert_eq!(m.outputs(), 1.0f32.to_le_bytes(), "{:?} {block:?}", k.front);
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}
