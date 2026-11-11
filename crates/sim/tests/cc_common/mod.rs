// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Shared helpers of the concolic tests (symb.rs, concolic.rs).

#![allow(dead_code)]

use std::path::PathBuf;
use std::sync::Arc;

use ptxv_case::Case;
use ptxv_sim::build_launch;
use ptxv_sim::machine::Machine;

pub fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

pub fn scratch_dir(name: &str, variant: &str) -> PathBuf {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "ptxv-cc-{}-{n}-{name}-{variant}",
        std::process::id()
    ))
}

pub fn kernel_of(case: &Case, variant: &str, front: &str) -> anyhow::Result<ptxv_ir::Kernel> {
    if let Some(c) = &case.m.cubin {
        let dir = scratch_dir(&case.m.name, variant);
        std::fs::create_dir_all(&dir)?;
        let p = dir.join("k.cubin");
        std::fs::copy(case.dir.join(c), &p)?;
        ptxv_sass::scoreboard::patch_file(&p, &case.m.entry, &case.variant(variant)?.patch)?;
        return ptxv_sass::load_cubin(&p, &case.m.entry);
    }
    let ptx = case.variant_ptx(variant)?;
    if !case.m.launches.is_empty() {
        anyhow::bail!("multi-kernel launch");
    }
    match front {
        "ptx" => ptxv_ptx::parse_kernel(&ptx, Some(&case.m.entry)),
        _ => {
            let dir = scratch_dir(&case.m.name, variant);
            std::fs::create_dir_all(&dir)?;
            let p = dir.join("k.ptx");
            std::fs::write(&p, &ptx)?;
            let c = dir.join("k.cubin");
            ptxv_sass::ptxas(&p, &c, &case.m.arch)?;
            ptxv_sass::scoreboard::patch_file(&c, &case.m.entry, &case.variant(variant)?.patch)?;
            ptxv_sass::load_cubin(&c, &case.m.entry)
        }
    }
}

pub fn machine(dir: &std::path::Path, variant: &str, front: &str) -> anyhow::Result<Machine> {
    let case = Case::load(dir)?;
    let case = case.for_variant(variant)?;
    let k = kernel_of(&case, variant, front)?;
    let launch = build_launch(&case, &k)?;
    Ok(Machine::new(Arc::new(k), Arc::new(launch)))
}

pub fn fixture(name: &str, variant: &str, front: &str) -> Machine {
    machine(&root().join("fixtures").join(name), variant, front).unwrap()
}
