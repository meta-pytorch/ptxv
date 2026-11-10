/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Multicast tile::gather4 TMA loads with and without `.cta_group::2` against GB300
//! (probes/tma_g4_mcast; SASS UTMALDG.2D.GATHER4.MULTICAST[.2CTA]): 56 configs of ctaMask,
//! mbarrier rank and destination rank, cluster of 4, each rank expecting exactly the bytes
//! S-TMA-2CTA-MCAST delivers to it. GB300's output (the gathered rows in every masked rank,
//! every mbarrier completed) is replayed bit for bit on both front-ends.

use std::path::PathBuf;
use std::sync::Arc;

use ptxv_case::Case;
use ptxv_sim::build_launch;
use ptxv_sim::machine::Machine;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn config_case(dir: &std::path::Path, c: &serde_json::Value) -> Case {
    let probe = root().join("probes/tma_g4_mcast");
    let name = c["name"].as_str().unwrap();
    let mut t = format!(
        "name = \"tma_g4_mcast_{name}\"\ndescription = \"probes/tma_g4_mcast {name}\"\n\
         ptx = \"{}\"\nentry = \"tma_g4_mcast\"\ngrid = [4, 1, 1]\nblock = [64, 1, 1]\n\
         cluster = [4, 1, 1]\noutputs = [\"out\"]\n\
         [[buffers]]\nname = \"src\"\ndtype = \"u32\"\nshape = [32, 16]\ninit = \"iota\"\n\
         [[buffers]]\nname = \"out\"\ndtype = \"u32\"\nshape = [4, 129]\ninit = \"zeros\"\n\
         [[params]]\nkind = \"tensormap\"\nbuffer = \"src\"\nbox = [16, 1]\n\
         [[params]]\nkind = \"ptr\"\nbuffer = \"out\"\n",
        probe.join("kernel.ptx").display()
    );
    for k in ["mask", "mbar_rank", "dst_rank", "exp", "pair"] {
        t += &format!("[[params]]\nkind = \"u32\"\nvalue = {}\n", c[k]);
    }
    t += "[[variants]]\nid = \"good\"\nexpect = [\"clean\"]\n";
    let d = dir.join(name);
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(d.join("case.toml"), t).unwrap();
    Case::load(&d).unwrap()
}

#[test]
fn tma_g4_mcast_both_fronts_match_gb300() {
    let probe = root().join("probes/tma_g4_mcast");
    let exp = probe.join("expected");
    let dir = std::env::temp_dir().join(format!("ptxv-test-tmag4-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let ptx = std::fs::read_to_string(probe.join("kernel.ptx")).unwrap();
    let cubin = dir.join("k.cubin");
    ptxv_sass::ptxas(&probe.join("kernel.ptx"), &cubin, "sm_103a").unwrap();
    let cfgs: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(exp.join("configs.json")).unwrap()).unwrap();
    let mut checked = 0;
    for c in &cfgs {
        let name = c["name"].as_str().unwrap();
        let hw = std::fs::read(exp.join(format!("{name}.bin"))).unwrap();
        let case = config_case(&dir, c);
        for k in [
            ptxv_sass::load_cubin(&cubin, &case.m.entry).unwrap(),
            ptxv_ptx::parse_kernel_for(
                &ptx,
                Some(&case.m.entry),
                case.m
                    .ptxas_toolkit
                    .as_deref()
                    .and_then(ptxv_ptx::toolkit_version),
            )
            .unwrap(),
        ] {
            let front = format!("{:?}", k.front);
            let mut launch = build_launch(&case, &k).unwrap();
            // S-MBAR-SCOPE: on the PTX front the peer CTA's completion (commit / TMA
            // complete-tx) waited at .cta scope is a contract finding (`ptxv check` reruns with
            // it at the mbarrier's CTA); the output is checked under that rerun's model.
            launch.async_mb_dest = k.front == ptxv_ir::Front::Ptx;
            let (m, findings, _) =
                ptxv_sim::explore::run_once(Machine::new(Arc::new(k), Arc::new(launch)));
            assert!(findings.is_empty(), "{name} {front}: {findings:?}");
            assert!(
                m.outputs() == hw,
                "{name} {front}: output differs from GB300"
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 2 * 56);
    let _ = std::fs::remove_dir_all(&dir);
}
