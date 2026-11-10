/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! tcgen05.mma at M = 128 against GB300 (probes/tc_m128): `.cta_group::2` with M = 128 (PTX ISA
//! Layout B: each CTA's 64 D rows in lanes 0..63 for the first N/2 columns and lanes 64..127 for
//! the rest, N/2 columns wide; A rows from each CTA's own shared memory, B rows N/2 from each),
//! M = 256 as the control, and `.cta_group::1` M = 128 with N = 8, 24, 40, 136 (and M = 64,
//! N = 8). kind::f16 (bf16) and kind::f8f6f4 (e4m3). `mma_img`: MN-major operands without
//! swizzle (A, B, both; two LBO / SBO arrangements), M = 128 / 64, N = 8..32. Each config's GB300 output is the dump of
//! all 128 lanes x 256 columns of every CTA's TMEM after the MMA, sentinel-filled before it, so
//! the comparison covers where D lands, what stays untouched, and the arithmetic, bit for bit.

use std::path::PathBuf;
use std::sync::Arc;

use ptxv_case::Case;
use ptxv_sim::build_launch;
use ptxv_sim::machine::Machine;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The case for one config, as probes/tc_m128/run.py writes it for `ptxv probe`.
fn config_case(dir: &std::path::Path, c: &serde_json::Value) -> Case {
    let probe = root().join("probes/tc_m128");
    let exp = probe.join("expected");
    let name = c["name"].as_str().unwrap();
    let ctas = c["ctas"].as_u64().unwrap();
    let cluster = if ctas == 2 {
        "cluster = [2, 1, 1]\n"
    } else {
        ""
    };
    let mut t = format!(
        "name = \"tc_m128_{name}\"\ndescription = \"probes/tc_m128 {name}\"\nptx = \"{ptx}\"\n\
         entry = \"{entry}\"\ngrid = [{ctas}, 1, 1]\nblock = [128, 1, 1]\n{cluster}outputs = [\"D\"]\n\
         [[buffers]]\nname = \"A\"\ndtype = \"u16\"\nshape = [{ctas}, 128, 16]\ninit = \"file\"\nfile = \"{e}/{name}.A.bin\"\n\
         [[buffers]]\nname = \"B\"\ndtype = \"u16\"\nshape = [{ctas}, 256, 16]\ninit = \"file\"\nfile = \"{e}/{name}.B.bin\"\n\
         [[buffers]]\nname = \"D\"\ndtype = \"u32\"\nshape = [{ctas}, 128, 256]\ninit = \"zeros\"\n\
         [[params]]\nkind = \"ptr\"\nbuffer = \"A\"\n[[params]]\nkind = \"ptr\"\nbuffer = \"B\"\n\
         [[params]]\nkind = \"ptr\"\nbuffer = \"D\"\n[[params]]\nkind = \"u32\"\nvalue = {idesc}\n\
         [[variants]]\nid = \"good\"\nexpect = [\"clean\"]\n",
        ptx = probe.join("kernel.ptx").display(),
        entry = c["entry"].as_str().unwrap(),
        e = exp.display(),
        idesc = c["idesc"],
    );
    // mma_img: raw shared-memory images and the descriptors' flag bits as parameters.
    if c["entry"] == "mma_img" {
        t = t
            .replace("shape = [1, 128, 16]", "shape = [2048]")
            .replace("shape = [1, 256, 16]", "shape = [4096]");
        let extra: String = ["aflags", "bflags"]
            .iter()
            .map(|k| format!("[[params]]\nkind = \"u64\"\nvalue = {}\n", c[k]))
            .collect();
        t = t.replace("[[variants]]", &(extra + "[[variants]]"));
    }
    let d = dir.join(name);
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(d.join("case.toml"), t).unwrap();
    Case::load(&d).unwrap()
}

#[test]
fn tc_m128_layouts_both_fronts_match_gb300() {
    let probe = root().join("probes/tc_m128");
    let exp = probe.join("expected");
    let dir = std::env::temp_dir().join(format!("ptxv-test-tc_m128-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let ptx = std::fs::read_to_string(probe.join("kernel.ptx")).unwrap();
    let cubin = dir.join("k.cubin");
    ptxv_sass::ptxas(&probe.join("kernel.ptx"), &cubin, "sm_103a").unwrap();
    let cfgs: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(exp.join("configs.json")).unwrap()).unwrap();
    let mut checked = 0;
    for c in &cfgs {
        let name = c["name"].as_str().unwrap();
        let case = config_case(&dir, c);
        let hw = std::fs::read(exp.join(format!("{name}.D.bin"))).unwrap();
        let w = |x: &[u8], i: usize| u32::from_le_bytes(x[4 * i..4 * i + 4].try_into().unwrap());
        // The premise of the layout check: GB300 wrote exactly M x N / (CTAs) cells per CTA.
        let (m, n) = (c["m"].as_u64().unwrap(), c["n"].as_u64().unwrap());
        let written = (0..hw.len() / 4)
            .filter(|&i| {
                let (lane, col) = ((i / 256) % 128, i % 256);
                w(&hw, i) != 0xff80_0000 | (lane as u32) << 8 | col as u32
            })
            .count() as u64;
        assert_eq!(written, m * n, "{name}: cells GB300 wrote");
        for k in [
            ptxv_sass::load_cubin(&cubin, &case.m.entry).unwrap(),
            ptxv_ptx::parse_kernel(&ptx, Some(&case.m.entry)).unwrap(),
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
            let got = m.outputs();
            assert_eq!(got.len(), hw.len(), "{name} {front}");
            let bad: Vec<(usize, u32, u32)> = (0..hw.len() / 4)
                .filter(|&i| w(&got, i) != w(&hw, i))
                .map(|i| (i, w(&got, i), w(&hw, i)))
                .collect();
            assert!(
                bad.is_empty(),
                "{name} {front}: {} words differ (word, sim, GB300): {:x?}",
                bad.len(),
                &bad[..bad.len().min(8)]
            );
            checked += 1;
        }
    }
    assert_eq!(checked, 2 * cfgs.len());
    assert_eq!(cfgs.len(), 34);
    let _ = std::fs::remove_dir_all(&dir);
}
