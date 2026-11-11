/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Signed zeros of the MMA models against GB300 (probes/mma_zero): every modeled config of
//! `probes/mma_zero/expected/configs.json` (tcgen05 kind::f16 bf16 M128 / M64, kind::f16 f16,
//! kind::tf32 and kind::f8f6f4 e4m3 / e2m1 with C disabled or C = +-0, mma.sync m16n8k16 bf16 and m16n8k8 tf32 with C from
//! registers or RZ) through both front-ends, bit for bit. The cases: one
//! product that underflows f32 (the reported witness among them), products that are all +-0, exact
//! cancellation, a cancellation residue below 2^-149, subnormal-range results, an underflowing
//! product next to a nonzero accumulator. GB300 gives +0 for every zero result.

use std::path::PathBuf;
use std::sync::Arc;

use ptxv_case::Case;
use ptxv_sim::build_launch;
use ptxv_sim::machine::Machine;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// The case for one config (as probes/mma_zero/run.py writes it for `ptxv probe`).
fn config_case(dir: &std::path::Path, c: &serde_json::Value) -> Case {
    let probe = root().join("probes/mma_zero");
    let exp = probe.join("expected");
    let name = c["name"].as_str().unwrap();
    let entry = c["entry"].as_str().unwrap();
    let mut t = format!(
        "name = \"mma_zero_{name}\"\ndescription = \"probes/mma_zero {name}\"\nptx = \"{}\"\n\
         entry = \"{entry}\"\ngrid = [{}, 1, 1]\nblock = [{}, 1, 1]\noutputs = [\"D\"]\n",
        probe.join("kernel.ptx").display(),
        c["grid"],
        c["block"]
    );
    for b in ["A", "B", "C"] {
        let f = exp.join(format!("{name}.{b}.bin"));
        let n = std::fs::metadata(&f).unwrap().len();
        t += &format!(
            "[[buffers]]\nname = \"{b}\"\ndtype = \"u8\"\nshape = [{n}]\ninit = \"file\"\nfile = \"{}\"\n",
            f.display()
        );
    }
    t += &format!(
        "[[buffers]]\nname = \"D\"\ndtype = \"u32\"\nshape = [{}]\ninit = \"nan\"\n",
        c["d_bytes"].as_u64().unwrap() / 4
    );
    for b in ["A", "B", "C", "D"] {
        t += &format!("[[params]]\nkind = \"ptr\"\nbuffer = \"{b}\"\n");
    }
    if entry.starts_with("tc") {
        for k in ["idesc", "en"] {
            t += &format!("[[params]]\nkind = \"u32\"\nvalue = {}\n", c[k]);
        }
    }
    t += "[[variants]]\nid = \"good\"\nexpect = [\"clean\"]\n";
    let d = dir.join(name);
    std::fs::create_dir_all(&d).unwrap();
    std::fs::write(d.join("case.toml"), t).unwrap();
    Case::load(&d).unwrap()
}

#[test]
fn mma_zero_both_fronts_match_gb300() {
    let probe = root().join("probes/mma_zero");
    let exp = probe.join("expected");
    let dir = std::env::temp_dir().join(format!("ptxv-test-mma_zero-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let ptx = std::fs::read_to_string(probe.join("kernel.ptx")).unwrap();
    let cubin = dir.join("k.cubin");
    ptxv_sass::ptxas(&probe.join("kernel.ptx"), &cubin, "sm_103a").unwrap();
    let cfgs: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(exp.join("configs.json")).unwrap()).unwrap();
    let mut checked = 0;
    let mut witness = 0;
    for c in cfgs.iter().filter(|c| c["modeled"].as_bool().unwrap()) {
        let name = c["name"].as_str().unwrap();
        let case = config_case(&dir, c);
        let hw = std::fs::read(exp.join(format!("{name}.D.bin"))).unwrap();
        let w = |x: &[u8], i: usize| u32::from_le_bytes(x[4 * i..4 * i + 4].try_into().unwrap());
        // The probe's premise: every zero result on GB300 is +0 (f16 D: the low half).
        for cs in c["cases"].as_array().unwrap() {
            let u = w(&hw, cs["word"].as_u64().unwrap() as usize);
            let (u, neg0) = if name.contains("f16d") {
                (u & 0xffff, 0x8000)
            } else {
                (u, 0x8000_0000)
            };
            assert_ne!(u, neg0, "{name} {cs}");
            if cs["name"].as_str().unwrap().starts_with("witness/") {
                assert_eq!(u, 0, "{name}: the reported witness is +0 on GB300");
                witness += 1;
            }
        }
        let fronts = vec![
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
        ];
        for k in fronts {
            let front = format!("{:?}", k.front);
            let launch = build_launch(&case, &k).unwrap();
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
    assert_eq!(checked, 36, "18 modeled configs, both fronts");
    assert!(witness >= 30);
    let _ = std::fs::remove_dir_all(&dir);
}
