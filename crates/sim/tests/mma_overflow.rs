/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Replay the GB300 overflow/cancellation capture through the complete execution models.

use std::path::PathBuf;
use std::sync::Arc;

use ptxv_case::Case;
use ptxv_sim::build_launch;
use ptxv_sim::machine::Machine;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn mma_overflow_matches_gb300() {
    let probe = root().join("probes/mma_overflow");
    let exp = probe.join("expected");
    let ptx_path = root().join("probes/mma_zero/kernel.ptx");
    let ptx = std::fs::read_to_string(&ptx_path).unwrap();
    let cfgs: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(exp.join("configs.json")).unwrap()).unwrap();
    let dir = std::env::temp_dir().join(format!("ptxv-test-mma-overflow-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut reports = Vec::new();
    let mut mismatches = 0;
    for c in cfgs {
        let name = c["name"].as_str().unwrap();
        let entry = c["entry"].as_str().unwrap();
        let mut t = format!(
            "name = \"{name}\"\ndescription = \"GB300 MMA overflow capture\"\nptx = \"{}\"\nentry = \"{entry}\"\n\
             grid = [{}, 1, 1]\nblock = [{}, 1, 1]\noutputs = [\"D\"]\n",
            ptx_path.display(),
            c["grid"],
            c["block"]
        );
        for b in ["A", "B", "C"] {
            let f = exp.join(format!("{name}.{b}.bin"));
            let n = std::fs::metadata(&f).unwrap().len();
            t += &format!(
                "[[buffers]]\nname = \"{b}\"\ndtype = \"u8\"\nshape = [{n}]\n\
                 init = \"file\"\nfile = \"{}\"\n",
                f.display()
            );
        }
        let hw = std::fs::read(exp.join(format!("{name}.D.bin"))).unwrap();
        t += &format!(
            "[[buffers]]\nname = \"D\"\ndtype = \"u32\"\nshape = [{}]\ninit = \"nan\"\n",
            hw.len() / 4
        );
        for b in ["A", "B", "C", "D"] {
            t += &format!("[[params]]\nkind = \"ptr\"\nbuffer = \"{b}\"\n");
        }
        if entry.starts_with("tc") {
            for val in [
                c["idesc"].as_u64().unwrap(),
                u64::from(c["enabled"].as_bool().unwrap()),
            ] {
                t += &format!("[[params]]\nkind = \"u32\"\nvalue = {val}\n");
            }
        }
        t += "[[variants]]\nid = \"good\"\nexpect = [\"clean\"]\n";
        let d = dir.join(name);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("case.toml"), t).unwrap();
        let case = Case::load(&d).unwrap();
        let mut fronts = vec![ptxv_sass::load_cubin(&exp.join("kernel.cubin"), entry).unwrap()];
        if entry.starts_with("tc") {
            fronts.push(ptxv_ptx::parse_kernel(&ptx, Some(entry)).unwrap());
        }
        for k in fronts {
            let front = format!("{:?}", k.front);
            let launch = build_launch(&case, &k).unwrap();
            let (m, findings, _) =
                ptxv_sim::explore::run_once(Machine::new(Arc::new(k), Arc::new(launch)));
            assert!(findings.is_empty(), "{name} {front}: {findings:?}");
            let got = m.outputs();
            assert_eq!(got.len(), hw.len(), "{name} {front}");
            let cells = hw.len() / 4 / c["grid"].as_u64().unwrap() as usize;
            let w =
                |x: &[u8], i: usize| u32::from_le_bytes(x[4 * i..4 * i + 4].try_into().unwrap());
            let bad: Vec<_> = (0..hw.len() / 4)
                .filter(|&i| w(&got, i) != w(&hw, i))
                .collect();
            let mut cases = Vec::new();
            for (i, cs) in c["cases"].as_array().unwrap().iter().enumerate() {
                cases.push(serde_json::json!({
                    "name": cs["name"], "sim": format!("{:08x}", w(&got, i*cells)),
                    "gb300": format!("{:08x}", w(&hw, i*cells)),
                    "mismatches": bad.iter().filter(|&&j| j/cells == i).count()
                }));
            }
            mismatches += bad.len();
            reports.push(serde_json::json!({"config": name, "front": front,
                "words": hw.len()/4, "mismatches": bad.len(), "cases": cases}));
        }
    }
    println!(
        "MMA_OVERFLOW_REPLAY {}",
        serde_json::to_string(&reports).unwrap()
    );
    assert_eq!(reports.len(), 8);
    assert_eq!(mismatches, 0, "Full simulator outputs differ from GB300");
    std::fs::remove_dir_all(dir).unwrap();
}
