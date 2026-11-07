/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! N-MMA-F8: tcgen05.mma.kind::f8f6f4 (probes/mma_f8) through both front-ends against recorded
//! GB300 outputs, bit for bit: FP8 / FP6 / FP4 pairs, f32 and f16 D, wide first steps
//! (expected/configs.json, written from `ptxv calib-mma probes/mma_f8` runs).

use std::path::PathBuf;
use std::sync::Arc;

use ptxv_case::Case;
use ptxv_case::ParamSpec;
use ptxv_sim::build_launch;
use ptxv_sim::machine::Machine;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn run(case: &Case, k: ptxv_ir::Kernel) -> (Vec<u8>, Vec<String>) {
    let launch = build_launch(case, &k).unwrap();
    let m = Machine::new(Arc::new(k), Arc::new(launch));
    let (m, findings, _) = ptxv_sim::explore::run_once(m);
    (
        m.outputs(),
        findings.iter().map(|f| format!("{f:?}")).collect(),
    )
}

#[test]
fn mma_f8_both_fronts_match_gb300() {
    let base = Case::load(root().join("probes/mma_f8")).unwrap();
    let dir = std::env::temp_dir().join("ptxv-test-mma_f8");
    std::fs::create_dir_all(&dir).unwrap();
    let ptx = base.variant_ptx("good").unwrap();
    let (p, cubin) = (dir.join("k.ptx"), dir.join("k.cubin"));
    std::fs::write(&p, &ptx).unwrap();
    ptxv_sass::ptxas(&p, &cubin, &base.m.arch).unwrap();
    let exp = base.dir.join("expected");
    let cfgs: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(exp.join("configs.json")).unwrap()).unwrap();
    assert_eq!(cfgs.len(), 9);
    for c in &cfgs {
        let name = c["name"].as_str().unwrap();
        let mut case = base.clone();
        for b in &mut case.m.buffers {
            if b.name == "A" || b.name == "B" {
                b.file = Some(
                    exp.join(format!("{name}.{}.bin", b.name))
                        .to_string_lossy()
                        .into(),
                );
            }
        }
        let mut u = 0;
        for prm in &mut case.m.params {
            if let ParamSpec::U32 { value } = prm {
                *value = c[if u == 0 { "idesc" } else { "idesc0" }].as_u64().unwrap() as u32;
                u += 1;
            }
        }
        let hw = std::fs::read(exp.join(format!("{name}.D.bin"))).unwrap();
        for k in [
            ptxv_ptx::parse_kernel(&ptx, Some(&case.m.entry)).unwrap(),
            ptxv_sass::load_cubin(&cubin, &case.m.entry).unwrap(),
        ] {
            let front = format!("{:?}", k.front);
            let (got, findings) = run(&case, k);
            assert_eq!(got.len(), hw.len());
            let w =
                |x: &[u8], i: usize| u32::from_le_bytes(x[4 * i..4 * i + 4].try_into().unwrap());
            let bad: Vec<(usize, u32, u32)> = (0..hw.len() / 4)
                .filter(|&i| w(&got, i) != w(&hw, i))
                .map(|i| (i, w(&got, i), w(&hw, i)))
                .take(6)
                .collect();
            assert!(
                bad.is_empty(),
                "{name} {front}: {} (word, sim, GB300): {bad:x?}; findings {findings:?}",
                (0..hw.len() / 4)
                    .filter(|&i| w(&got, i) != w(&hw, i))
                    .count()
            );
        }
    }
}

/// N-MMA-BS: the block-scaled kinds (probes/mma_bs: mxf8f6f4, mxf4, mxf4nvf4 with ue8m0 and
/// ue4m3 scales in TMEM) through both front-ends against recorded GB300 outputs.
#[test]
fn mma_bs_both_fronts_match_gb300() {
    mma_bs_configs("", 10);
}

/// N-MMA-BS: block-scaled M = 128, N = 8 (FA4 decode's UTCQMMA): probes/mma_bs with the first
/// MMA at N = 64 and the three accumulating ones at N = 8 (expected/n8/configs.json). GB300:
/// columns 0..7 = the N = 64 result bit for bit, columns 8.. keep the first MMA's.
#[test]
fn mma_bs_n8_both_fronts_match_gb300() {
    mma_bs_configs("n8", 5);
}

/// probes/mma_bs's configurations in expected/<sub>/configs.json (operands from expected/,
/// GB300's D from expected/<sub>/), both front-ends.
fn mma_bs_configs(sub: &str, count: usize) {
    let base = Case::load(root().join("probes/mma_bs")).unwrap();
    let dir = std::env::temp_dir().join(format!("ptxv-test-mma_bs{sub}"));
    std::fs::create_dir_all(&dir).unwrap();
    let ptx = base.variant_ptx("good").unwrap();
    let (p, cubin) = (dir.join("k.ptx"), dir.join("k.cubin"));
    std::fs::write(&p, &ptx).unwrap();
    ptxv_sass::ptxas(&p, &cubin, &base.m.arch).unwrap();
    let exp = base.dir.join("expected");
    let cfgs: Vec<serde_json::Value> =
        serde_json::from_str(&std::fs::read_to_string(exp.join(sub).join("configs.json")).unwrap())
            .unwrap();
    assert_eq!(cfgs.len(), count);
    for c in &cfgs {
        let name = c["name"].as_str().unwrap();
        let mut case = base.clone();
        case.m.entry = c["entry"].as_str().unwrap().to_string();
        for b in &mut case.m.buffers {
            if ["A", "B", "SFA", "SFB"].contains(&b.name.as_str()) {
                b.file = Some(
                    exp.join(format!("{name}.{}.bin", b.name))
                        .to_string_lossy()
                        .into(),
                );
            }
        }
        let mut u = 0;
        for prm in &mut case.m.params {
            if let ParamSpec::U32 { value } = prm {
                *value = c[if u == 0 { "idesc" } else { "idesc0" }].as_u64().unwrap() as u32;
                u += 1;
            }
        }
        let hw = std::fs::read(exp.join(sub).join(format!("{name}.D.bin"))).unwrap();
        for k in [
            ptxv_ptx::parse_kernel(&ptx, Some(&case.m.entry)).unwrap(),
            ptxv_sass::load_cubin(&cubin, &case.m.entry).unwrap(),
        ] {
            let front = format!("{:?}", k.front);
            let (got, findings) = run(&case, k);
            assert_eq!(got.len(), hw.len());
            let w =
                |x: &[u8], i: usize| u32::from_le_bytes(x[4 * i..4 * i + 4].try_into().unwrap());
            let bad: Vec<(usize, u32, u32)> = (0..hw.len() / 4)
                .filter(|&i| w(&got, i) != w(&hw, i))
                .map(|i| (i, w(&got, i), w(&hw, i)))
                .take(6)
                .collect();
            assert!(
                bad.is_empty(),
                "{name} {front}: {} (word, sim, GB300): {bad:x?}; findings {findings:?}",
                (0..hw.len() / 4)
                    .filter(|&i| w(&got, i) != w(&hw, i))
                    .count()
            );
        }
    }
}

/// E-PAIR, S-TCCP, N-MMA-BS: the cta_group::2 forms (probes/tc_pair_bs): a block-scaled
/// mxf4nvf4 pair MMA (each CTA's D rows use the scale factors in its own TMEM) and a pair
/// tcgen05.cp (each CTA's smem to its own TMEM), both front-ends against GB300, bit for bit.
#[test]
fn pair_block_scaled_mma_and_cp_match_gb300() {
    for (dir, out) in [("probes/tc_pair_bs", "D"), ("probes/tc_pair_bs/cp", "out")] {
        let case = Case::load(root().join(dir)).unwrap();
        let tmp = std::env::temp_dir().join(format!("ptxv-test-{}", case.m.name));
        std::fs::create_dir_all(&tmp).unwrap();
        let ptx = case.variant_ptx("good").unwrap();
        let (p, cubin) = (tmp.join("k.ptx"), tmp.join("k.cubin"));
        std::fs::write(&p, &ptx).unwrap();
        ptxv_sass::ptxas(&p, &cubin, &case.m.arch).unwrap();
        let hw = std::fs::read(case.dir.join(format!("expected/{out}.bin"))).unwrap();
        for k in [
            ptxv_ptx::parse_kernel(&ptx, Some(&case.m.entry)).unwrap(),
            ptxv_sass::load_cubin(&cubin, &case.m.entry).unwrap(),
        ] {
            let front = format!("{:?}", k.front);
            let (got, findings) = run(&case, k);
            assert!(findings.is_empty(), "{dir} {front}: {findings:?}");
            assert!(
                got == hw,
                "{dir} {front}: simulator output differs from GB300"
            );
        }
    }
}
