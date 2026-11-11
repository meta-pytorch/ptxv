/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Rung 3b, the whole MoE (docs/rung3b.md): flashinfer 0.5.3's `trtllm_fp8_block_scale_moe` as
//! one 5-launch case (routing, gemm1, activation, gemm2, finalize), fixtures/r3b_trtllm_moe.
//! The good binary races in gemm2 (T3b-2); `fixed` closes it and is certified for all five
//! launches. The numerics claims are checked stage by stage against exact references built
//! from the dequantized fp8 data (f64), and end to end.

use std::path::PathBuf;
use std::sync::Arc;

use ptxv_case::Case;
use ptxv_sim::build_launch;
use ptxv_sim::machine::Finding;
use ptxv_sim::machine::Machine;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

/// Third-party kernel binaries are not distributed (fixtures/VENDOR_FIXTURES.toml): true when
/// one these fixtures need is missing (the test then skips).
fn vendor_missing(needs: &[&str]) -> bool {
    ptxv_case::skip_missing_vendor_files(&root().join("fixtures"), needs)
}

fn moe() -> Case {
    Case::load(root().join("fixtures/r3b_trtllm_moe")).unwrap()
}

/// Every launch's kernel from the variant's patched cubins, combined.
fn kernel(case: &Case, variant: &str) -> ptxv_ir::Kernel {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir =
        std::env::temp_dir().join(format!("ptxv-r3b-moe-{variant}-{}-{n}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let c = dir.join("k.cubin");
    std::fs::copy(case.dir.join(case.m.cubin.as_ref().unwrap()), &c).unwrap();
    let patch = &case.variant(variant).unwrap().patch;
    let (own, main): (Vec<_>, Vec<_>) = patch.iter().cloned().partition(|p| p.cubin.is_some());
    ptxv_sass::scoreboard::patch_file(&c, &case.m.entry, &main).unwrap();
    let mut patched = std::collections::BTreeMap::new();
    for p in &own {
        let name = p.cubin.clone().unwrap();
        let dst = patched.entry(name.clone()).or_insert_with(|| {
            let d = dir.join(&name);
            std::fs::copy(case.dir.join(&name), &d).unwrap();
            d
        });
        let entry = p.kernel.clone().unwrap_or_default();
        ptxv_sass::scoreboard::patch_file(dst, &entry, std::slice::from_ref(p)).unwrap();
    }
    let ks: Vec<_> = case
        .m
        .all_launches()
        .iter()
        .map(|l| match l.cubin {
            Some(own) if patched.contains_key(own) => {
                ptxv_sass::load_cubin(&patched[own], l.entry).unwrap()
            }
            Some(own) => ptxv_sass::load_cubin(&case.dir.join(own), l.entry).unwrap(),
            None => ptxv_sass::load_cubin(&c, l.entry).unwrap(),
        })
        .collect();
    ptxv_ir::Kernel::concat(ks)
}

fn machine(case: &Case, variant: &str) -> Machine {
    let k = kernel(case, variant);
    let launch = build_launch(case, &k).unwrap();
    Machine::new(Arc::new(k), Arc::new(launch))
}

fn run(case: &Case, variant: &str) -> (Machine, Vec<Finding>, bool) {
    ptxv_sim::explore::run_once(machine(case, variant))
}

fn kinds(fs: &[Finding], ok: bool) -> std::collections::BTreeSet<String> {
    let mut k: std::collections::BTreeSet<String> = fs.iter().map(|f| f.kind.clone()).collect();
    if !ok {
        k.insert("deadlock".into());
    }
    k
}

// ---------------------------------------------------------------- data

/// T = 16 tokens, H = 256, I = 128, E = 4 experts, top-2, 8-token tiles, 128-blocks.
const T: usize = 16;
const H: usize = 256;
const I: usize = 128;
const E: usize = 4;
const K: usize = 2;
const BLK: usize = 128;

fn f32s(b: &[u8]) -> Vec<f64> {
    b.chunks(4)
        .map(|c| f32::from_le_bytes(c.try_into().unwrap()) as f64)
        .collect()
}
fn i32s(b: &[u8]) -> Vec<i64> {
    b.chunks(4)
        .map(|c| i32::from_le_bytes(c.try_into().unwrap()) as i64)
        .collect()
}
fn bf16s(b: &[u8]) -> Vec<f64> {
    b.chunks(2)
        .map(|c| f32::from_bits((u16::from_le_bytes(c.try_into().unwrap()) as u32) << 16) as f64)
        .collect()
}
fn e4m3(c: u8) -> f64 {
    ptxv_sim::lowp::value(c as u32, ptxv_ir::Lp::E4m3)
}

/// Every buffer's final bytes, by name (machine allocation i is case buffer i).
fn buffers(case: &Case, m: &Machine) -> std::collections::HashMap<String, Vec<u8>> {
    case.m
        .buffers
        .iter()
        .enumerate()
        .map(|(i, b)| {
            let g = &m.gmem[i];
            (
                b.name.clone(),
                (0..g.bytes.len()).map(|j| *g.bytes.get(j)).collect(),
            )
        })
        .collect()
}

/// The inputs, dequantized: hs [T][H], hss [H/128][T], w1 [E][2I][H] (+ scales [E][2I/128][H/128]),
/// w2 [E][H][I] (+ scales [E][H/128][I/128]), logits [T][E].
struct Inputs {
    hs: Vec<f64>,
    hss: Vec<f64>,
    w1: Vec<f64>,
    w1s: Vec<f64>,
    w2: Vec<f64>,
    w2s: Vec<f64>,
    logits: Vec<f64>,
}

fn inputs(case: &Case) -> Inputs {
    let b = |n: &str| case.buffer(n).unwrap().init_bytes();
    Inputs {
        hs: b("hs").iter().map(|&c| e4m3(c)).collect(),
        hss: f32s(&b("hss")),
        w1: b("w1").iter().map(|&c| e4m3(c)).collect(),
        w1s: f32s(&b("w1s")),
        w2: b("w2").iter().map(|&c| e4m3(c)).collect(),
        w2s: f32s(&b("w2s")),
        logits: bf16s(&b("logits")),
    }
}

/// Renormalize routing, exact: the top-2 logits (ties: lower expert) and their softmax.
fn route(x: &Inputs, t: usize) -> [(usize, f64); K] {
    let l = &x.logits[t * E..(t + 1) * E];
    let mut ix: Vec<usize> = (0..E).collect();
    ix.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap().then(a.cmp(&b)));
    let z: f64 = ix[..K].iter().map(|&e| l[e].exp()).sum();
    [(ix[0], l[ix[0]].exp() / z), (ix[1], l[ix[1]].exp() / z)]
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// Exact gemm1 output channel ch of token t for expert e: (value, sum of |terms|).
fn gemm1_exact(x: &Inputs, e: usize, t: usize, ch: usize) -> (f64, f64) {
    let (mut v, mut mag) = (0.0, 0.0);
    for kb in 0..H / BLK {
        let s = x.w1s[(e * (2 * I / BLK) + ch / BLK) * (H / BLK) + kb] * x.hss[kb * T + t];
        let (mut d, mut a) = (0.0, 0.0);
        for k in kb * BLK..(kb + 1) * BLK {
            let p = x.w1[(e * 2 * I + ch) * H + k] * x.hs[t * H + k];
            d += p;
            a += p.abs();
        }
        v += s * d;
        mag += s.abs() * a;
    }
    (v, mag)
}

/// The claims, stage by stage, each against the exact function (f64) of the dequantized values
/// the stage read, and end to end against the exact MoE of the inputs. Returns the violations
/// per stage: (routing, gemm1, activation, gemm2, finalize, end to end).
fn claims(case: &Case, m: &Machine) -> [usize; 6] {
    let x = inputs(case);
    let b = buffers(case, m);
    let rows = i32s(&b["W0"])[0] as usize;
    let exp_to_perm = i32s(&b["W1"]);
    let perm_to_tok = i32s(&b["W2"]);
    let tile_expert = i32s(&b["W5"]);
    let wts = bf16s(&b["W8"]);
    let (g1, s1) = (&b["W9"], f32s(&b["W10"]));
    let (act, s2) = (&b["W12"], f32s(&b["W11"]));
    let y = bf16s(&b["W13"]);
    let out = bf16s(&b["out"]);
    let mut bad = [0usize; 6];
    let in_rows = |p: i64| p >= 0 && (p as usize) < rows;
    // Routing: each (t, k) has a permuted row of its expert's tile holding t; the weights are
    // the renormalized softmax in bf16 (half an ulp, 2^-8 relative, plus 2^-18 for the f32
    // softmax).
    for t in 0..T {
        for (k, &(e, w)) in route(&x, t).iter().enumerate() {
            let p = exp_to_perm[t * K + k];
            let ok = in_rows(p)
                && perm_to_tok[p as usize] == t as i64
                && tile_expert[p as usize / 8] == e as i64
                && (wts[t * K + k] - w).abs() <= w * ((-8f64).exp2() + (-18f64).exp2());
            bad[0] += !ok as usize;
        }
    }
    if bad[0] > 0 {
        bad[5] = usize::MAX;
        return bad; // the later stages' references need the routing
    }
    let row_of = |p: usize| (perm_to_tok[p] as usize, tile_expert[p / 8] as usize);
    // gemm1 (r3b_bmm_good's claim): e4m3 half ulp of the block scale's domain plus 2^-18 of
    // the terms.
    let g1v = |p: usize, ch: usize| e4m3(g1[p * 2 * I + ch]) * s1[(ch / BLK) * rows + p];
    for p in 0..rows {
        let (t, e) = row_of(p);
        for ch in 0..2 * I {
            let (v, mag) = gemm1_exact(&x, e, t, ch);
            let acc = mag * (-18f64).exp2();
            let s = s1[(ch / BLK) * rows + p];
            bad[1] += ((g1v(p, ch) - v).abs()
                > (v.abs() + acc) / 16.0 + s * (-10f64).exp2() * 1.001 + acc)
                as usize;
        }
    }
    // Activation: silu(gate) * up of gemm1's dequantized outputs (gate = channels I..2I),
    // e4m3 with one f32 scale per row (amax / 448): half an ulp, plus 2^-17 for the f32 math.
    let actv = |p: usize, i: usize| e4m3(act[p * I + i]) * s2[p];
    for (p, &s2p) in s2[..rows].iter().enumerate() {
        for i in 0..I {
            let v = silu(g1v(p, I + i)) * g1v(p, i);
            let tol = v.abs() * ((-4f64).exp2() + (-17f64).exp2()) + s2p * (-10f64).exp2() * 1.001;
            bad[2] += ((actv(p, i) - v).abs() > tol) as usize;
        }
    }
    // gemm2: bf16 of the exact product with the activation (half an ulp: 2^-8 relative) plus
    // 2^-18 of the terms.
    for p in 0..rows {
        let (_, e) = row_of(p);
        for h in 0..H {
            let s = x.w2s[e * (H / BLK) + h / BLK];
            let (mut v, mut mag) = (0.0, 0.0);
            for i in 0..I {
                let q = x.w2[(e * H + h) * I + i] * actv(p, i) * s;
                v += q;
                mag += q.abs();
            }
            let acc = mag * (-18f64).exp2();
            bad[3] += ((y[p * H + h] - v).abs() > (v.abs() + acc) * (-8f64).exp2() + acc) as usize;
        }
    }
    // Finalize: bf16 (2^-8 relative) of sum_k w_k y[row(t, k)] with the exact weights; the bf16
    // weights and the f32 accumulation add 2^-8 of the terms.
    for t in 0..T {
        let r = route(&x, t);
        for h in 0..H {
            let terms: Vec<f64> = (0..K)
                .map(|k| r[k].1 * y[exp_to_perm[t * K + k] as usize * H + h])
                .collect();
            let v: f64 = terms.iter().sum();
            let mag: f64 = terms.iter().map(|q| q.abs()).sum();
            let tol = v.abs() * (-8f64).exp2() + mag * ((-8f64).exp2() + (-18f64).exp2());
            bad[4] += ((out[t * H + h] - v).abs() > tol) as usize;
        }
    }
    // End to end, the exact MoE of the inputs (no intermediate rounding): every stage's
    // rounding propagated in worst case (|silu'| <= 1.1). Loose (cancellation in the sums),
    // but a bound for every output.
    for t in 0..T {
        let r = route(&x, t);
        let (mut ys, mut dys) = (vec![[0.0f64; K]; H], vec![[0.0f64; K]; H]);
        for (k, &(e, _)) in r.iter().enumerate() {
            let p = exp_to_perm[t * K + k] as usize;
            let (mut a, mut da) = (vec![0.0; I], vec![0.0; I]);
            for i in 0..I {
                let ((x1, m1), (g, mg)) = (gemm1_exact(&x, e, t, i), gemm1_exact(&x, e, t, I + i));
                let (a1, ag) = (m1 * (-18f64).exp2(), mg * (-18f64).exp2());
                let dx1 = (x1.abs() + a1) / 16.0 + s1[p] * (-10f64).exp2() + a1;
                let dg = (g.abs() + ag) / 16.0 + s1[rows + p] * (-10f64).exp2() + ag;
                a[i] = silu(g) * x1;
                let pre = silu(g).abs() * dx1 + 1.1 * dg * (x1.abs() + dx1);
                let pre = pre + (a[i].abs() + pre) * (-17f64).exp2();
                da[i] = pre * (1.0 + 1.0 / 16.0) + a[i].abs() / 16.0 + s2[p] * (-10f64).exp2();
            }
            for h in 0..H {
                let s = x.w2s[e * (H / BLK) + h / BLK];
                let (mut v, mut d, mut mag) = (0.0, 0.0, 0.0);
                for i in 0..I {
                    let c = x.w2[(e * H + h) * I + i] * s;
                    v += c * a[i];
                    d += c.abs() * da[i];
                    mag += c.abs() * (a[i].abs() + da[i]);
                }
                d += mag * (-18f64).exp2();
                d += (v.abs() + d) * (-8f64).exp2();
                ys[h][k] = v;
                dys[h][k] = d;
            }
        }
        for h in 0..H {
            let v: f64 = (0..K).map(|k| r[k].1 * ys[h][k]).sum();
            let mut d: f64 = (0..K)
                .map(|k| {
                    r[k].1 * dys[h][k] + r[k].1 * (-8f64).exp2() * (ys[h][k].abs() + dys[h][k])
                })
                .sum();
            d += (v.abs() + d) * (-8f64).exp2();
            bad[5] += ((out[t * H + h] - v).abs() > d) as usize;
        }
    }
    bad
}

/// The good binary in one deterministic run: the only findings besides the E-BULK-EXIT /
/// E-TC-EXIT contracts are gemm2's TMEM races (T3b-2: warp 0 frees TMEM after a barrier of
/// epilogue warps 0-3 only), the output equals GB300's bitwise, and the routing and gemm1
/// intermediates equal the GPU's snapshots (fixtures/r3b_trtllm_bmm, the same inputs).
#[test]
#[ignore = "heavy: 32 s"]
fn r3b_moe_good_races_in_gemm2_and_matches_the_gpu() {
    if vendor_missing(&["r3b_trtllm_moe", "r3b_trtllm_bmm"]) {
        return;
    }
    let case = moe();
    let (m, fs, ok) = run(&case, "good");
    assert!(ok);
    assert_eq!(
        kinds(&fs, ok),
        ["contract", "race"].into_iter().map(String::from).collect()
    );
    for f in fs.iter().filter(|f| f.kind == "race") {
        assert!(
            f.msg.contains("bmm_Bfloat16") && f.msg.contains("UTCATOMSWS.AND"),
            "{}",
            f.msg
        );
    }
    let report = case.expect_report(&m.outputs()).unwrap();
    assert!(
        report.len() == 1 && report[0].contains("bitwise equal"),
        "{report:?}"
    );
    let b = buffers(&case, &m);
    let snap = Case::load(root().join("fixtures/r3b_trtllm_bmm")).unwrap();
    let gpu = |n: &str| snap.buffer(n).unwrap().init_bytes();
    let rows = i32s(&b["W0"])[0] as usize;
    assert_eq!(b["W0"], gpu("W0"));
    assert_eq!(
        b["W2"][..rows * 4],
        gpu("W1")[..rows * 4],
        "permuted row -> token"
    );
    assert_eq!(b["W5"][..rows / 2], gpu("W2")[..rows / 2], "tile -> expert");
    assert_eq!(
        b["W6"][..rows / 2],
        gpu("W3")[..rows / 2],
        "tile -> row limit"
    );
    assert_eq!(b["W8"], gpu("W5"), "routing weights");
    let g1 = std::fs::read(snap.dir.join("expected/W6.bin")).unwrap();
    let g1s = std::fs::read(snap.dir.join("expected/W7.bin")).unwrap();
    assert_eq!(b["W9"][..rows * 2 * I], g1[..rows * 2 * I], "gemm1 output");
    assert_eq!(
        b["W10"][..2 * rows * 4],
        g1s[..2 * rows * 4],
        "gemm1 scales"
    );
}

/// `fixed` (gemm2's epilogue halves meet before the TMEM free): one run certifies all five
/// launches (E-WELLSYNC), contracts only, output = GB300's bitwise, and every numerics claim
/// holds for all outputs.
#[test]
#[ignore = "heavy: 74 s"]
fn r3b_moe_fixed_certified_and_claims_hold() {
    if vendor_missing(&["r3b_trtllm_moe"]) {
        return;
    }
    let case = moe();
    let m0 = machine(&case, "fixed");
    let c = ptxv_sim::wellsync::certify(m0);
    let v = c
        .verdict
        .unwrap_or_else(|| panic!("not certified: {:?}", c.reasons));
    assert!(v.complete);
    assert_eq!(v.classes(), ["contract"], "{:#?}", v.findings);
    let report = case.expect_report(&v.outputs[0]).unwrap();
    assert!(report[0].contains("bitwise equal"), "{report:?}");
    // The claims read every buffer: rerun (deterministic) to get the machine.
    let (m, _, ok) = run(&case, "fixed");
    assert!(ok);
    assert_eq!(m.outputs(), v.outputs[0]);
    assert_eq!(claims(&case, &m), [0; 6]);
}

/// B70 (routing: the expert-count atomic adds 0, so an expert's tokens share slot 0) is a race
/// on the permutation the routing kernel writes, and the routing claim fails. B71 / B72
/// (finalize: the second expert gets the first one's weight / row) execute like `fixed`
/// (contracts only); the finalize claim fails for most outputs, every earlier stage holds.
#[test]
#[ignore = "heavy: 78 s"]
fn r3b_moe_mutants() {
    if vendor_missing(&["r3b_trtllm_moe"]) {
        return;
    }
    let case = moe();
    let (m, fs, ok) = run(&case, "B70");
    let got = kinds(&fs, ok);
    assert!(got.contains("race"), "B70: {got:?}");
    assert!(
        fs.iter()
            .any(|f| f.kind == "race" && f.msg.contains("routingIndicesClusterKernel")),
        "{fs:#?}"
    );
    if ok {
        assert!(claims(&case, &m)[0] > 0, "B70: routing claim holds");
    }
    for v in ["B71", "B72"] {
        let (m, fs, ok) = run(&case, v);
        assert_eq!(
            kinds(&fs, ok),
            ["contract".to_string()].into_iter().collect()
        );
        let bad = claims(&case, &m);
        assert_eq!(bad[..4], [0; 4], "{v}: {bad:?}");
        assert!(
            bad[4] > T * H / 2,
            "{v}: finalize claim fails for only {}",
            bad[4]
        );
    }
}

/// Run `fixed` with input buffer `name` replaced by `bytes`; the output.
fn run_with(case: &Case, name: &str, bytes: &[u8], tag: &str) -> Vec<u8> {
    let mut c = case.clone();
    let dir = std::env::temp_dir().join(format!("ptxv-r3b-moe-dep-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join(format!("{name}_{tag}.bin"));
    std::fs::write(&f, bytes).unwrap();
    let b = c.m.buffers.iter_mut().find(|b| b.name == name).unwrap();
    b.file = Some(f.to_str().unwrap().to_string());
    let (m, fs, ok) = run(&c, "fixed");
    assert!(
        ok && fs.iter().all(|f| f.kind == "contract"),
        "{tag}: {fs:?}"
    );
    m.outputs()
}

/// The routing dependency claim: token t's output row depends exactly on its top-k experts.
/// Perturbing expert e's weights (both GEMMs) changes exactly the rows of the tokens routed to
/// e; perturbing one token's activations changes exactly its row.
#[test]
#[ignore = "heavy: 97 s"]
fn r3b_moe_token_depends_exactly_on_its_experts() {
    if vendor_missing(&["r3b_trtllm_moe"]) {
        return;
    }
    let case = moe();
    let x = inputs(&case);
    let (base, _, ok) = run(&case, "fixed");
    assert!(ok);
    let base = base.outputs();
    let rows_changed = |o: &[u8]| -> Vec<usize> {
        (0..T)
            .filter(|&t| o[t * H * 2..(t + 1) * H * 2] != base[t * H * 2..(t + 1) * H * 2])
            .collect()
    };
    for e in [3usize] {
        let mut w1 = case.buffer("w1").unwrap().init_bytes();
        for (i, b) in w1[e * 2 * I * H..(e + 1) * 2 * I * H]
            .iter_mut()
            .enumerate()
        {
            *b = ((i * 53 + 7 + e) % 0x70) as u8 | if i % 2 == 0 { 0x80 } else { 0 };
        }
        let mut c = case.clone();
        let mut w2 = c.buffer("w2").unwrap().init_bytes();
        for (i, b) in w2[e * H * I..(e + 1) * H * I].iter_mut().enumerate() {
            *b = ((i * 29 + 3 + e) % 0x70) as u8 | if i % 3 == 0 { 0x80 } else { 0 };
        }
        let dir = std::env::temp_dir().join(format!("ptxv-r3b-moe-dep-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let f = dir.join(format!("w2_e{e}.bin"));
        std::fs::write(&f, &w2).unwrap();
        c.m.buffers
            .iter_mut()
            .find(|b| b.name == "w2")
            .unwrap()
            .file = Some(f.to_str().unwrap().to_string());
        let o = run_with(&c, "w1", &w1, &format!("e{e}"));
        let routed: Vec<usize> = (0..T)
            .filter(|&t| route(&x, t).iter().any(|&(g, _)| g == e))
            .collect();
        assert_eq!(routed.len(), 8);
        assert_eq!(rows_changed(&o), routed, "expert {e}");
    }
    let t = 6;
    let mut hs = case.buffer("hs").unwrap().init_bytes();
    for (i, b) in hs[t * H..(t + 1) * H].iter_mut().enumerate() {
        *b = ((i * 37 + 11) % 0x70) as u8 | if i % 3 == 0 { 0x80 } else { 0 };
    }
    assert_eq!(rows_changed(&run_with(&case, "hs", &hs, "hs")), [t]);
}

/// PTX leaves the order of a warp's same-address atomic lanes open (the model takes lane
/// order, E-LOCKSTEP). Routing assigns each (token, k) its slot within an expert with
/// `ATOMS.ADD`; in reverse lane order the permutation changes, the MoE output does not.
#[test]
#[ignore = "heavy: 64 s"]
fn r3b_moe_output_does_not_depend_on_atomic_lane_order() {
    if vendor_missing(&["r3b_trtllm_moe"]) {
        return;
    }
    let case = moe();
    let (m0, _, ok) = run(&case, "fixed");
    assert!(ok);
    let mut m = machine(&case, "fixed");
    m.atom_lanes_rev = true;
    let (m1, fs, ok) = ptxv_sim::explore::run_once(m);
    assert!(ok && fs.iter().all(|f| f.kind == "contract"), "{fs:?}");
    let (b0, b1) = (buffers(&case, &m0), buffers(&case, &m1));
    assert_ne!(b0["W1"], b1["W1"], "the permutation did not change");
    assert_eq!(m0.outputs(), m1.outputs());
    assert_eq!(claims(&case, &m1), [0; 6]);
}
