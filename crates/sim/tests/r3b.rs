/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Rung 3b (docs/rung3b.md): flashinfer's trtllm-gen gemm1 batched GEMM (fp8 block scales,
//! TMA gather4 of routed tokens, per-128-K software promotion, e4m3 output with per-token
//! amax scales), fixtures/r3b_trtllm_bmm. One deterministic schedule per run (the launch is
//! certified well-synchronized, E-WELLSYNC); the numerics claim is checked against an exact
//! reference built from the case's own inputs and routing buffers.

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

/// The case's cubin with the variant's SASS patches.
fn kernel(case: &Case, variant: &str) -> ptxv_ir::Kernel {
    static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let dir = std::env::temp_dir().join(format!(
        "ptxv-r3b-{}-{variant}-{}-{n}",
        case.m.name,
        std::process::id()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    let c = dir.join("k.cubin");
    std::fs::copy(case.dir.join(case.m.cubin.as_ref().unwrap()), &c).unwrap();
    let patch = &case.variant(variant).unwrap().patch;
    ptxv_sass::scoreboard::patch_file(&c, &case.m.entry, patch).unwrap();
    ptxv_sass::load_cubin(&c, &case.m.entry).unwrap()
}

fn run(case: &Case, variant: &str) -> (Machine, Vec<Finding>, bool) {
    let k = kernel(case, variant);
    let launch = build_launch(case, &k).unwrap();
    ptxv_sim::explore::run_once(Machine::new(Arc::new(k), Arc::new(launch)))
}

fn kinds(fs: &[Finding], ok: bool) -> std::collections::BTreeSet<String> {
    let mut k: std::collections::BTreeSet<String> = fs.iter().map(|f| f.kind.clone()).collect();
    if !ok {
        k.insert("deadlock".into());
    }
    k
}

fn bmm() -> Case {
    Case::load(root().join("fixtures/r3b_trtllm_bmm")).unwrap()
}

/// Shape of the capture: T = 16 tokens, H = K = 256 (`Inputs::h`), 2I = M = 256, E = 4 experts,
/// 8-token tiles.
const H: usize = 256;
const M: usize = 256;
const BLK: usize = 128;
const TILE: usize = 8;

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
fn e4m3(c: u8) -> f64 {
    ptxv_sim::lowp::value(c as u32, ptxv_ir::Lp::E4m3)
}

/// The routed GEMM's inputs as the kernel sees them: permuted row p holds token perm[p] of
/// expert expert[p / 8]; `rows` = total padded rows (W0).
struct Inputs {
    hs: Vec<u8>,
    hss: Vec<f64>,
    w1: Vec<u8>,
    w1s: Vec<f64>,
    perm: Vec<i64>,
    expert: Vec<i64>,
    rows: usize,
    t: usize,
    /// K (the hidden size): 256, or 2048 in fixtures/r3b_trtllm_bmm_k2048.
    h: usize,
}

fn inputs(case: &Case) -> Inputs {
    let b = |n: &str| case.buffer(n).unwrap().init_bytes();
    let rows = i32s(&b("W0"))[0] as usize;
    Inputs {
        t: case.buffer("logits").unwrap().elems() / 4,
        h: case.buffer("hs").unwrap().elems() / (case.buffer("logits").unwrap().elems() / 4),
        hs: b("hs"),
        hss: f32s(&b("hss")),
        w1: b("w1"),
        w1s: f32s(&b("w1s")),
        perm: i32s(&b("W1"))[..rows].to_vec(),
        expert: i32s(&b("W2"))[..rows / TILE].to_vec(),
        rows,
    }
}

/// Exact pre-quantization value of output (p, ch) and the sum of the magnitudes of its terms:
/// sum over 128-K blocks kb of w1s[g][ch/128][kb] * hss[kb][tok] * sum_k w1[g][ch][k] hs[tok][k].
fn reference(x: &Inputs, p: usize, ch: usize) -> (f64, f64) {
    let (g, tok) = (x.expert[p / TILE] as usize, x.perm[p] as usize);
    let (mut v, mut mag) = (0.0, 0.0);
    let h = x.h;
    for kb in 0..h / BLK {
        let s = x.w1s[(g * (M / BLK) + ch / BLK) * (h / BLK) + kb] * x.hss[kb * x.t + tok];
        let (mut d, mut a) = (0.0, 0.0);
        for k in kb * BLK..(kb + 1) * BLK {
            let pr = e4m3(x.w1[(g * M + ch) * h + k]) * e4m3(x.hs[tok * h + k]);
            d += pr;
            a += pr.abs();
        }
        v += s * d;
        mag += s.abs() * a;
    }
    (v, mag)
}

/// The e4m3 output rows (M bytes each) and their f32 scales ([M/128][rows]) of a run's
/// outputs: W6 then W7 (fixtures/r3b_trtllm_bmm), or W6 = scales, W7 = rows (_k2048).
fn split_out<'a>(case: &Case, out: &'a [u8]) -> (&'a [u8], Vec<f64>) {
    let w = case.buffer(&case.m.outputs[0]).unwrap().elems();
    if w == 262_144 {
        (&out[..w], f32s(&out[w..]))
    } else {
        (&out[w..], f32s(&out[..w]))
    }
}

fn dequant(e4: &[u8], sc: &[f64], rows: usize, p: usize, ch: usize) -> (f64, f64) {
    let s = sc[(ch / BLK) * rows + p];
    (e4m3(e4[p * M + ch]) * s, s)
}

/// The numerics claim: every output, dequantized with its block scale, is within e4m3
/// rounding (half an ulp: 2^-4 relative, 2^-10 absolute in the scaled domain) of the exact
/// value, plus 2^-18 of its terms' magnitude for the f32 / tensor-core accumulation. Returns
/// the outputs that violate it.
fn bound_violations(case: &Case, out: &[u8]) -> Vec<(usize, usize, f64, f64)> {
    let x = inputs(case);
    let (e4, sc) = split_out(case, out);
    let mut bad = Vec::new();
    for p in 0..x.rows {
        for ch in 0..M {
            let (v, mag) = reference(&x, p, ch);
            let (d, s) = dequant(e4, &sc, x.rows, p, ch);
            let acc = mag * (-18f64).exp2();
            if (d - v).abs() > (v.abs() + acc) / 16.0 + s * (-10f64).exp2() * 1.001 + acc {
                bad.push((p, ch, d, v));
            }
        }
    }
    bad
}

/// The good kernel: one run is the whole verdict (certificate), only the E-BULK-EXIT and
/// E-TC-EXIT contracts, outputs bitwise equal to the GPU's (in the MoE call), and the
/// numerics claim holds for all 32 x 256 outputs.
#[test]
fn r3b_bmm_good() {
    if vendor_missing(&["r3b_trtllm_bmm"]) {
        return;
    }
    let case = bmm();
    let k = kernel(&case, "good");
    let launch = build_launch(&case, &k).unwrap();
    let c = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)));
    let v = c
        .verdict
        .unwrap_or_else(|| panic!("not certified: {:?}", c.reasons));
    assert!(v.complete);
    assert_eq!(v.classes(), ["contract"], "{:#?}", v.findings);
    let sigs: std::collections::BTreeSet<&str> = v
        .findings
        .iter()
        .map(|f| f.sig.split(':').next().unwrap())
        .collect();
    assert_eq!(
        sigs,
        ["bulk-store-at-exit", "tc-commit-at-exit"]
            .into_iter()
            .collect()
    );
    let report = case.expect_report(&v.outputs[0]).unwrap();
    assert!(
        report.len() == 2 && report.iter().all(|l| l.contains("bitwise equal")),
        "{report:?}"
    );
    let bad = bound_violations(&case, &v.outputs[0]);
    assert!(
        bad.is_empty(),
        "{} outputs off: {:?}",
        bad.len(),
        &bad[..bad.len().min(5)]
    );
}

/// B25 / B27 / B28: the expected classes in one run. B30 executes like the good kernel; the
/// numerics claim catches it.
#[test]
fn r3b_bmm_mutants() {
    if vendor_missing(&["r3b_trtllm_bmm"]) {
        return;
    }
    let case = bmm();
    for v in ["B25", "B27", "B28"] {
        let (_, fs, ok) = run(&case, v);
        let got = kinds(&fs, ok);
        for e in &case.variant(v).unwrap().expect {
            assert!(got.contains(e), "{v}: want {e}, got {got:?}");
        }
    }
    let (m, fs, ok) = run(&case, "B30");
    assert!(ok);
    assert_eq!(
        kinds(&fs, ok),
        ["contract".to_string()].into_iter().collect()
    );
    let bad = bound_violations(&case, &m.outputs());
    assert!(
        bad.len() > 1000,
        "B30: only {} outputs off the claim",
        bad.len()
    );
}

/// Run the good kernel with buffer `name` replaced by `bytes`; the outputs.
fn run_with(case: &Case, name: &str, bytes: &[u8], tag: &str) -> Vec<u8> {
    let mut c = case.clone();
    let dir = std::env::temp_dir().join(format!("ptxv-r3b-dep-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let f = dir.join(format!("{name}_{tag}.bin"));
    std::fs::write(&f, bytes).unwrap();
    let b = c.m.buffers.iter_mut().find(|b| b.name == name).unwrap();
    b.file = Some(f.to_str().unwrap().to_string());
    let (m, fs, ok) = run(&c, "good");
    assert!(
        ok && fs.iter().all(|f| f.kind == "contract"),
        "{tag}: {fs:?}"
    );
    m.outputs()
}

/// (row, channel) outputs and (block, row) scales.
type Changed = (Vec<(usize, usize)>, Vec<(usize, usize)>);

/// Which (row, channel) outputs and (block, row) scales differ.
fn changed(a: &[u8], b: &[u8], rows: usize) -> Changed {
    let w6 = 262_144;
    assert_eq!(
        a[rows * M..w6],
        b[rows * M..w6],
        "bytes past the routed rows changed"
    );
    let mut o = Vec::new();
    for p in 0..rows {
        for ch in 0..M {
            if a[p * M + ch] != b[p * M + ch] {
                o.push((p, ch));
            }
        }
    }
    let (sa, sb) = (&a[w6..], &b[w6..]);
    let mut s = Vec::new();
    for i in 0..sa.len() / 4 {
        if sa[4 * i..4 * i + 4] != sb[4 * i..4 * i + 4] {
            assert!(
                i < (M / BLK) * rows,
                "scale word {i} past the routed rows changed"
            );
            s.push((i / rows, i % rows));
        }
    }
    (o, s)
}

/// The dependency claim: output (p, ch) of expert g = expert[p / 8] depends exactly on token
/// perm[p]'s row of hs and its scales hss[.][tok], on weight rows w1[g][ch'] of ch's 128-channel
/// block and their scales (the block's amax sets the e4m3 output scale), and on nothing else.
/// Each perturbation changes exactly the outputs (and scales) the claim names.
#[test]
fn r3b_bmm_dependency_claim() {
    if vendor_missing(&["r3b_trtllm_bmm"]) {
        return;
    }
    let case = bmm();
    let x = inputs(&case);
    let (base, fs, ok) = run(&case, "good");
    assert!(ok && fs.iter().all(|f| f.kind == "contract"));
    let base = base.outputs();
    let rows_of_token =
        |t: usize| -> Vec<usize> { (0..x.rows).filter(|&p| x.perm[p] as usize == t).collect() };
    let rows_of_expert = |g: usize| -> Vec<usize> {
        (0..x.rows)
            .filter(|&p| x.expert[p / TILE] as usize == g)
            .collect()
    };
    let rows_changed = |o: &[(usize, usize)]| -> Vec<usize> {
        let mut r: Vec<usize> = o.iter().map(|&(p, _)| p).collect();
        r.dedup();
        r
    };
    // A token's activations: exactly its two routed rows, every block.
    let t = 5;
    let mut hs = x.hs.clone();
    for (i, b) in hs[t * H..(t + 1) * H].iter_mut().enumerate() {
        *b = ((i * 37 + 11) % 0x70) as u8 | if i % 3 == 0 { 0x80 } else { 0 }; // finite e4m3
    }
    let (o, s) = changed(&base, &run_with(&case, "hs", &hs, "hs"), x.rows);
    assert_eq!(rows_changed(&o), rows_of_token(t));
    assert_eq!(rows_of_token(t).len(), 2);
    assert!(s.iter().all(|&(_, p)| x.perm[p] as usize == t) && s.len() == 2 * (M / BLK));
    // A token's K-block 1 scale: its rows.
    let t = 9;
    let mut hss = case.buffer("hss").unwrap().init_bytes();
    let i = (x.t + t) * 4;
    let v = f32::from_le_bytes(hss[i..i + 4].try_into().unwrap()) * 1.75;
    hss[i..i + 4].copy_from_slice(&v.to_le_bytes());
    let (o, _) = changed(&base, &run_with(&case, "hss", &hss, "hss"), x.rows);
    assert_eq!(rows_changed(&o), rows_of_token(t));
    // One weight row of expert 2: its channel in all of expert 2's rows; other changes only
    // in that channel's block (the amax), and only in expert 2's rows.
    let (g, ch) = (2, 37);
    let mut w1 = x.w1.clone();
    for (i, b) in w1[(g * M + ch) * H..(g * M + ch + 1) * H]
        .iter_mut()
        .enumerate()
    {
        *b = ((i * 53 + 7) % 0x70) as u8 | if i % 2 == 0 { 0x80 } else { 0 };
    }
    let (o, s) = changed(&base, &run_with(&case, "w1", &w1, "w1"), x.rows);
    assert_eq!(rows_changed(&o), rows_of_expert(g));
    assert!(o.iter().all(|&(_, c)| c / BLK == ch / BLK), "{o:?}");
    assert!(rows_of_expert(g).iter().all(|&p| o.contains(&(p, ch))));
    assert!(
        s.iter()
            .all(|&(blk, p)| blk == ch / BLK && x.expert[p / TILE] as usize == g)
    );
    // A weight block scale (expert 1, channel block 1, K-block 0): that block of expert 1.
    let (g, blk) = (1, 1);
    let mut w1s = case.buffer("w1s").unwrap().init_bytes();
    let i = ((g * (M / BLK) + blk) * (H / BLK)) * 4;
    let v = f32::from_le_bytes(w1s[i..i + 4].try_into().unwrap()) * 1.5;
    w1s[i..i + 4].copy_from_slice(&v.to_le_bytes());
    let (o, s) = changed(&base, &run_with(&case, "w1s", &w1s, "w1s"), x.rows);
    assert_eq!(rows_changed(&o), rows_of_expert(g));
    assert!(o.iter().all(|&(_, c)| c / BLK == blk));
    assert!(
        s.iter()
            .all(|&(b, p)| b == blk && x.expert[p / TILE] as usize == g)
    );
}

/// Partial tiles (fixtures/r3b_trtllm_bmm_partial: 8 tokens routed at random, so tiles hold
/// 1-4 tokens): the scale loader skips the padded tokens' scales, the epilogue still reads
/// them (S-INIT: a violation); GB300 and the simulator agree bitwise (the TMA store clips the
/// padded rows and the scale stores are guarded), with and without scrubbed shared memory.
#[test]
fn r3b_bmm_partial_tiles_read_unwritten_scales() {
    if vendor_missing(&["r3b_trtllm_bmm_partial"]) {
        return;
    }
    let case = Case::load(root().join("fixtures/r3b_trtllm_bmm_partial")).unwrap();
    let (m, fs, ok) = run(&case, "good");
    assert!(ok);
    assert_eq!(
        kinds(&fs, ok),
        ["contract", "violation"]
            .into_iter()
            .map(String::from)
            .collect()
    );
    assert!(
        fs.iter()
            .any(|f| f.kind == "violation" && f.msg.contains("sass:0x29c0")),
        "{fs:#?}"
    );
    let report = case.expect_report(&m.outputs()).unwrap();
    assert!(
        report.iter().all(|l| l.contains("bitwise equal")),
        "{report:?}"
    );
}

/// K = 2048 (fixtures/r3b_trtllm_bmm_k2048): 16 K-blocks through 8 stages, so every stage's
/// mbarriers complete phases 0 and 1. Certified (the producers' try-acquire peeks are decided
/// by the peek walk, `wellsync::walk_dead`), contracts only, = GB300 bitwise, and the numerics
/// claim holds for all outputs. B73 (the weight loader's empty-barrier parity not flipped on
/// the wrap) re-arms a stage barrier whose phase is pending (fault) and deadlocks.
#[test]
fn r3b_bmm_k2048_stages_wrap() {
    if vendor_missing(&["r3b_trtllm_bmm_k2048"]) {
        return;
    }
    let case = Case::load(root().join("fixtures/r3b_trtllm_bmm_k2048")).unwrap();
    let k = kernel(&case, "good");
    let launch = build_launch(&case, &k).unwrap();
    let c = ptxv_sim::wellsync::certify(Machine::new(Arc::new(k), Arc::new(launch)));
    let v = c
        .verdict
        .unwrap_or_else(|| panic!("not certified: {:?}", c.reasons));
    assert!(v.complete);
    assert_eq!(v.classes(), ["contract"], "{:#?}", v.findings);
    let report = case.expect_report(&v.outputs[0]).unwrap();
    assert!(
        report.len() == 2 && report.iter().all(|l| l.contains("bitwise equal")),
        "{report:?}"
    );
    assert_eq!(inputs(&case).h, 2048);
    let bad = bound_violations(&case, &v.outputs[0]);
    assert!(
        bad.is_empty(),
        "{} outputs off: {:?}",
        bad.len(),
        &bad[..bad.len().min(5)]
    );
    let (_, fs, ok) = run(&case, "B73");
    let got = kinds(&fs, ok);
    for e in &case.variant("B73").unwrap().expect {
        assert!(got.contains(e), "B73: want {e}, got {got:?}");
    }
}
