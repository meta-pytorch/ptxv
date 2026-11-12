/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Rung 2e: the equivalence demo gives each candidate its expected level (`expect` in the
//! equiv.toml): the other front-end L0, a reordered reduction L1, a dropped K slice not L1,
//! an add.rz epilogue L1 but not L2, a data-dependent branch unproven.

#[test]
fn r2e_tile_levels() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["equiv", "fixtures/r2e_tile/equiv.toml", "--no-hw"])
        .current_dir(&root)
        .env("PTXV_QUIET", "1")
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "{text}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(text.matches("expected").count(), 5, "{text}");
}

/// Third-party kernel binaries are not distributed (fixtures/VENDOR_FIXTURES.toml): true when
/// one that `file` needs is missing (the test then skips).
fn vendor_missing(file: &str) -> bool {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    ptxv_case::skip_missing_vendor_files(&root.join("fixtures"), &[file])
}

/// Run `ptxv equiv` on a fixture's equiv.toml (no GPU) and check every expectation held.
fn equiv_ok(file: &str, expectations: usize) -> String {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    // A private output directory (tests run in parallel).
    let dir = std::env::temp_dir().join(format!(
        "ptxv-test-equiv-{}-{}",
        file.replace('/', "_"),
        std::process::id()
    ));
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["equiv", file, "--no-hw", "--out", dir.to_str().unwrap()])
        .current_dir(&root)
        .env("PTXV_QUIET", "1")
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    assert!(
        out.status.success(),
        "{text}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert_eq!(text.matches(": ok").count(), expectations, "{text}");
    assert!(!text.contains("MISMATCH"), "{text}");
    text
}

/// Input leaves through plain moves: torch.cat vs a Triton cat is L0 (identical bits).
#[test]
fn cat_is_bitwise() {
    let t = equiv_ok("fixtures/c_equiv_cat/equiv.toml", 1);
    assert!(t.contains("L0 bitwise               PROVEN"), "{t}");
}

/// Reassociated sums: torch.sum vs a Triton butterfly row sum, a split-K row sum (two
/// launches) and an fp32-atomic row sum, each L1 with its bound, and no random trial above it.
#[test]
fn row_sums_are_bounded() {
    let t = equiv_ok("fixtures/c_equiv_sum/equiv.toml", 6);
    assert_eq!(t.matches(", 0 over the bound").count(), 3, "{t}");
    assert!(t.contains("relies on: N-ATOMIC"), "{t}");
    // MISS 98's exemption: the atomic row sum's outputs (one order of the 4 adds each) are
    // not held to its capture (the GPU's order); their bound, which covers every order, is.
    assert!(
        t.contains(
            "candidate: simulated output not held to the captured GPU output at 4 elements that depend on the order of float atomic adds"
        ),
        "{t}"
    );
    assert!(
        t.contains("equivalence scope: primary output buffers"),
        "{t}"
    );
    let dir = std::env::temp_dir().join(format!(
        "ptxv-test-equiv-fixtures_c_equiv_sum_equiv.toml-{}",
        std::process::id()
    ));
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("equiv.json")).unwrap()).unwrap();
    assert!(
        report[0]["output_scope"]["reference"]["checked"].is_string(),
        "{report}"
    );
    assert!(
        !report[0]["output_scope"]["candidate"]["unchecked"]
            .as_array()
            .unwrap()
            .is_empty(),
        "{report}"
    );
}

/// Reassociated prefix sums: torch.cumsum vs Triton tl.cumsum.
#[test]
fn cumsums_are_bounded() {
    let t = equiv_ok("fixtures/c_equiv_cumsum/equiv.toml", 2);
    assert!(t.contains(", 0 over the bound"), "{t}");
}

/// Softmax: torch vs Triton two-pass and online softmax, L1 in the softmax normal form with
/// a bound (exhaustive slice sweeps from the fixture's cache).
#[test]
fn softmaxes_are_bounded() {
    let t = equiv_ok("fixtures/c_equiv_softmax/equiv.toml", 4);
    assert_eq!(t.matches("softmax form").count(), 2, "{t}");
    assert_eq!(t.matches(", 0 over the bound").count(), 2, "{t}");
}

/// Normalizations: torch vs Triton RMS norm (rsqrt of the mean square) and layer norm (Welford
/// against two-pass, rsqrt against 1 / sqrt), f32, in the norm form (`norm.rs`): RMS norm L1
/// with a bound relative to |y| (kappa <= 1 proven), layer norm L1~ (Welford's f32 constants
/// 1/3 change the exact polynomial by ~2^-19 relative, charged in the bound), and no random
/// trial above its bound.
#[test]
fn norms_are_bounded() {
    let t = equiv_ok("fixtures/c_equiv_rmsnorm/equiv.toml", 2);
    assert!(t.contains("L1 same function         PROVEN"), "{t}");
    assert!(t.contains("kappa <= 1 (proven)"), "{t}");
    assert!(t.contains(", 0 over the bound"), "{t}");
    assert!(t.contains("0 output-trials not covered"), "{t}");
    let t = equiv_ok("fixtures/c_equiv_layernorm/equiv.toml", 2);
    assert!(t.contains("strongest level: L1~"), "{t}");
    assert!(t.contains(", 0 over the bound"), "{t}");
    assert!(t.contains("0 output-trials not covered"), "{t}");
}

/// Attention (f32): a normalize-first reference against the flash form (running max,
/// accumulator rescaled by exp(m_old - m_new)), L1 in the attention normal form
/// sum_j exp(x_j - max) v_j / sum_j exp(x_j - max), scores given and scores computed (q k^T),
/// and no random trial above the bound.
#[test]
fn attention_core_is_bounded() {
    let t = equiv_ok("fixtures/c_equiv_attn/equiv.toml", 2);
    assert!(t.contains("L1 same function         PROVEN"), "{t}");
    assert!(t.contains("attention form"), "{t}");
    assert!(t.contains(", 0 over the bound"), "{t}");
    // With the scores computed (q k^T): score atoms and the e^(2 Delta) term.
    let t = equiv_ok("fixtures/c_equiv_attn/full.toml", 2);
    assert!(t.contains("L1 same function         PROVEN"), "{t}");
    assert!(t.contains("Q)u * V1"), "{t}");
    assert!(t.contains(", 0 over the bound"), "{t}");
}

/// Elementwise mode (docs/equiv-elementwise.md): torch vs Triton tanh-GELU, erf-GELU and SiLU
/// in f32 and erf-GELU in bf16, each swept over every input value (the committed sweeps), the
/// level EW, and each kernel against the true function in f64.
#[test]
fn elementwise_pairs_are_swept() {
    for (f, func) in [
        ("gelu_tanh", "gelu_tanh"),
        ("gelu_erf", "gelu"),
        ("silu", "silu"),
        ("gelu_bf16", "gelu"),
    ] {
        let t = equiv_ok(&format!("fixtures/c_equiv_elementwise/{f}.toml"), 1);
        assert!(t.contains("every output i is f(x[i])"), "{t}");
        assert!(t.contains(&format!("vs {func} in f64")), "{t}");
        let all = if f.ends_with("bf16") {
            "2^16 bf16"
        } else {
            "2^32 f32"
        };
        assert!(t.contains(&format!("over all {all} inputs")), "{t}");
    }
}

/// bf16 GEMM on tensor cores (docs/equiv-mma.md): torch.mm (cuBLAS nvjet, tcgen05, TMA store)
/// vs a Triton matmul, MMA nodes under the calibrated model's contract (N-MMA) and bf16
/// output roundings (N-ROUND): the same K order is L2 (bitwise up to the sign of zero), the
/// reversed K order L1 with the round form's bound, and no random trial above it.
#[test]
fn bf16_gemm_is_bounded() {
    if vendor_missing("fixtures/c_equiv_mm_bf16/equiv.toml") {
        return;
    }
    let t = equiv_ok("fixtures/c_equiv_mm_bf16/equiv.toml", 2);
    assert!(t.contains("L2 no less accurate      PROVEN"), "{t}");
    assert!(t.contains("L1 same function         PROVEN"), "{t}");
    assert!(t.contains(", 0 over the bound"), "{t}");
}

/// bf16 flash attention on tensor cores (docs/equiv-mma.md): torch SDPA's flash kernel (sm80
/// HMMA, P rounded to bf16) vs the Triton tcgen05 flash kernel, L1 in the attention normal
/// form with computed scores for all 8192 outputs, and no random trial above the bound.
#[test]
#[ignore = "heavy: 87 s"]
fn bf16_attention_is_bounded() {
    if vendor_missing("fixtures/c_equiv_sdpa/equiv.toml") {
        return;
    }
    let t = equiv_ok("fixtures/c_equiv_sdpa/equiv.toml", 1);
    assert!(t.contains("L1 same function         PROVEN"), "{t}");
    assert!(t.contains("out[0..=8191]: L1 x8192; attention form"), "{t}");
    assert!(t.contains(", 0 over the bound"), "{t}");
}

/// MXF8 GEMM (docs/equiv-mma.md §5): with the operand and scale bytes as leaves and the
/// contract derived from the narrow model (`mma_lp::lp_eps`), the CuTeDSL kernel against itself
/// is L0 for every input with an own bound of 255.5u per term against the exact MXF8 GEMM
/// (8 K = 32 steps; was eps >= 2 per step), no random-input trial (random e4m3 codes with
/// subnormals, random ue8m0 scales) above it; the B30 mutant is not L1.
#[test]
#[ignore = "heavy: 89 s"]
fn mxf8_gemm_is_bounded() {
    if vendor_missing("fixtures/c_equiv_mxf8/equiv.toml") {
        return;
    }
    let t = equiv_ok("fixtures/c_equiv_mxf8/equiv.toml", 2);
    assert!(t.contains("L0 bitwise               PROVEN"), "{t}");
    assert!(t.contains("reference 255.50u, candidate 255.50u"), "{t}");
    assert!(t.contains("0 over the bound"), "{t}");
    let own = t
        .split("each kernel vs exact: at most ")
        .nth(1)
        .and_then(|s| s.split(' ').next())
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap();
    assert!(own > 0.0 && own < 1.0, "{t}");
}

/// MXF8 GEMM across implementations (docs/equiv-mma.md §5): CuTeDSL against cuBLASLt's
/// torch._scaled_mm is L0~0, L0 up to the sign of zero, and not L0 (cuBLAS adds +0 in its
/// alpha / beta epilogue; its BAR.SYNC waits on the bulk read scoreboard in the control word).
/// Until MISS 109 the pair was reported L0.
#[test]
#[ignore = "heavy: 60 s"]
fn mxf8_cublas_is_bitwise_up_to_zero_signs() {
    if vendor_missing("fixtures/c_equiv_mxf8_cublas/equiv.toml") {
        return;
    }
    let t = equiv_ok("fixtures/c_equiv_mxf8_cublas/equiv.toml", 1);
    assert!(
        t.contains("L0 bitwise               FAILS at 16384 of 16384 outputs"),
        "{t}"
    );
    assert!(t.contains("L0~0 up to the sign of zero PROVEN"), "{t}");
    assert!(
        t.contains("L0~0 at 16384 outputs: one kernel's output is the other's plus +0"),
        "{t}"
    );
    assert!(t.contains("=> strongest level: L0~0\n"), "{t}");
    assert!(t.contains("0 over the bound"), "{t}");
}

/// NVFP4 GEMM (docs/equiv-mma.md §5): CuTeDSL (one CTA) against cuBLASLt's torch._scaled_mm,
/// a cta_group::2 kernel (pair block-scaled MMA, UTCCP.2CTA, LDTM.16dp128bit, LOP3.PAND), is
/// L0~0, L0 up to the sign of zero, and not L0; each kernel's own bound is 9.88u x sum |a b sa
/// sb| (ue4m3 scales nonzero), no random trial above it.
#[test]
#[ignore = "heavy: 109 s"]
fn nvf4_cublas_is_bitwise_up_to_zero_signs() {
    if vendor_missing("fixtures/c_equiv_nvf4_cublas/equiv.toml") {
        return;
    }
    let t = equiv_ok("fixtures/c_equiv_nvf4_cublas/equiv.toml", 1);
    assert!(
        t.contains("L0 bitwise               FAILS at 16384 of 16384 outputs"),
        "{t}"
    );
    assert!(t.contains("L0~0 up to the sign of zero PROVEN"), "{t}");
    assert!(
        t.contains("L0~0 at 16384 outputs: one kernel's output is the other's plus +0"),
        "{t}"
    );
    assert!(t.contains("=> strongest level: L0~0\n"), "{t}");
    assert!(t.contains("reference 9.88u, candidate 9.88u"), "{t}");
    assert!(t.contains("0 over the bound"), "{t}");
    assert!(t.contains("0 outputs differed bitwise"), "{t}");
}

/// NVFP4 against a dequantize-then-bf16 GEMM (Triton: cvt of each code and ue4m3 scale, f32
/// product, bf16 rounding, bf16 tl.dot): the dequantization is exact for every input
/// (N-EXACT), so it is the same function (L1) with only the two kernels' MMA contracts in the
/// bound, 169.38u x sum |a b sa sb|, no random trial above it.
#[test]
#[ignore = "heavy: 54 s"]
fn nvf4_dequant_bf16_is_bounded() {
    if vendor_missing("fixtures/c_equiv_nvf4_dq/equiv.toml") {
        return;
    }
    let t = equiv_ok("fixtures/c_equiv_nvf4_dq/equiv.toml", 1);
    assert!(t.contains("L1 same function         PROVEN"), "{t}");
    assert!(t.contains("reference 9.88u, candidate 159.50u"), "{t}");
    assert!(t.contains("bound: |C - R| <= 169.38u * sum|terms|"), "{t}");
    assert!(t.contains("0 over the bound"), "{t}");
}

/// ... and against Triton tl.dot_scaled: L0 in K order; K blocks of 32 reversed is L1 with
/// 306u x sum |a b sa sb| (Thm 4 over both kernels' per-monomial bounds), no trial above it.
#[test]
#[ignore = "heavy: equiv subprocess, 3 certified sides; 114 s"]
fn mxf8_triton_is_bounded() {
    if vendor_missing("fixtures/c_equiv_mxf8_triton/equiv.toml") {
        return;
    }
    let t = equiv_ok("fixtures/c_equiv_mxf8_triton/equiv.toml", 2);
    assert!(t.contains("L0 bitwise               PROVEN"), "{t}");
    assert!(t.contains("expected L1: ok"), "{t}");
    assert!(t.contains("306.00u * sum|terms|"), "{t}");
    assert!(!t.contains("MISMATCH"), "{t}");
    assert!(t.matches("0 over the bound").count() == 2, "{t}");
}

/// MXF4 (FP4 nibble leaves, 4-bit tensor maps): CuTeDSL against Triton tl.dot_scaled "e2m1" is
/// L0 in K order; K blocks of 64 reversed is L1 with 10.04u x sum |a b sa sb| (each kernel's
/// own bound 8.03u), no trial above it.
#[test]
#[ignore = "heavy: 86 s"]
fn mxf4_triton_is_bounded() {
    if vendor_missing("fixtures/c_equiv_mxf4_triton/equiv.toml") {
        return;
    }
    let t = equiv_ok("fixtures/c_equiv_mxf4_triton/equiv.toml", 2);
    assert!(t.contains("L0 bitwise               PROVEN"), "{t}");
    assert!(t.contains("reference 8.03u, candidate 8.03u"), "{t}");
    assert!(t.contains("10.04u * sum|terms|"), "{t}");
    assert!(t.matches("0 over the bound").count() == 2, "{t}");
}

/// f16 outputs: two f32 sum orders rounded to f16 (synthetic PTX, SASS front-end) are L1 in
/// the 16-bit output form; one f16 ulp where sum |terms| <= 512 |y|; no trial above the
/// per-input bound.
#[test]
fn f16_outputs_are_bounded() {
    let t = equiv_ok("fixtures/c_equiv_f16/equiv.toml", 1);
    assert!(t.contains("f16 output form"), "{t}");
    assert!(t.contains("sum|terms| <= 512 |y|"), "{t}");
    assert!(t.contains("0 over the bound"), "{t}");
}

/// TF32 GEMMs (N-TF32, docs/equiv-mma.md §6) against an f32 SIMT GEMM: the 3xTF32 splits
/// (CUTLASS's big + small, K forward and reversed) are L1 with their three products per (a, b)
/// matched (k <= 240u), 1xTF32 is L1 through each operand's TF32 truncation (~2^-9 per
/// product), and no random trial is above its bound; the splits' finiteness guards rely on
/// N-FINITE. A dropped correction (2xTF32) leaves the other unmatched: not proven.
#[test]
fn tf32_gemms_are_bounded() {
    let t = equiv_ok("fixtures/c_equiv_tf32/equiv.toml", 4);
    assert_eq!(
        t.matches("L1 same function         PROVEN").count(),
        3,
        "{t}"
    );
    assert!(t.contains("bound: |C - R| <= 239.52u * sum|terms|"), "{t}");
    assert!(t.contains("bound: |C - R| <= 183.52u * sum|terms|"), "{t}");
    assert!(t.contains("condition: N-TF32"), "{t}");
    assert!(t.contains("relies on: N-FINITE (48 comparisons"), "{t}");
    assert!(
        t.contains("a 3xTF32 correction product without its hi * hi product"),
        "{t}"
    );
    assert_eq!(t.matches(", 0 over the bound").count(), 3, "{t}");
}

/// N-FINITE negatives: a sign test of the split's small part and a finiteness test of a value
/// finite inputs do not make finite stay data-dependent guards (no proof).
#[test]
fn tf32_data_dependent_guards_block() {
    let t = equiv_ok("fixtures/c_equiv_tf32/neg.toml", 2);
    assert!(t.contains("data-dependent: guard of"), "{t}");
    assert_eq!(t.matches("relies on: N-FINITE").count(), 1, "{t}");
}

/// N-FMAX: row maxima through CUTLASS's atomicMaxFloat (signed max / unsigned min by the sign
/// bit, into shared memory) are L0 against a sequential fmaxf; the same with the choice made
/// by value > 0.5 keeps its data-dependent branch.
#[test]
fn float_atomic_max_is_the_maximum() {
    let t = equiv_ok("fixtures/c_equiv_fmax/equiv.toml", 2);
    assert!(t.contains("L0 bitwise               PROVEN"), "{t}");
    assert!(
        t.contains("relies on: N-FMAX (1 float atomic max branches"),
        "{t}"
    );
    assert!(
        t.contains("data-dependent: guard of sass:0x02a0 `BRA"),
        "{t}"
    );
}

/// f32 SDPA (CUTLASS fmha_cutlassF_f32, 3xTF32): torch.compile and eager launch the same kernel;
/// L0 for every admissible input. The row maximum (atomicMaxFloat into shared memory), the
/// online-softmax branch m_prime < mi, the ex2 range guards kept in a flag word, the rcp.rn
/// variants and P through shared memory to the P·V MMAs all have nodes or slices.
#[test]
#[ignore = "heavy: 96 s"]
fn sdpa_f32_is_bitwise() {
    if vendor_missing("fixtures/c_cve_sdpa_f32/equiv.toml") {
        return;
    }
    let t = equiv_ok("fixtures/c_cve_sdpa_f32/equiv.toml", 1);
    assert!(t.contains("L0 bitwise               PROVEN"), "{t}");
    assert!(!t.contains("precondition"), "{t}");
    assert!(
        t.contains("relies on: N-FMAX (4 float atomic max branches"),
        "{t}"
    );
    assert!(t.contains("relies on: N-FINITE"), "{t}");
}

/// Online-softmax bookkeeping and exact taint (fixtures/c_equiv_online): the kernel is L0
/// against itself (the -inf start's rescaling, masked exponentials, constants through a shuffle
/// and a float2 store decide nothing); a finite m_prime start, a mask chosen by the data, a
/// shuffle from data lanes and the float2's data word each keep a data-dependent guard.
#[test]
fn online_bookkeeping_taint_is_exact() {
    let t = equiv_ok("fixtures/c_equiv_online/equiv.toml", 5);
    assert!(t.contains("L0 bitwise               PROVEN"), "{t}");
    for (case, guard) in [
        ("c_online_bad_mp", "guard of sass:0x0bf0 `BRA"),
        ("c_online_bad_sel", "guard of sass:0x0a10 `BRA"),
        ("c_online_bad_shfl", "guard of sass:0x2010 `STG"),
        ("c_online_bad_word", "guard of sass:0x2020 `STG"),
    ] {
        let part = t.split("candidate ").find(|p| p.starts_with(case)).unwrap();
        assert!(part.contains(guard), "{case}: {part}");
    }
}

/// MISS 65 (fixtures/errdist/m65vote/equiv.toml): lanes 16..31 branch on a ballot of lanes
/// 0..15's input-derived predicate. On the captured input the ballot is 0 and the DAGs are
/// equal, but an input with a nonzero ballot takes the other branch. Each lane's ballot result
/// carried only its own predicate's taint, so L0 was proven for every input; the branch on the
/// ballot is a data-dependent precondition now (no proof; the rounding it guards has no node).
#[test]
fn ballot_decided_branch_blocks_the_proof() {
    let t = equiv_ok("fixtures/errdist/m65vote/equiv.toml", 1);
    assert!(t.contains("data-dependent: guard of"), "{t}");
}

#[test]
fn empty_output_list_is_rejected() {
    let source =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/r2e_tile/case.toml");
    let manifest = std::fs::read_to_string(source).unwrap();
    let dir = std::env::temp_dir().join(format!("ptxv-output-scope-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("case.toml"),
        manifest.replace("outputs = [\"D\"]", "outputs = []"),
    )
    .unwrap();
    std::fs::write(
        dir.join("equiv.toml"),
        "candidates = []\nreference = { case = \".\", variant = \"good\", front = \"ptx\" }\n",
    )
    .unwrap();
    let output = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["equiv", dir.join("equiv.toml").to_str().unwrap(), "--no-hw"])
        .output()
        .unwrap();
    std::fs::remove_dir_all(&dir).unwrap();
    let error = String::from_utf8_lossy(&output.stderr);
    assert!(!output.status.success());
    assert!(
        error.contains("equivalence requires an output buffer"),
        "{error}"
    );
}

#[test]
fn captured_zero_without_half_history_is_reported_unproven() {
    let dir = std::env::temp_dir().join(format!("ptxv-untracked-half-zero-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("kernel.ptx"),
        r#"
.version 8.8
.target sm_103a
.address_size 64
.visible .entry k(.param .u64 input, .param .u64 output) {
    .reg .b64 %p<2>;
    .reg .b32 %r<2>;
    ld.param.u64 %p0, [input];
    ld.param.u64 %p1, [output];
    ld.global.u32 %r0, [%p0];
    shr.u32 %r1, %r0, 1; // @SHIFT
    st.global.u16 [%p1], %r1;
    ret;
}
"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("case.toml"),
        r#"
name = "untracked-half-zero"
description = "input-derived zero without a half numeric history"
ptx = "kernel.ptx"
entry = "k"
grid = [1, 1, 1]
block = [1, 1, 1]
outputs = ["out"]
[[buffers]]
name = "x"
dtype = "f32"
shape = [1]
init = "zeros"
[[buffers]]
name = "out"
dtype = "bf16"
shape = [1]
init = "zeros"
[[params]]
kind = "ptr"
buffer = "x"
[[params]]
kind = "ptr"
buffer = "out"
[[variants]]
id = "good"
expect = ["clean"]
[[variants]]
id = "mutant"
expect = ["clean"]
edits = [{marker = "@SHIFT", find = "%r0, 1", replace = "%r0, 2"}]
"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("equiv.toml"),
        r#"
max_states = 100
trials = 0
reference = {case = ".", variant = "good", front = "ptx"}
[[candidates]]
case = "."
variant = "mutant"
front = "ptx"
expect = "L0"
"#,
    )
    .unwrap();
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args([
            "equiv",
            dir.join("equiv.toml").to_str().unwrap(),
            "--no-hw",
            "--out",
            dir.join("result").to_str().unwrap(),
        ])
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    assert!(!out.status.success(), "{text}");
    assert!(
        text.contains("L0 bitwise") && text.contains("FAILS"),
        "{text}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!text.contains("PROVEN"), "{text}");
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn incomplete_numeric_output_elements_are_refused() {
    let dir = std::env::temp_dir().join(format!("ptxv-output-tail-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("kernel.ptx"),
        r#"
.version 8.8
.target sm_103a
.address_size 64
.visible .entry k(.param .u64 output) {
    .reg .b64 %p;
    ld.param.u64 %p, [output];
    st.global.u8 [%p], 0;
    ret;
}
"#,
    )
    .unwrap();
    std::fs::write(
        dir.join("equiv.toml"),
        r#"
trials = 0
reference = {case = ".", variant = "good", front = "ptx"}
candidates = [{case = ".", variant = "good", front = "ptx"}]
"#,
    )
    .unwrap();
    for bytes in [1, 3, 5, 7] {
        std::fs::write(
            dir.join("case.toml"),
            format!(
                r#"
name = "output-tail"
description = "all output bytes must be covered"
ptx = "kernel.ptx"
entry = "k"
grid = [1,1,1]
block = [1,1,1]
outputs = ["out"]
params = [{{kind = "ptr", buffer = "out"}}]
[[buffers]]
name = "out"
dtype = "u8"
shape = [{bytes}]
init = "zeros"
[[variants]]
id = "good"
expect = ["clean"]
"#
            ),
        )
        .unwrap();
        let result = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
            .args([
                "equiv",
                dir.join("equiv.toml").to_str().unwrap(),
                "--no-hw",
                "--out",
                dir.join("result").to_str().unwrap(),
            ])
            .env("CUDA_VISIBLE_DEVICES", "")
            .output()
            .unwrap();
        let error = String::from_utf8_lossy(&result.stderr);
        assert!(
            !result.status.success(),
            "{}",
            String::from_utf8_lossy(&result.stdout)
        );
        assert!(
            error.contains(&format!("cannot cover all {bytes} bytes")),
            "{error}"
        );
        assert!(!error.contains("panicked"), "{error}");
    }
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn async_memory_rewrites_do_not_reuse_old_histories() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = std::env::temp_dir().join(format!("ptxv-async-history-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for case in ["e_bulk_load_history", "e_bulk_shared_half", "e_clc_history"] {
        for front in ["ptx", "sass"] {
            for segments in [false, true] {
                let source = root.join("fixtures").join(case);
                let config = dir.join("equiv.toml");
                std::fs::write(
                    &config,
                    format!(
                        r#"
max_states = 10000
trials = 0
segments = {segments}
elementwise = false
reference = {{case = {source:?}, variant = "reference", front = "{front}", certify = true}}
candidates = [{{case = {source:?}, variant = "candidate", front = "{front}", certify = true}}]
"#
                    ),
                )
                .unwrap();
                let result = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
                    .args([
                        "equiv",
                        config.to_str().unwrap(),
                        "--no-hw",
                        "--out",
                        dir.join("result").to_str().unwrap(),
                    ])
                    .env("CUDA_VISIBLE_DEVICES", "")
                    .env("PTXV_NO_VCACHE", "1")
                    .output()
                    .unwrap();
                assert!(
                    result.status.success(),
                    "{case}/{front}/{segments}: {}\n{}",
                    String::from_utf8_lossy(&result.stdout),
                    String::from_utf8_lossy(&result.stderr)
                );
                let report: serde_json::Value =
                    serde_json::from_slice(&std::fs::read(dir.join("result/equiv.json")).unwrap())
                        .unwrap();
                assert_eq!(
                    report[0]["report"]["levels"][0]["holds"], false,
                    "{case}/{front}/{segments}: {report}"
                );
            }
        }
    }
    std::fs::remove_dir_all(dir).unwrap();
}

/// ASSUME(E-CLOCK): the equivalence preconditions go through the launch check's wrapper
/// (smid::check_placements), so a clock-dependent output (fixtures/e_clock_data: a select on
/// a clock comparison stored) is not "one output per input": the precondition is unproven.
/// Before, `ptxv equiv` listed no failed precondition for it.
#[test]
fn clock_dependent_output_is_an_unmet_precondition() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let dir = std::env::temp_dir().join(format!("ptxv-test-equiv-clock-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let case = root.join("fixtures/e_clock_data");
    let toml = format!(
        "max_states = 200000\nreference = {{ case = {c:?}, variant = \"good\", front = \"ptx\" }}\n\n\
         [[candidates]]\ncase = {c:?}\nvariant = \"good\"\nfront = \"ptx\"\n",
        c = case.to_str().unwrap()
    );
    let file = dir.join("equiv.toml");
    std::fs::write(&file, toml).unwrap();
    let out_dir = dir.join("out");
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args([
            "equiv",
            file.to_str().unwrap(),
            "--no-hw",
            "--out",
            out_dir.to_str().unwrap(),
        ])
        .current_dir(&root)
        .env("PTXV_QUIET", "1")
        .output()
        .unwrap();
    let text = String::from_utf8_lossy(&out.stdout);
    let json = std::fs::read_to_string(out_dir.join("equiv.json"))
        .unwrap_or_else(|e| panic!("{e}: {text}\n{}", String::from_utf8_lossy(&out.stderr)));
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    let pre = v[0]["preconditions"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    assert!(
        pre.iter()
            .any(|p| p.as_str().is_some_and(|s| s.contains("inconclusive"))),
        "{json}"
    );
}

/// MISS 97 (fixtures/c_equiv_rsign): add.sat.s32 on the bits (ptxas: PLOP3 on `R.SIGN` /
/// `UR.SIGN` inputs) against the function a slice reading those predicates as nonzero
/// computes for it. Elementwise mode sweeps all 2^16 bf16 inputs: they differ on 16,385, so
/// the level is EW, not L0 (before, the slices made them equal: L0).
#[test]
fn sign_predicate_slices_are_swept_apart() {
    let t = equiv_ok("fixtures/c_equiv_rsign/equiv.toml", 1);
    assert!(t.contains("bitwise equal on 49151 of 65536 inputs"), "{t}");
    assert!(t.contains("=> strongest level: EW"), "{t}");
}

/// MISS 98 (fixtures/c_equiv_captured): the same kernel twice, the candidate's captured GPU
/// output differing from the model's in one word. Equiv refuses the level in rounding-DAG
/// mode (the candidate's simulated output differs from its capture) and in elementwise mode
/// (its element functions differ from its capture); before, both reported L0.
#[test]
fn captured_gpu_mismatch_refuses_the_level() {
    let t = equiv_ok("fixtures/c_equiv_captured/equiv.toml", 1);
    assert!(
        t.contains("candidate: simulated output at the captured inputs: out: differs from"),
        "{t}"
    );
    let t = equiv_ok("fixtures/c_equiv_captured/equiv_ew.toml", 1);
    assert!(
        t.contains("captured GPU outputs (candidate): 1 of 1024 differ from the element functions"),
        "{t}"
    );
    let dir = std::env::temp_dir().join(format!(
        "ptxv-test-equiv-fixtures_c_equiv_captured_equiv_ew.toml-{}",
        std::process::id()
    ));
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("equiv.json")).unwrap()).unwrap();
    assert_eq!(
        report[0]["elementwise"]["level"], "none (contradicted by the captured GPU outputs)",
        "{report}"
    );
}

/// MISS 98 (fixtures/c_equiv_atomic, from its review): float atomic adds exempt from the
/// simulated-vs-captured check only the outputs whose value depends on their order. With an
/// atomic add into a scratch buffer (no output), an L0 pair whose candidate capture has
/// another NaN payload and an L1 pair whose capture is 1 ulp off, +inf or NaN at an output
/// are refused. With four atomic adds into out[1024], a 1-ulp change at out[225] is refused;
/// out[1024] passes its last bit changed (another order of the adds) and is held to the
/// bound (+inf and + 1.0 are refused). Two adds into +0 give the same bits in every order, so
/// that out[1024] is compared like the rest. Before, all of these but the controls without an
/// atomic kept the level.
#[test]
fn atomic_adds_exempt_only_their_outputs() {
    let t = equiv_ok("fixtures/c_equiv_atomic/eq_l0.toml", 3);
    assert_eq!(
        t.matches("3 of 4096 bytes differ, first at byte 28 (sim 0x45, gpu 0x00)")
            .count(),
        2,
        "{t}"
    );
    let t = equiv_ok("fixtures/c_equiv_atomic/eq_l1_out.toml", 6);
    assert_eq!(
        t.matches(
            "candidate: simulated output not held to the captured GPU output at 1 element that depends on the order of float atomic adds"
        )
        .count(),
        5,
        "{t}"
    );
    assert_eq!(
        t.matches("1 of 4100 bytes differ, first at byte 900 (sim 0x00, gpu 0x01)")
            .count(),
        2,
        "{t}"
    );
    assert!(
        t.contains("captured GPU outputs contradict L1: 1 of 1025 outputs are non-finite on one side and differ (first: out[1024])"),
        "{t}"
    );
    assert!(
        t.contains("captured GPU outputs contradict L1: 1 of 1025 outputs are farther apart than the proven bound (first: out[1024])"),
        "{t}"
    );
    let t = equiv_ok("fixtures/c_equiv_atomic/eq_l1_scr.toml", 5);
    assert!(!t.contains("not held to the captured GPU output"), "{t}");
    assert_eq!(
        t.matches("captured GPU outputs contradict L1: the candidate's simulated output differs from its capture")
            .count(),
        4,
        "{t}"
    );
    let t = equiv_ok("fixtures/c_equiv_atomic/eq_l1_two.toml", 2);
    assert!(!t.contains("not held to the captured GPU output"), "{t}");
    assert!(
        t.contains("1 of 4100 bytes differ, first at byte 4096"),
        "{t}"
    );
}

/// MISS 98 (fixtures/c_equiv_atomic/eq_unproven.toml): an unproven level is held to the
/// captures too. The candidate's guard depends on input data (and its stores race), so
/// its L0 is unproven; with its capture changed in one bit the level is refused as
/// contradicted (before, "L0 (unproven)" was kept).
#[test]
fn unproven_levels_are_held_to_the_captures() {
    let t = equiv_ok("fixtures/c_equiv_atomic/eq_unproven.toml", 2);
    assert!(
        t.contains("captured GPU outputs contradict L0 (unproven): the candidate's simulated output differs from its capture"),
        "{t}"
    );
}

/// MISS 110 (fixtures/c_equiv_round16/eq_probes.toml): equiv compares each kernel's simulated
/// output with its capture as `ptxv check` does, within one ulp only in bytes of calibrated
/// f16 / bf16 multimem lanes. The review probes r1h_nvls_ulp (an exact f32 multimem sum, a
/// capture one ulp off) and r1h_nvls_nan (NaN against a NaN with another payload) differ from
/// their captures on both fronts. Until MISS 110 neither difference was reported: every output
/// of a case with a multicast object was compared within one ulp.
#[test]
fn multimem_probes_differ_from_their_captures() {
    let t = equiv_ok("fixtures/c_equiv_round16/eq_probes.toml", 4);
    let line = "candidate: simulated output at the captured inputs: out0: differs from the captured GPU output";
    assert_eq!(t.matches(line).count(), 4, "{t}");
}

/// The first file named `name` under `dir`.
fn find(dir: &std::path::Path, name: &str) -> Option<std::path::PathBuf> {
    for e in std::fs::read_dir(dir).ok()?.flatten() {
        let p = e.path();
        if p.is_dir() {
            if let Some(f) = find(&p, name) {
                return Some(f);
            }
        } else if p.file_name().is_some_and(|n| n == name) {
            return Some(p);
        }
    }
    None
}

/// MISS 110 (N-MULTIMEM-ROUND16 in equiv): fixtures/r1h_nvls_bf16 copied with captured outputs
/// written from its own simulated output, lane 1 of outa0 (an exact bf16 sum: no calibrated
/// rounding) one encoding up. equiv's check of the simulated output against the capture reports
/// the difference on both fronts; with every byte taken as a calibrated lane it would pass as
/// within one ulp.
#[test]
fn exact_multimem_lane_one_ulp_off_differs_from_its_capture() {
    const OUTS: [&str; 8] = [
        "outa0", "outb0", "outc0", "outm0", "outa1", "outb1", "outc1", "outm1",
    ];
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..");
    let src = root.join("fixtures/r1h_nvls_bf16");
    let tmp = std::env::temp_dir().join(format!("ptxv-test-equiv-nvls16-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&tmp);
    let mut eq = format!(
        "reference = {{ case = \"{}\", variant = \"good\", front = \"ptx\" }}\n",
        src.display()
    );
    for front in ["ptx", "sass"] {
        let out = tmp.join(format!("sim-{front}"));
        let o = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
            .arg("check")
            .arg(&src)
            .args(["--variant", "good", "--front", front, "--json"])
            .args(["--no-input-coverage", "--out"])
            .arg(&out)
            .env("PTXV_DUMP_OUTPUTS", "1")
            .env("CUDA_VISIBLE_DEVICES", "")
            .output()
            .unwrap();
        assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stderr));
        let mut sim =
            std::fs::read(find(&out, &format!("good.{front}.output0.bin")).unwrap()).unwrap();
        assert_eq!(sim.len(), 64, "{front}: 8 outputs of 4 bf16");
        let case = tmp.join(format!("case-{front}"));
        std::fs::create_dir_all(case.join("expected")).unwrap();
        std::fs::create_dir_all(case.join("data")).unwrap();
        for f in ["kernel.ptx", "data/in0.bin", "data/in1.bin"] {
            std::fs::copy(src.join(f), case.join(f)).unwrap();
        }
        let mut toml = std::fs::read_to_string(src.join("case.toml")).unwrap();
        for b in OUTS {
            toml = toml.replace(
                &format!("name = \"{b}\"\n"),
                &format!("name = \"{b}\"\nexpect = \"expected/{b}.bin\"\n"),
            );
        }
        std::fs::write(case.join("case.toml"), toml).unwrap();
        // outa0 lane 1: bytes 2..4 of the first output.
        let e = u16::from_le_bytes([sim[2], sim[3]]) + 1;
        sim[2..4].copy_from_slice(&e.to_le_bytes());
        for (i, b) in OUTS.iter().enumerate() {
            std::fs::write(
                case.join(format!("expected/{b}.bin")),
                &sim[8 * i..8 * i + 8],
            )
            .unwrap();
        }
        eq += &format!(
            "\n[[candidates]]\ncase = \"{}\"\nvariant = \"good\"\nfront = \"{front}\"\nexpect = \"none (unproven)\"\n",
            case.display()
        );
    }
    let toml = tmp.join("eq.toml");
    std::fs::write(&toml, eq).unwrap();
    let t = equiv_ok(toml.to_str().unwrap(), 2);
    let line = "candidate: simulated output at the captured inputs: outa0: differs from the captured GPU output";
    assert_eq!(t.matches(line).count(), 2, "{t}");
    let _ = std::fs::remove_dir_all(&tmp);
}

/// MISS 158 (fixtures/c_equiv_local/eq_local.toml): input data keeps its taint through local
/// memory. ref_local64 (x through st/ld.local.b64) and ref_local16 (x's high half through
/// st/ld.local.b16, STL.U16 / LDL.U16 on SASS) branch on it, so the data-dependent precondition
/// fails on both fronts, as for the register-direct control ref_reg. Before, both were L0 PROVEN
/// against cand_add, which GB300 refutes at x = 1.25 + t/256 (1.25 against 2.25).
#[test]
fn local_memory_keeps_input_taint() {
    let t = equiv_ok("fixtures/c_equiv_local/eq_local.toml", 5);
    assert!(!t.contains("PROVEN"), "{t}");
    let line = "precondition: data-dependent: guard of";
    assert_eq!(t.matches(line).count(), 11, "{t}");
}

/// MISS 159 (fixtures/c_equiv_cas/eq_cas.toml): a compare-and-swap whose compare is input data
/// leaves its word input data where the swap fails (at the captured input), so an address that
/// depends on the word fails the precondition on both fronts. Before, the word kept its taint
/// and ref_cas was L0 PROVEN against cand_add, which GB300 refutes at x = 0.5 (1.515625 against
/// 1.5).
#[test]
fn failed_cas_with_input_compare_taints_its_word() {
    let t = equiv_ok("fixtures/c_equiv_cas/eq_cas.toml", 2);
    assert!(!t.contains("PROVEN"), "{t}");
    let line = "precondition: data-dependent: an address / control operand of";
    assert_eq!(t.matches(line).count(), 2, "{t}");
}

/// MISS 98 (fixtures/c_equiv_atomic/eq_offset.toml): captures of part of the output
/// (`expect_offset` 2048; the candidate's also `expect_ignore` of out[600]) are compared where
/// both cases captured them, each output against its own bound. Before, both captures were
/// read from byte 0: 512 pairs, out[512 + i] against the bound of out[i].
#[test]
fn partial_captures_are_compared_where_captured() {
    let t = equiv_ok("fixtures/c_equiv_atomic/eq_offset.toml", 1);
    assert!(
        t.contains(
            "captured GPU outputs: 358 of 511 bitwise equal (513 outputs not captured by both cases)"
        ),
        "{t}"
    );
    let dir = std::env::temp_dir().join(format!(
        "ptxv-test-equiv-fixtures_c_equiv_atomic_eq_offset.toml-{}",
        std::process::id()
    ));
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("equiv.json")).unwrap()).unwrap();
    let cap = &report[0]["captured"];
    assert_eq!(cap["outputs"], 511, "{cap}");
    assert_eq!(cap["uncaptured"], 513, "{cap}");
    assert!(cap["worst_output"].as_u64().unwrap() >= 512, "{cap}");
    assert_eq!(cap["bound_abs"].as_array().unwrap().len(), 1024, "{cap}");
}

/// MISS 109 (fixtures/c_equiv_zero_sign): an output that is the other's plus +0 (cuBLAS's
/// alpha = 1, beta = 0 epilogue) is equal only up to the sign of zero: level L0~0, L0
/// (bitwise) fails at every output, the bound is 0. GB300's captures, +0 against -0 at the 16
/// outputs whose sum is -0, agree with L0~0. b + a stays L0. A capture changed in another bit,
/// or a +0 made -0 (equal to the reference's there; x + (+0) is never -0), is refused. Before,
/// the pair was "L0", which those captures contradicted (MISS 98).
#[test]
fn zero_sign_is_its_own_level() {
    let t = equiv_ok("fixtures/c_equiv_zero_sign/eq.toml", 4);
    let cand = |name: &str| {
        t.split(&format!("candidate c_equiv_zero_sign_{name} good sass\n"))
            .nth(1)
            .and_then(|s| s.split("\n\n").next())
            .unwrap_or_else(|| panic!("{name}: {t}"))
            .to_string()
    };
    let c = cand("cand_add0");
    for s in [
        "L0 bitwise               FAILS at 1024 of 1024 outputs (first: 0)\n        equal bits up to the sign of zero",
        "L1 same function         PROVEN",
        "L0~0 up to the sign of zero PROVEN",
        "out[0..=1023]: L0~0 x1024; |C-R| <= k u sum|terms|, k from 0.00 to 0.00",
        "captured GPU outputs: 1008 of 1024 bitwise equal, max |C - R| 0.000e0",
        "=> strongest level: L0~0\n",
    ] {
        assert!(c.contains(s), "{s}: {c}");
    }
    assert!(!c.contains("contradict"), "{c}");
    let c = cand("cand_ba");
    assert!(c.contains("L0 bitwise               PROVEN"), "{c}");
    assert!(c.contains("=> strongest level: L0\n"), "{c}");
    assert!(!c.contains("L0~0"), "{c}");
    for (name, byte) in [
        ("cand_add0_t1", "36 (sim 0x08, gpu 0x09)"),
        ("cand_add0_z", "279 (sim 0x00, gpu 0x80)"),
    ] {
        let c = cand(name);
        assert!(
            c.contains(&format!("1 of 4096 bytes differ, first at byte {byte}")),
            "{c}"
        );
        assert!(
            c.contains("captured GPU outputs contradict L0~0: the candidate's simulated output differs from its capture"),
            "{c}"
        );
    }
    let dir = std::env::temp_dir().join(format!(
        "ptxv-test-equiv-fixtures_c_equiv_zero_sign_eq.toml-{}",
        std::process::id()
    ));
    let report: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("equiv.json")).unwrap()).unwrap();
    let r = &report[0];
    assert_eq!(r["strongest"], "L0~0", "{r}");
    assert_eq!(r["proven"], true, "{r}");
    assert_eq!(r["report"]["l0_zero_sign"], 1024, "{r}");
    assert_eq!(r["report"]["per_output"][5]["level"], "L0~0", "{r}");
    let cap = &r["captured"];
    assert_eq!(cap["zero_sign_pairs"], 16, "{cap}");
    assert_eq!(cap["differing_beyond_zero_sign"], 0, "{cap}");
    assert_eq!(cap["first_differing"], 5, "{cap}");
}

/// MISS 134 (fixtures/c_equiv_atomic_ftz): an fp32 atomic add's flush flag is part of the
/// canonical form. red.add.f32 (FTZ) against a bulk f32 add (no flush) of the same value into
/// +0 (i), and two adds of both modes into +0 in opposite orders (mixA), are not L0: on inputs
/// with subnormals (set s) GB300's outputs differ in 64 and 79 of 1024 words, within the bound.
/// mixA's chains depend on their order, so their outputs are held to the bound, not to one
/// order's capture. Adds of one mode in both orders (cftz, cblk) stay L0. Before, i and mixA
/// were L0 (refused on set s, as the captures contradict it).
#[test]
fn atomic_flush_modes_are_not_bitwise() {
    for (pair, level, equal) in [
        ("i_n", "L2", 1024),
        ("i_s", "L2", 960),
        ("mixA_n", "L2", 1024),
        ("mixA_s", "L2", 945),
        ("cftz_s", "L0", 1024),
        ("cblk_s", "L0", 1024),
    ] {
        let t = equiv_ok(&format!("fixtures/c_equiv_atomic_ftz/eq_{pair}.toml"), 1);
        for s in [
            format!("=> strongest level: {level}\n"),
            format!("captured GPU outputs: {equal} of 1024 bitwise equal"),
        ] {
            assert!(t.contains(&s), "{pair}: {s}: {t}");
        }
        assert!(!t.contains("contradict"), "{pair}: {t}");
        let mixed = "mix flushing (red/atom.add.f32) and non-flushing (bulk reduction) adds, \
                     whose order changes the bits";
        let exempt = "simulated output not held to the captured GPU output at 1024 elements \
                      that depend on the order of float atomic adds";
        if pair.starts_with("mixA") {
            assert!(t.contains(mixed), "{pair}: {t}");
            assert_eq!(t.matches(exempt).count(), 2, "{pair}: {t}");
        } else {
            assert!(!t.contains(exempt), "{pair}: {t}");
        }
        if pair.starts_with('i') {
            assert!(
                t.contains("L0 bitwise               FAILS at 1024 of 1024 outputs (first: 0)\n        canonical rounding DAGs differ"),
                "{pair}: {t}"
            );
        }
    }
}

/// MISS 135 (fixtures/c_equiv_zero_sign_nan): L0~0 needs the peeled x + (+0) to keep x's
/// NaNs. A copy of an input against the copy + 0 (FADD with RZ) is not L0~0: the input can
/// hold a NaN payload, which the + 0 turns into 0x7fffffff. GB300's captures differ at the 16
/// zero signs and, on set p, also at its 7 non-canonical NaNs; the pair is L2 with bound 0,
/// which they do not contradict. Before, it was L0~0 (refused on set p through the captures).
#[test]
fn zero_sign_needs_canonical_nans() {
    for (set, equal) in [("p", 1001), ("q", 1008)] {
        let t = equiv_ok(&format!("fixtures/c_equiv_zero_sign_nan/eq_{set}.toml"), 1);
        for s in [
            "L0 bitwise               FAILS at 1024 of 1024 outputs (first: 0)\n        canonical rounding DAGs differ".to_string(),
            "bound: |C - R| <= 0.00u * sum|terms| for every output and every input (PROVEN)".to_string(),
            format!("captured GPU outputs: {equal} of 1024 bitwise equal, max |C - R| 0.000e0"),
            "=> strongest level: L2\n".to_string(),
        ] {
            assert!(t.contains(&s), "{set}: {s}: {t}");
        }
        assert!(!t.contains("L0~0"), "{set}: {t}");
        assert!(!t.contains("contradict"), "{set}: {t}");
    }
}
