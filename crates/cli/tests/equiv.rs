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
/// torch._scaled_mm is L0 up to the sign of zero (cuBLAS adds +0 in its alpha / beta epilogue;
/// its BAR.SYNC waits on the bulk read scoreboard in the control word).
#[test]
fn mxf8_cublas_is_bitwise() {
    if vendor_missing("fixtures/c_equiv_mxf8_cublas/equiv.toml") {
        return;
    }
    let t = equiv_ok("fixtures/c_equiv_mxf8_cublas/equiv.toml", 1);
    assert!(t.contains("L0 bitwise               PROVEN"), "{t}");
    assert!(
        t.contains("L0 at 16384 outputs is up to the sign of zero"),
        "{t}"
    );
    assert!(t.contains("0 over the bound"), "{t}");
}

/// NVFP4 GEMM (docs/equiv-mma.md §5): CuTeDSL (one CTA) against cuBLASLt's torch._scaled_mm,
/// a cta_group::2 kernel (pair block-scaled MMA, UTCCP.2CTA, LDTM.16dp128bit, LOP3.PAND), is
/// L0 up to the sign of zero; each kernel's own bound is 9.88u x sum |a b sa sb| (ue4m3 scales
/// nonzero), no random trial above it.
#[test]
fn nvf4_cublas_is_bitwise() {
    if vendor_missing("fixtures/c_equiv_nvf4_cublas/equiv.toml") {
        return;
    }
    let t = equiv_ok("fixtures/c_equiv_nvf4_cublas/equiv.toml", 1);
    assert!(t.contains("L0 bitwise               PROVEN"), "{t}");
    assert!(
        t.contains("L0 at 16384 outputs is up to the sign of zero"),
        "{t}"
    );
    assert!(t.contains("reference 9.88u, candidate 9.88u"), "{t}");
    assert!(t.contains("0 over the bound"), "{t}");
    assert!(t.contains("0 outputs differed bitwise"), "{t}");
}

/// NVFP4 against a dequantize-then-bf16 GEMM (Triton: cvt of each code and ue4m3 scale, f32
/// product, bf16 rounding, bf16 tl.dot): the dequantization is exact for every input
/// (N-EXACT), so it is the same function (L1) with only the two kernels' MMA contracts in the
/// bound, 169.38u x sum |a b sa sb|, no random trial above it.
#[test]
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
