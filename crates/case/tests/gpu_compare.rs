/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `Case::expect_compare_tol` / `expect_compare_perm`: each comparison's result next to the line
//! that states it (the lines are `expect_report_tol` / `expect_report_perm`'s), and the output
//! buffers without captured bytes.

use ptxv_case::Case;
use ptxv_case::GpuMatch;
use ptxv_case::Round16;

/// A case with outputs a (u32 x 4, captured 1 2 3 4), b (bf16 x 2, captured 1.0 2.0) and c
/// (u32 x 2, not captured).
fn case(tag: &str) -> (Case, std::path::PathBuf) {
    let dir = std::env::temp_dir().join(format!(
        "ptxv-case-gpu-compare-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(
        dir.join("case.toml"),
        r#"name = "gpu_compare"
description = "test"
ptx = "k.ptx"
entry = "k"
grid = [1, 1, 1]
block = [1, 1, 1]
outputs = ["a", "b", "c"]
params = []

[[buffers]]
name = "a"
dtype = "u32"
shape = [4]
init = "zeros"
expect = "a.bin"

[[buffers]]
name = "b"
dtype = "bf16"
shape = [2]
init = "zeros"
expect = "b.bin"

[[buffers]]
name = "c"
dtype = "u32"
shape = [2]
init = "zeros"

[[variants]]
id = "good"
expect = ["clean"]
"#,
    )
    .unwrap();
    std::fs::write(dir.join("a.bin"), words(&[1, 2, 3, 4])).unwrap();
    std::fs::write(dir.join("b.bin"), floats(&[1.0, 2.0])).unwrap();
    (Case::load(dir.clone()).unwrap(), dir)
}

fn words(w: &[u32]) -> Vec<u8> {
    w.iter().flat_map(|x| x.to_le_bytes()).collect()
}

/// bf16 encodings of `f` (exact: 1.0, 2.0 and the neighbour above 2.0).
fn floats(f: &[f32]) -> Vec<u8> {
    f.iter()
        .flat_map(|x| ((x.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

/// The simulated output: a, b, c concatenated.
fn output(a: &[u32], b: &[f32]) -> Vec<u8> {
    [words(a), floats(b), words(&[7, 7])].concat()
}

fn results(c: &[ptxv_case::GpuCompare]) -> Vec<(&str, GpuMatch)> {
    c.iter().map(|c| (c.part.as_str(), c.result)).collect()
}

#[test]
fn each_comparison_has_its_result() {
    let (case, dir) = case("tol");
    let one_ulp_up = f32::from_bits(2.0f32.to_bits() + 0x1_0000);
    // With the tolerance: b's bytes (16 to 20 of the output) hold calibrated bf16 lanes.
    let marks: Vec<Option<Round16>> = (0..28)
        .map(|i| (16..20).contains(&i).then_some(Round16::Bf16))
        .collect();
    for (a, b, ulp_ok, want) in [
        (
            [1, 2, 3, 4],
            [1.0, 2.0],
            false,
            [GpuMatch::Bitwise, GpuMatch::Bitwise],
        ),
        (
            [1, 2, 3, 5],
            [1.0, 2.0],
            false,
            [GpuMatch::Mismatch, GpuMatch::Bitwise],
        ),
        // One ulp off: within one ulp only with N-MULTIMEM-ROUND16's tolerance (calibrated
        // bf16 lanes).
        (
            [1, 2, 3, 4],
            [1.0, one_ulp_up],
            true,
            [GpuMatch::Bitwise, GpuMatch::WithinUlp],
        ),
        (
            [1, 2, 3, 4],
            [1.0, one_ulp_up],
            false,
            [GpuMatch::Bitwise, GpuMatch::Mismatch],
        ),
        (
            [1, 2, 3, 5],
            [1.0, one_ulp_up],
            true,
            [GpuMatch::Mismatch, GpuMatch::WithinUlp],
        ),
    ] {
        let out = output(&a, &b);
        let tol = ulp_ok.then_some(marks.as_slice());
        let c = case.expect_compare_tol(&out, tol).unwrap();
        assert_eq!(
            results(&c),
            [("a", want[0]), ("b", want[1]), ("c", GpuMatch::NoCapture)],
            "{a:?} {b:?} {ulp_ok}"
        );
        // The lines are the report's, one per compared buffer, none for c.
        let lines: Vec<String> = c.iter().filter_map(|c| c.line.clone()).collect();
        assert_eq!(lines, case.expect_report_tol(&out, tol).unwrap());
        assert!(c[2].line.is_none());
        for (c, w) in c.iter().zip(want) {
            let l = c.line.as_deref().unwrap();
            match w {
                GpuMatch::Bitwise => assert!(l.contains(": bitwise equal to the captured"), "{l}"),
                GpuMatch::WithinUlp => {
                    assert!(l.contains(": within one ulp of the captured"), "{l}")
                }
                _ => assert!(l.contains(": MISMATCH vs the captured"), "{l}"),
            }
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// MISS 110: NaN against a NaN with another payload is a mismatch in `gpu_compare`, even in
/// bytes marked as calibrated bf16 lanes; the same NaN bits are bitwise equal.
#[test]
fn nan_payloads_are_a_mismatch() {
    let (case, dir) = case("nan");
    // b captured as (NaN 0x7fc0, 2.0).
    std::fs::write(dir.join("b.bin"), [0xc0, 0x7f, 0x00, 0x40]).unwrap();
    let marks: Vec<Option<Round16>> = (0..28)
        .map(|i| (16..20).contains(&i).then_some(Round16::Bf16))
        .collect();
    for (b0, want) in [
        (0x7fc1u16, GpuMatch::Mismatch),
        (0x7fc0, GpuMatch::Bitwise),
        (0xffc0, GpuMatch::Mismatch),
    ] {
        let mut out = output(&[1, 2, 3, 4], &[1.0, 2.0]);
        out[16..18].copy_from_slice(&b0.to_le_bytes());
        let c = case.expect_compare_tol(&out, Some(&marks)).unwrap();
        assert_eq!(c[1].result, want, "{b0:#x}: {:?}", c[1].line);
    }
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn permuted_rows_have_their_result() {
    let (case, dir) = case("perm");
    // Counter word 0: two tickets, rows a[0] and a[1] (buffer 0, bytes 0-4 and 4-8). Counter
    // word 1: a row in c (buffer 2), which has no captured bytes.
    let rows = vec![
        vec![vec![(0usize, 0u64, 4u64)], vec![(0, 4, 8)]],
        vec![vec![(2, 0, 4)]],
    ];
    for (a, want_word0, want_a) in [
        // The rows swapped: equal as a multiset; the rest of a bitwise.
        ([2, 1, 3, 4], GpuMatch::PermutedEqual, GpuMatch::Bitwise),
        ([2, 9, 3, 4], GpuMatch::Mismatch, GpuMatch::Bitwise),
        ([2, 1, 3, 9], GpuMatch::PermutedEqual, GpuMatch::Mismatch),
    ] {
        let out = output(&a, &[1.0, 2.0]);
        let c = case.expect_compare_perm(&out, &rows).unwrap();
        assert_eq!(
            results(&c),
            [
                ("c", GpuMatch::NoCapture),
                ("counter word 0", want_word0),
                ("counter word 1", GpuMatch::NotCompared),
                ("a", want_a),
                ("b", GpuMatch::Bitwise),
            ],
            "{a:?}"
        );
        let lines: Vec<String> = c.iter().filter_map(|c| c.line.clone()).collect();
        assert_eq!(lines, case.expect_report_perm(&out, &rows).unwrap());
    }
    let _ = std::fs::remove_dir_all(&dir);
}
