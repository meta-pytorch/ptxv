/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `ptxv num` (and so `ptxv hw`'s expected bytes) on a PTX case whose PTX the front-end does
//! not parse, though ptxas compiles it: the SASS front alone gives the run.

use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

#[test]
fn num_runs_on_sass_when_the_ptx_front_refuses() {
    let src = root().join("probes/f64_queue");
    let dir = std::env::temp_dir().join(format!("ptxv-test-num-ptx-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(dir.join("expected")).unwrap();
    for f in ["case.toml", "in.bin", "expected/out.bin"] {
        std::fs::copy(src.join(f), dir.join(f)).unwrap();
    }
    // `prefetchu.L1`: valid PTX (ptxas compiles it to nothing observable), not modeled by the
    // PTX front-end.
    let ptx = std::fs::read_to_string(src.join("kernel.ptx")).unwrap();
    let at = ptx.find("  ld.global.v2.f64").expect("probe PTX layout");
    std::fs::write(
        dir.join("kernel.ptx"),
        format!("{}  prefetchu.L1 [%rd4];\n{}", &ptx[..at], &ptx[at..]),
    )
    .unwrap();
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["num", dir.to_str().unwrap(), "--out"])
        .arg(dir.join("out"))
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    let _ = std::fs::remove_dir_all(&dir);
    assert!(out.status.success(), "{text}");
    assert!(text.contains("PTX front-end refuses it"), "{text}");
    assert!(text.contains("num good sass:"), "{text}");
}
