/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A PTX case with an `executed_cubin`: ptxv compiles the PTX (with the ptxas of the case's
//! `ptxas_toolkit` when the registry has it) and compares the rebuilt kernel code with the
//! executed cubin. Equal code: `ptx_rebuild.status = "match"`, an executed-image verdict.
//! Different code: `"mismatch"`, `executed_image: false` and an `image_note` on both fronts.
//! Needs ptxas (CUDA_HOME), no GPU.

use std::path::Path;
use std::path::PathBuf;
use std::process::Command;

const PTX: &str = r#".version 8.8
.target sm_103a
.address_size 64

.visible .entry store_tid(.param .u64 out)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd2, %rd2, %rd3;
    add.u32 %r2, %r1, 7;
    st.global.u32 [%rd2], %r2;
    ret;
}
"#;

fn ptxas() -> String {
    let cuda = std::env::var("CUDA_HOME").unwrap_or_else(|_| "/usr/local/cuda-13.0".into());
    format!("{cuda}/bin/ptxas")
}

/// A case directory: the PTX, an executed cubin built by ptxas with `opt` (e.g. "-O3"), the
/// expected output (tid + 7).
fn case(tag: &str, opt: &str, toolkit: Option<&str>) -> PathBuf {
    case_with(tag, opt, toolkit, "", None)
}

/// `case`, plus manifest lines `extra`, the executed cubin built by ptxas run under the name
/// `tool` (its toolkit note records that name, as Triton's `ptxas-blackwell`).
fn case_with(
    tag: &str,
    opt: &str,
    toolkit: Option<&str>,
    extra: &str,
    tool: Option<&str>,
) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "ptxv-test-ptx-rebuild-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(d.join("expected")).unwrap();
    std::fs::write(d.join("kernel.ptx"), PTX).unwrap();
    let bin = match tool {
        Some(t) => {
            let link = d.join(t);
            std::os::unix::fs::symlink(ptxas(), &link).unwrap();
            link.to_str().unwrap().to_string()
        }
        None => ptxas(),
    };
    let st = Command::new(bin)
        .args(["-arch=sm_103a", opt, "kernel.ptx", "-o", "jit.cubin"])
        .current_dir(&d)
        .status()
        .unwrap();
    assert!(st.success());
    let exp: Vec<u8> = (0u32..32).flat_map(|t| (t + 7).to_le_bytes()).collect();
    std::fs::write(d.join("expected/out.bin"), exp).unwrap();
    let tk = toolkit
        .map(|t| format!("ptxas_toolkit = \"{t}\"\n"))
        .unwrap_or_default();
    std::fs::write(
        d.join("case.toml"),
        format!(
            "name = \"ptx_rebuild_{tag}\"\ndescription = \"\"\"test\"\"\"\nptx = \"kernel.ptx\"\n\
             executed_cubin = \"jit.cubin\"\n{tk}{extra}entry = \"store_tid\"\narch = \"sm_103a\"\n\
             grid = [1, 1, 1]\nblock = [32, 1, 1]\noutputs = [\"out\"]\n\n\
             [[buffers]]\nname = \"out\"\ndtype = \"u32\"\nshape = [32]\ninit = \"zeros\"\n\
             expect = \"expected/out.bin\"\n\n[[params]]\nkind = \"ptr\"\nbuffer = \"out\"\n\
             byte_offset = 0\n\n[[variants]]\nid = \"good\"\nsummary = \"as captured\"\n\
             expect = [\"clean\"]\n"
        ),
    )
    .unwrap();
    d
}

/// `ptxv check CASE --front FRONT --json`: the first row.
fn check(case: &Path, front: &str) -> serde_json::Value {
    let out = Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["check", case.to_str().unwrap(), "--front", front, "--json"])
        .args(["--out", case.join("out").to_str().unwrap()])
        .env("PTXV_QUIET", "1")
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    let doc: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap_or_else(|e| {
        panic!(
            "{e}: {}\n{}",
            String::from_utf8_lossy(&out.stdout),
            String::from_utf8_lossy(&out.stderr)
        )
    });
    doc["rows"][0].clone()
}

#[test]
fn rebuilt_code_equal_to_the_executed_cubin_is_a_match() {
    let d = case("match", "-O3", None);
    for front in ["ptx", "sass"] {
        let row = check(&d, front);
        assert_eq!(row["ptx_rebuild"]["status"], "match", "{front}: {row}");
        assert_eq!(row["executed_image"], true, "{front}: {row}");
        assert!(row["image_note"].is_null(), "{front}: {row}");
    }
}

#[test]
fn rebuilt_code_different_from_the_executed_cubin_is_a_mismatch() {
    // An -O0 build is different machine code from ptxv's default (-O3) rebuild.
    let d = case("mismatch", "-O0", None);
    for front in ["ptx", "sass"] {
        let row = check(&d, front);
        assert_eq!(row["ptx_rebuild"]["status"], "mismatch", "{front}: {row}");
        assert_eq!(
            row["ptx_rebuild"]["differing"][0], "store_tid",
            "{front}: {row}"
        );
        assert_eq!(row["executed_image"], false, "{front}: {row}");
        let note = row["image_note"].as_str().unwrap_or_default();
        assert!(note.starts_with("ptx_rebuild: mismatch"), "{front}: {row}");
    }
}

/// Two kernels, each executed from its own module (`[[launches]] executed_cubin`); the second
/// one is renamed in the merged PTX (`executed_entry`), as ptx_front.py derives two modules
/// defining the same name.
fn two_module_case(tag: &str, second_opt: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!(
        "ptxv-test-ptx-rebuild-{tag}-{}",
        std::process::id()
    ));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(d.join("expected")).unwrap();
    let second = PTX
        .replace("store_tid", "store_tid__m1")
        .replace("7;", "9;");
    let merged = format!("{PTX}{}", &second[second.find(".visible").unwrap()..]);
    std::fs::write(d.join("kernel.ptx"), &merged).unwrap();
    std::fs::write(d.join("a.ptx"), PTX).unwrap();
    // The second module as compiled: the kernel under its own name, `store_tid`.
    std::fs::write(d.join("b.ptx"), PTX.replace("7;", "9;")).unwrap();
    for (src, out, opt) in [
        ("a.ptx", "a.cubin", "-O3"),
        ("b.ptx", "b.cubin", second_opt),
    ] {
        let st = Command::new(ptxas())
            .args(["-arch=sm_103a", opt, src, "-o", out])
            .current_dir(&d)
            .status()
            .unwrap();
        assert!(st.success());
    }
    let exp: Vec<u8> = (0u32..32).flat_map(|t| (t + 9).to_le_bytes()).collect();
    std::fs::write(d.join("expected/out.bin"), exp).unwrap();
    let launch = "grid = [1, 1, 1]\nblock = [32, 1, 1]\n";
    let param = "kind = \"ptr\"\nbuffer = \"out\"\nbyte_offset = 0\n";
    std::fs::write(
        d.join("case.toml"),
        format!(
            "name = \"ptx_rebuild_{tag}\"\ndescription = \"\"\"test\"\"\"\nptx = \"kernel.ptx\"\n\
             executed_cubin = \"a.cubin\"\nentry = \"store_tid\"\narch = \"sm_103a\"\n{launch}\
             outputs = [\"out\"]\n\n[[buffers]]\nname = \"out\"\ndtype = \"u32\"\nshape = [32]\n\
             init = \"zeros\"\nexpect = \"expected/out.bin\"\n\n[[params]]\n{param}\n\
             [[launches]]\nentry = \"store_tid__m1\"\n{launch}executed_cubin = \"b.cubin\"\n\
             executed_entry = \"store_tid\"\n\n[[launches.params]]\n{param}\n\
             [[variants]]\nid = \"good\"\nsummary = \"as captured\"\nexpect = [\"clean\"]\n"
        ),
    )
    .unwrap();
    d
}

#[test]
fn launches_from_two_modules_compare_with_their_own_executed_cubins() {
    let d = two_module_case("two_match", "-O3");
    let row = check(&d, "sass");
    assert_eq!(row["ptx_rebuild"]["status"], "match", "{row}");
    let d = two_module_case("two_mismatch", "-O0");
    let row = check(&d, "sass");
    assert_eq!(row["ptx_rebuild"]["status"], "mismatch", "{row}");
    assert_eq!(
        row["ptx_rebuild"]["differing"],
        serde_json::json!(["store_tid__m1"]),
        "{row}"
    );
}

#[test]
fn a_toolkit_missing_from_the_registry_falls_back_to_the_default_ptxas() {
    // ptxas_toolkit names a toolkit no registry has: the default ptxas builds it, and the
    // rebuild record names the toolkit that built the executed cubin.
    let d = case("fallback", "-O3", Some("99.9"));
    let row = check(&d, "sass");
    assert_eq!(row["ptx_rebuild"]["status"], "match", "{row}");
    assert!(row["ptx_rebuild"]["rebuilt_toolkit"].is_string(), "{row}");
}

#[test]
fn ptxas_args_the_toolchain_used_reproduce_the_executed_code() {
    // The executed cubin was built with an option its toolkit note does not show to ptxv's
    // default rebuild; `ptxas_args` passes it.
    let d = case_with("args", "-O0", None, "ptxas_args = [\"-O0\"]\n", None);
    let row = check(&d, "sass");
    assert_eq!(row["ptx_rebuild"]["status"], "match", "{row}");
    assert_eq!(row["ptx_rebuild"]["ptxas_args"][0], "-O0", "{row}");
    assert_eq!(row["executed_image"], true, "{row}");
}

#[test]
fn a_hand_edited_variant_is_a_what_if_not_a_mismatch() {
    let d = case_with(
        "whatif",
        "-O0",
        None,
        "variant_of = \"ptx_rebuild_orig\"\n",
        None,
    );
    for front in ["ptx", "sass"] {
        let row = check(&d, front);
        assert_eq!(row["ptx_rebuild"]["status"], "what-if", "{front}: {row}");
        assert_eq!(
            row["ptx_rebuild"]["variant_of"], "ptx_rebuild_orig",
            "{front}: {row}"
        );
        assert_eq!(row["executed_image"], false, "{front}: {row}");
        let note = row["image_note"].as_str().unwrap_or_default();
        assert!(note.starts_with("ptx_rebuild: what-if"), "{front}: {row}");
    }
}

#[test]
fn a_triton_ptxas_blackwell_cubin_without_ptxas_args_is_refused() {
    // Triton's bundled ptxas-blackwell adds --regAllocOptLevel=2, which its toolkit note does
    // not record: without ptxas_args the rebuild could be a different binary.
    let d = case_with("triton", "-O3", None, "", Some("ptxas-blackwell"));
    let out = Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["check", d.to_str().unwrap(), "--front", "ptx", "--json"])
        .args(["--out", d.join("out").to_str().unwrap()])
        .env("PTXV_QUIET", "1")
        .env("CUDA_VISIBLE_DEVICES", "")
        .output()
        .unwrap();
    let text = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(
        text.contains("ptxas-blackwell") && text.contains("ptxas_args"),
        "{text}"
    );
    assert!(!text.contains("\"verdict\""), "{text}");
    // With the (verified) empty set it runs.
    let d = case_with(
        "triton_args",
        "-O3",
        None,
        "ptxas_args = []\n",
        Some("ptxas-blackwell"),
    );
    let row = check(&d, "sass");
    assert_eq!(row["ptx_rebuild"]["status"], "match", "{row}");
}
