/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `ptxv check` fails closed on a selection that names nothing to check: `--front ptx` on a
//! binary-only case, an unknown front and an unknown variant are errors (non-zero exit and a
//! message), not an empty passing run.

use std::path::PathBuf;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn check(args: &[&str]) -> (bool, String) {
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .arg("check")
        .args(args)
        .env("CUDA_VISIBLE_DEVICES", "")
        .current_dir(std::env::temp_dir())
        .output()
        .unwrap();
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

#[test]
fn empty_selections_are_errors() {
    let bin = root().join("fixtures/c_cve_add_bf16_eager");
    let ptx = root().join("fixtures/e_elect");
    let (ok, err) = check(&[bin.to_str().unwrap(), "--front", "ptx"]);
    assert!(!ok && err.contains("has no PTX"), "{err}");
    let (ok, err) = check(&[ptx.to_str().unwrap(), "--front", "pxt"]);
    assert!(!ok && err.contains("expected ptx, sass or both"), "{err}");
    let (ok, err) = check(&[ptx.to_str().unwrap(), "--variant", "nope"]);
    assert!(!ok && err.contains("no variant nope"), "{err}");
}

/// A PTX module newer than every installed ptxas is refused with the toolkit it needs, before
/// ptxas runs (it would otherwise fail with only its own "Unsupported .version"). A registry
/// ptxas that does not run here (another architecture's binary; here a script that fails) is
/// skipped and named, not run.
#[test]
fn ptx_newer_than_every_ptxas_is_refused_by_name() {
    let src = root().join("fixtures/e_elect");
    let d = std::env::temp_dir().join(format!("ptxv-isa-too-new-{}", std::process::id()));
    let reg = d.join("toolkits");
    let fake = reg.join("13.9/bin");
    std::fs::create_dir_all(&fake).unwrap();
    std::fs::write(fake.join("ptxas"), "#!/bin/sh\nexit 1\n").unwrap();
    {
        use std::os::unix::fs::PermissionsExt;
        let p = fake.join("ptxas");
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o755)).unwrap();
    }
    let case = d.join("case");
    std::fs::create_dir_all(&case).unwrap();
    std::fs::copy(src.join("case.toml"), case.join("case.toml")).unwrap();
    let ptx = std::fs::read_to_string(src.join("kernel.ptx")).unwrap();
    assert!(ptx.contains(".version 9.0"));
    std::fs::write(
        case.join("kernel.ptx"),
        ptx.replace(".version 9.0", ".version 9.9"),
    )
    .unwrap();
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_ptxv"))
        .args(["check", case.to_str().unwrap(), "--front", "ptx"])
        .env("CUDA_VISIBLE_DEVICES", "")
        .env("PTXV_TOOLKITS", &reg)
        .current_dir(&d)
        .output()
        .unwrap();
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    let _ = std::fs::remove_dir_all(&d);
    assert!(!out.status.success(), "{err}");
    assert!(
        err.contains("PTX ISA 9.9, which needs ptxas 13.9 or later"),
        "{err}"
    );
    assert!(
        err.contains("does not run here or reports another version: [13.9]"),
        "{err}"
    );
}
