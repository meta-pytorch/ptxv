#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the BSD-style license in the repository root.
"""Re-extract the actual strict-memory Rust roots into a new private directory."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
EXPORT_SPEC = importlib.util.spec_from_file_location(
    "generated_lean", PROJECT.parent / "generated_lean.py"
)
if EXPORT_SPEC is None or EXPORT_SPEC.loader is None:
    raise RuntimeError("Cannot load generated Lean export policy")
LEAN_EXPORT = importlib.util.module_from_spec(EXPORT_SPEC)
EXPORT_SPEC.loader.exec_module(LEAN_EXPORT)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def run(command, cwd, env, log):
    start = time.monotonic()
    with log.open("x") as stream:
        result = subprocess.run(
            command, cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT
        )
    receipt = {
        "argv": command,
        "exit_code": result.returncode,
        "seconds": time.monotonic() - start,
        "log": str(log),
        "log_sha256": digest(log),
    }
    if result.returncode:
        raise ValueError(
            f"Extraction failed; retained {log}\n"
            + "\n".join(log.read_text().splitlines()[-30:])
        )
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptxv-root", type=Path, default=PROJECT.parent.parent)
    parser.add_argument(
        "--tools",
        type=Path,
        required=True,
        help="Local charon/aeneas/rust_bin/cargo_home paths; no downloads",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="New directory outside the source tree"
    )
    args = parser.parse_args()
    root, out = args.ptxv_root.resolve(), args.out.resolve()
    if out.exists() or out == root or root in out.parents:
        raise ValueError("--out must be a new directory outside the PTXV source tree")
    tools = json.loads(args.tools.read_text())
    for name in ("charon", "aeneas", "rust_bin", "cargo_home"):
        tools[name] = str(Path(tools[name]).resolve())
    specification = json.loads((PROJECT / "extraction-spec.json").read_text())
    inputs = json.loads((PROJECT / "extraction-inputs.json").read_text())
    if specification["generated_export"] != LEAN_EXPORT.POLICY:
        raise ValueError("Extraction specification uses a different Lean export policy")
    for name in ("charon", "aeneas"):
        if digest(Path(tools[name])) != specification["tools"][name]["sha256"]:
            raise ValueError(f"{name} binary differs from the pinned extraction tool")
    driver = Path(tools["charon"]).with_name("charon-driver")
    if digest(driver) != specification["tools"]["charon_driver"]["sha256"]:
        raise ValueError("charon-driver differs from the pinned extraction tool")
    for name in ("cargo", "rustc"):
        binary = Path(tools["rust_bin"]) / name
        if digest(binary) != specification["rust_toolchain"][name]["sha256"]:
            raise ValueError(f"{name} differs from the pinned Rust toolchain")
    cargo_home = Path(tools["cargo_home"]).resolve()
    if not (cargo_home / "registry").is_dir():
        raise ValueError("--tools must name a prepopulated offline cargo_home")
    out.mkdir(parents=True)
    workspace = out / "workspace"
    workspace.mkdir()
    before = {}
    for destination, source in inputs.items():
        for relative in (destination, source):
            path = Path(relative)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Nonrelative extraction input: {relative}")
        original, copied = root / source, workspace / destination
        expected = digest(original)
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(original, copied)
        if digest(copied) != expected:
            raise ValueError(f"Source changed while copying {source}")
        before[source] = expected
    write_json(out / "source-before.json", before)
    env = os.environ.copy()
    for name in list(env):
        if name.startswith("PTXV_") or name in {
            "RUSTC",
            "RUSTDOC",
            "RUSTUP_TOOLCHAIN",
            "RUSTC_WRAPPER",
            "RUSTC_WORKSPACE_WRAPPER",
            "RUSTFLAGS",
            "CARGO_ENCODED_RUSTFLAGS",
        }:
            env.pop(name)
    env.update(
        CARGO_NET_OFFLINE="true",
        CARGO_HOME=str(cargo_home),
        CARGO_BUILD_JOBS="4",
        CARGO_TARGET_DIR=str(out / "target"),
        PYTHONDONTWRITEBYTECODE="1",
        CUDA_VISIBLE_DEVICES="",
    )
    env["PATH"] = (
        str(Path(tools["rust_bin"]).resolve()) + os.pathsep + env.get("PATH", "")
    )
    llbc = out / "Program.llbc"
    charon = [tools["charon"], "cargo", *specification["charon_options"]]
    for name in specification["roots"]:
        charon.extend(["--start-from", name])
    for name in specification["opaque"]:
        charon.extend(["--opaque", name])
    for name in specification["include"]:
        charon.extend(["--include", name])
    charon.extend(
        [
            "--dest-file",
            str(llbc),
            "--",
            "--offline",
            "--locked",
            "--lib",
            "-p",
            "ptxv-sim",
        ]
    )
    outcomes = []
    try:
        outcomes.append(run(charon, workspace, env, out / "charon.log"))
        fresh = out / "fresh"
        fresh.mkdir()
        aeneas = [
            tools["aeneas"],
            "-backend",
            "lean",
            "-namespace",
            specification["namespace"],
            "-dest",
            str(fresh),
            "-subdir",
            "MemorySafety/Code",
            *specification["aeneas_options"],
            str(llbc),
        ]
        outcomes.append(run(aeneas, workspace, env, out / "aeneas.log"))
        generated = fresh / "MemorySafety/Code"
        if sorted(path.name for path in generated.iterdir()) != [
            "Funs.lean",
            "Types.lean",
        ]:
            raise ValueError(
                "Extraction produced an external template or unexpected generated file"
            )
        generated_export = LEAN_EXPORT.publish(generated, out / "raw-generated")
        equality = {}
        for name in ("Types.lean", "Funs.lean"):
            equality[name] = (generated / name).read_bytes() == (
                PROJECT / "MemorySafety/Code" / name
            ).read_bytes()
        after = {source: digest(root / source) for source in before}
        write_json(out / "source-after.json", after)
        if before != after:
            raise ValueError("Source inputs changed during extraction")
        for destination, source in inputs.items():
            if digest(workspace / destination) != before[source]:
                raise ValueError(
                    f"Private snapshot changed during extraction: {destination}"
                )
        write_json(
            out / "receipt.json",
            {
                "status": "pass" if all(equality.values()) else "stale-generated-code",
                "generated_byte_equal": equality,
                "generated_export": generated_export,
                "outcomes": outcomes,
                "llbc_sha256": digest(llbc),
                "specification_sha256": digest(PROJECT / "extraction-spec.json"),
                "source_manifest_sha256": digest(out / "source-before.json"),
            },
        )
        if not all(equality.values()):
            raise ValueError(
                "Fresh extraction differs from checked-in Code; review private output before updating proofs"
            )
    except (OSError, ValueError) as error:
        write_json(
            out / "failure.json",
            {"status": "fail", "error": str(error), "outcomes": outcomes},
        )
        raise
    print(
        json.dumps({"status": "pass", "receipt": str(out / "receipt.json")}, indent=2)
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f"memory-safety extraction failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
