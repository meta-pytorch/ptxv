#!/usr/bin/env fbpython
# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license in the repository root.
"""Check composition against fresh instruction and memory-effect extractions."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "lean_model/subsystem_interop"
FILES = [
    "lakefile.lean",
    "lean-toolchain",
    "Interop.lean",
    "Audit.lean",
    "CompositionWitness.lean",
    "README.md",
] + [
    f"Interop/{name}.lean"
    for name in [
        "AtomicAdmissionInterop",
        "OwnerSymbolic",
        "OwnerCopy",
        "OwnerMemory",
        "OwnerMetadata",
        "RegisterIndexInterop",
        "MemoryGraphInterop",
        "ControlFlowMemoryInterop",
        "ControlFlowBounds",
        "ControlFlowMemoryTotal",
        "ControlFlowPublication",
        "BarrierMemoryPaths",
        "BarrierReachInterop",
        "ProgramGraphPublication",
        "DestinationRowsInterop",
        "KernelGraphPublication",
        "NativeGraphPublication",
        "CertificateCfg",
        "CertificateDynamicPaths",
        "CertifiedNativeQuery",
    ]
]
LOGGER: logging.Logger = logging.getLogger(__name__)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_checker(name: str):
    path = ROOT / f"probes/{name}/check.py"
    spec = importlib.util.spec_from_file_location(f"{name}_checker", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hashes(memory, instruction, sim_sources=()) -> dict[str, str]:
    result = memory.source_hashes()
    instruction_hashes = (
        instruction.hashes(sim_sources) if sim_sources else instruction.hashes()
    )
    for name, value in instruction_hashes.items():
        if name in result and result[name] != value:
            raise RuntimeError(f"Inconsistent source snapshots for {name}")
        result[name] = value
    result.update(
        {
            str((PROJECT / name).relative_to(ROOT)): digest(PROJECT / name)
            for name in FILES
        }
    )
    result[str(Path(__file__).relative_to(ROOT))] = digest(Path(__file__))
    return result


def copy_package(
    source: Path, destination: Path, names: list[str], core, backend: Path, name: str
) -> None:
    for relative in names:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / relative, target)
    core.configure(destination, backend)
    path = destination / "lake-manifest.json"
    manifest = json.loads(path.read_text())
    manifest["name"] = name
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def attach_dependencies(project: Path, dependencies: dict[str, Path]) -> None:
    path = project / "lake-manifest.json"
    manifest = json.loads(path.read_text())
    lakefile = project / "lakefile.lean"
    configuration = lakefile.read_text()
    for name, directory in dependencies.items():
        relative = os.path.relpath(directory, project)
        pattern = rf'(require {re.escape(name)} from )"[^"\n]+"'
        configuration, count = re.subn(
            pattern,
            lambda match, relative=relative: f'{match[1]}"{relative}"',
            configuration,
        )
        if count != 1:
            raise RuntimeError(f"Expected one dependency declaration for {name}")
        manifest["packages"].append(
            {
                "type": "path",
                "name": name,
                "dir": relative,
                "manifestFile": "lake-manifest.json",
                "configFile": "lakefile.lean",
                "inherited": False,
            }
        )
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    lakefile.write_text(configuration)


def verify_leaves(args, out: Path, core, env) -> dict[str, str]:
    receipts = {}
    for name in ["memory_write", "instruction_effects"]:
        command = [
            sys.executable,
            str(ROOT / f"probes/{name}/check.py"),
            "--tools",
            str(args.tools.resolve()),
            "--out",
            str(out / f"{name}-verification"),
        ]
        if args.regenerate:
            command.append("--regenerate")
        core.run(command, ROOT, out / f"{name}-verification.log", env)
        receipts[name] = digest(out / f"{name}-verification/receipt.json")
    return receipts


def check_source_changes(
    before: dict[str, str], after: dict[str, str], regenerate: bool
) -> None:
    generated = {
        f"lean_model/{package}/{module}/Code/{name}.lean"
        for package, module in [
            ("memory_write", "MemoryWrite"),
            ("instruction_effects", "Instruction"),
            ("instruction_effects", "Instruction/SelectionProgram"),
        ]
        for name in ["Funs", "Types"]
    }
    changed = [
        name
        for name, value in before.items()
        if after.get(name) != value and not (regenerate and name in generated)
    ]
    if changed:
        raise RuntimeError(f"Sources changed during verification: {changed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    core = load_checker("memory_write")
    instruction = load_checker("instruction_effects")
    tools = json.loads(args.tools.read_text())
    env = core.verification_environment(out, tools)
    env["CARGO_BUILD_JOBS"] = "4"
    env["CARGO_NET_OFFLINE"] = "true"
    sim_inventory_out = out / "sim-source-inventory"
    sim_inventory_out.mkdir()
    selection_program = instruction.load_selection_program()
    _, sim_sources = selection_program.source_archive(sim_inventory_out)
    before = hashes(core, instruction, sim_sources)
    tool_hashes = {
        name: digest(Path(tools[name])) for name in ["charon", "aeneas", "lake"]
    }
    backend_hash = digest(Path(tools["backend"]) / "lake-manifest.json")
    receipts = verify_leaves(args, out, core, env)
    after = hashes(core, instruction, sim_sources)
    check_source_changes(before, after, args.regenerate)
    check_composition(out, core, instruction, tools, env, receipts)
    if hashes(core, instruction, sim_sources) != after:
        raise RuntimeError("Sources changed during composition validation")
    if tool_hashes != {
        name: digest(Path(tools[name])) for name in tool_hashes
    } or backend_hash != digest(Path(tools["backend"]) / "lake-manifest.json"):
        raise RuntimeError("Verification tools changed during validation")
    receipt = {
        "source_sha256": after,
        "tool_sha256": tool_hashes,
        "backend_manifest_sha256": backend_hash,
        "leaf_receipt_sha256": receipts,
        "reused_fresh_leaf_packages": True,
        "leaf_package_paths": {
            name: f"../{name}-verification/lean" for name in receipts
        },
        "fresh_extractions": True,
        "sim_source_inventory": sorted(map(str, sim_sources)),
        "sim_constructor_extraction": True,
        "lean_build": True,
        "axiom_audit": True,
        "wrong_fold_source_witness": True,
    }
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    LOGGER.info("Verified subsystem composition; receipt: %s", out / "receipt.json")


def verified_leaf_package(
    out: Path,
    leaf: str,
    source: Path,
    names: list[str],
    receipt_sha256: str,
    tools: dict[str, str],
) -> tuple[Path, dict[str, str]]:
    directory = out / f"{leaf}-verification"
    receipt = directory / "receipt.json"
    if digest(receipt) != receipt_sha256:
        raise RuntimeError(f"Verified leaf receipt changed: {leaf}")
    verified = json.loads(receipt.read_text())
    for flag in ["fresh_extraction", "lean_build", "axiom_audit"]:
        if verified.get(flag) is not True:
            raise RuntimeError(f"Missing {flag} evidence for {leaf}")
    expected_tools = {
        name: digest(Path(tools[name])) for name in ["charon", "aeneas", "lake"]
    }
    if verified["tool_sha256"] != expected_tools or verified[
        "backend_manifest_sha256"
    ] != digest(Path(tools["backend"]) / "lake-manifest.json"):
        raise RuntimeError(f"Verified leaf tools changed: {leaf}")
    project = directory / "lean"
    pins = {}
    for name in names:
        canonical = source / name
        expected = verified["source_sha256"][str(canonical.relative_to(ROOT))]
        if digest(canonical) != expected or digest(project / name) != expected:
            raise RuntimeError(f"Verified leaf source changed: {leaf}/{name}")
        pins[str((project / name).relative_to(out))] = expected
    return project, pins


def check_composition(
    out: Path, core, instruction, tools, env, leaf_receipts: dict[str, str]
) -> None:
    backend = Path(tools["backend"])
    memory, memory_pins = verified_leaf_package(
        out,
        "memory_write",
        ROOT / "lean_model/memory_write",
        core.LEAN_FILES + core.GENERATED,
        leaf_receipts["memory_write"],
        tools,
    )
    ir, ir_pins = verified_leaf_package(
        out,
        "instruction_effects",
        instruction.PROJECT,
        instruction.LEAN,
        leaf_receipts["instruction_effects"],
        tools,
    )
    project = out / "subsystem_interop"
    copy_package(PROJECT, project, FILES, core, backend, "ptxv_subsystem_interop")
    attach_dependencies(
        project, {"ptxv_memory_write": memory, "ptxv_instruction_effects": ir}
    )
    for name in FILES:
        if name.endswith(".lean") and re.search(
            r"\b(sorry|admit|axiom|native_decide|bv_decide)\b",
            (project / name).read_text(),
        ):
            raise RuntimeError(f"Forbidden proof shortcut in {name}")
    core.run([tools["lake"], "build", "Interop"], project, out / "lean-build.log", env)
    core.run(
        [tools["lake"], "env", "lean", "-j4", "Audit.lean"],
        project,
        out / "axiom-audit.log",
        env,
    )

    core.run(
        [tools["lake"], "env", "lean", "-j4", "CompositionWitness.lean"],
        project,
        out / "composition-witness.log",
        env,
    )

    for name, expected in (memory_pins | ir_pins).items():
        if digest(out / name) != expected:
            raise RuntimeError(f"Verified leaf changed during composition: {name}")
    for name, expected in leaf_receipts.items():
        if digest(out / f"{name}-verification/receipt.json") != expected:
            raise RuntimeError(
                f"Verified leaf receipt changed during composition: {name}"
            )


if __name__ == "__main__":
    main()
