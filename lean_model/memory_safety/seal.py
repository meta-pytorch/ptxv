#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the BSD-style license in the repository root.
"""Seal a reviewed proof project after successful genuine extraction."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--project", type=Path, required=True)
parser.add_argument("--ptxv-root", type=Path, required=True)
parser.add_argument("--extraction", type=Path, required=True)
args = parser.parse_args()
project, root, extraction = (
    args.project.resolve(),
    args.ptxv_root.resolve(),
    args.extraction.resolve(),
)

spec = importlib.util.spec_from_file_location("ptxv_formal_check", project / "check.py")
checker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checker)
checker.check_build_mode(root)
checker.load_endpoint_types()
checker.load_backend_policy()
checker.check_specification_inputs()
if (
    json.loads((project / "extraction-spec.json").read_text())["generated_export"]
    != checker.LEAN_EXPORT.POLICY
):
    raise ValueError("Extraction specification uses a different Lean export policy")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


receipt = json.loads((extraction / "receipt.json").read_text())
if receipt["status"] != "pass" or receipt["generated_byte_equal"] != {
    "Funs.lean": True,
    "Types.lean": True,
}:
    raise ValueError("Only a successful exact genuine extraction can be sealed")
if len(receipt["outcomes"]) != 2 or any(
    stage["exit_code"] for stage in receipt["outcomes"]
):
    raise ValueError("Both extraction commands must have succeeded")
inputs = json.loads((project / "extraction-inputs.json").read_text())
before = json.loads((extraction / "source-before.json").read_text())
after = json.loads((extraction / "source-after.json").read_text())
if before != after or set(before) != set(inputs.values()):
    raise ValueError("Extraction must bind every input, unchanged before and after")
for path, expected in before.items():
    if digest(root / path) != expected:
        raise ValueError("Changed since genuine extraction: " + path)
if digest(project / "extraction-spec.json") != receipt["specification_sha256"]:
    raise ValueError("Extraction specification changed")
if digest(extraction / "source-before.json") != receipt["source_manifest_sha256"]:
    raise ValueError("Extraction input record changed")
if digest(extraction / "Program.llbc") != receipt["llbc_sha256"]:
    raise ValueError("Extracted LLBC changed")
generated = ["MemorySafety/Code/Types.lean", "MemorySafety/Code/Funs.lean"]
for path in generated:
    if (project / path).read_bytes() != (extraction / "fresh" / path).read_bytes():
        raise ValueError("Generated code is not the exact genuine output: " + path)
if sorted(path.name for path in (extraction / "fresh/MemorySafety/Code").iterdir()) != [
    "Funs.lean",
    "Types.lean",
]:
    raise ValueError("Unexpected generated output")
checker.LEAN_EXPORT.verify(
    project / "MemorySafety/Code",
    receipt["generated_export"],
    extraction / "raw-generated",
)
qualified = {
    "version": 1,
    "status": "pass",
    "generated_byte_equal": receipt["generated_byte_equal"],
    "generated_hand_edits": False,
    "generated_export": receipt["generated_export"],
    "generated_files": {path: digest(project / path) for path in generated},
    "receipt_sha256": digest(extraction / "receipt.json"),
    "source_manifest_sha256": receipt["source_manifest_sha256"],
    "specification_sha256": receipt["specification_sha256"],
    "llbc_sha256": receipt["llbc_sha256"],
    "source_inputs_unchanged": True,
    "external_templates": [],
}
write(project / "qualified-extraction.json", qualified)
namespaces = (
    "MemoryWrite",
    "MemorySafety",
    "MemorySafetyIR",
    "MemorySafetyExecution",
    "ValueFlow",
    "ValueFlowDependencies",
    "Frontend",
    "FrontendProgram",
)
proofs = sorted(
    str(path.relative_to(project))
    for namespace in namespaces
    for path in (project / namespace).rglob("*.lean")
    if str(path.relative_to(project)) not in generated
)
project_paths = [
    "endpoint-types.json",
    "specification-inputs.json",
    "backend-inputs.json",
    "seal.py",
    "test_check.py",
    "test_kernel.py",
    ".gitignore",
    "README.md",
    "EXTRACTION.md",
    "check.py",
    "extract.py",
    "extraction-inputs.json",
    "extraction-spec.json",
    "qualified-extraction.json",
    "lakefile.lean",
    "lean-toolchain",
    *generated,
]
project_paths.extend(
    str(path.relative_to(project))
    for path in (project / "backend").rglob("*")
    if path.is_file()
)
project_paths.extend(
    ["MemorySafety/Primitives.lean", "MemoryWrite/TotalPrimitives.lean"]
)
cargo_paths = {
    path
    for path in before
    if Path(path).name in {"Cargo.toml", "Cargo.lock", "selection-Cargo.lock"}
} | {"Cargo.lock"}
native_paths = {
    "crates/cli/src/main.rs",
    "crates/cli/tests/strict_memory.rs",
    "crates/cli/tests/strict_memory_composition.rs",
    "crates/cli/tests/strict_memory_source_syntax.rs",
    "crates/sim/tests/strict_memory_source_syntax.rs",
    "crates/sim/tests/strict_memory_source_text.rs",
    "crates/sim/tests/strict_memory_predicate_stems.rs",
    "crates/sim/tests/strict_memory_counted_stems.rs",
    "crates/sim/src/machine.rs",
    "crates/sim/src/machine_memory_safety.rs",
    "crates/sim/src/strict_memory_source.rs",
    "crates/sim/src/strict_memory_frontend_parse.rs",
    "crates/sim/src/strict_memory_frontend_launch.rs",
    "crates/memory-write/src/execution_core.rs",
    "crates/memory-write/src/checked_execution.rs",
    "crates/memory-write/src/input_independent_execution.rs",
}
manifest = {
    "version": 2,
    "description": "Genuine extraction of the actual raw-source admission and shared Machine path. Every explicit extraction input, conservative Rust closure, native acceptance path, generated file and authored proof is sealed.",
    "extraction_inputs": before,
    "rust_closure": {
        path: digest(root / path) for path in sorted(before) if path.endswith(".rs")
    },
    "cargo_inputs": {path: digest(root / path) for path in sorted(cargo_paths)},
    "native_acceptance": {path: digest(root / path) for path in sorted(native_paths)},
    "project_inputs": {
        path: digest(project / path) for path in sorted(set(project_paths))
    },
    "generated_files": generated,
    "proof_files": proofs,
}
write(project / "source-manifest.json", manifest)
proof_manifest = {
    "version": 2,
    "source_manifest_sha256": digest(project / "source-manifest.json"),
    "files": {path: digest(project / path) for path in proofs},
    "required_theorems": sorted(checker.REQUIRED_THEOREMS),
}
write(project / "proof-manifest.json", proof_manifest)
print(
    json.dumps(
        {
            "status": "sealed",
            "proof_modules": len(proofs),
            "extraction_inputs": len(before),
            "rust_files": len(manifest["rust_closure"]),
            "required_theorems": proof_manifest["required_theorems"],
        },
        indent=2,
    )
)
