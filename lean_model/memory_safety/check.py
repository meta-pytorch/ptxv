#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the BSD-style license in the repository root.
"""Check strict-memory Rust freshness, build the proofs offline, and audit axioms."""

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
import tomllib
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
EXPORT_SPEC = importlib.util.spec_from_file_location(
    "generated_lean", PROJECT.parent / "generated_lean.py"
)
if EXPORT_SPEC is None or EXPORT_SPEC.loader is None:
    raise RuntimeError("Cannot load generated Lean export policy")
LEAN_EXPORT = importlib.util.module_from_spec(EXPORT_SPEC)
EXPORT_SPEC.loader.exec_module(LEAN_EXPORT)
ALLOWED_AXIOMS = ["propext", "Classical.choice", "Quot.sound"]
REQUIRED_THEOREMS = {
    "MemorySafetyIR.reported_safe_launch_memory_safe",
    "MemorySafetyIR.reported_is_safe_launch_memory_safe",
    "MemorySafetyIR.accepted_certificate_launch_memory_safe",
    "MemorySafety.Source.accepted_rust_program_memory_safe",
    "MemorySafetyExecution.Source.source_check_is_safe_full_memory_safety",
    "MemorySafetyExecution.Source.source_check_reports_safe_full_memory_safety",
    "MemorySafetyExecution.Source.source_check_reports_safe_every_source_program",
    "MemorySafetyExecution.Source.source_check_reports_safe_every_source_execution",
    "MemorySafetyExecution.Source.source_check_is_safe_instruction_spans",
    "MemorySafetyExecution.Native.Independent.source_execute_independent",
    "MemorySafetyExecution.Native.Independent.source_input_verify_all_independent_instances",
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_build_mode(root):
    path = root / "Cargo.toml"
    with path.open("rb") as stream:
        manifest = tomllib.load(stream)
    profiles = manifest.get("profile", {})
    release = profiles.get("release") if isinstance(profiles, dict) else None
    if not isinstance(release, dict) or release.get("overflow-checks") is not True:
        raise ValueError(
            f"Checked release requires [profile.release].overflow-checks = true: {path}"
        )
    packages = release.get("package", {})
    if not isinstance(packages, dict):
        raise ValueError(f"Invalid [profile.release.package] table: {path}")
    for name, override in packages.items():
        if not isinstance(override, dict):
            raise ValueError(f"Invalid release package override {name!r}: {path}")
        if "overflow-checks" in override and override["overflow-checks"] is not True:
            raise ValueError(
                f"Release package override {name!r} must not disable overflow-checks: {path}"
            )


def relative_file(root, relative):
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Manifest path must be relative: {relative}")
    result = root / path
    if not result.is_file():
        raise ValueError(f"Missing required input: {result}")
    return result


def check_records(root, records):
    for relative, expected in records.items():
        actual = digest(relative_file(root, relative))
        if actual != expected:
            raise ValueError(
                f"Stale input: {root / relative}\nexpected {expected}\nactual   {actual}"
            )


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def records_digest(records):
    return hashlib.sha256(
        (json.dumps(records, indent=2, sort_keys=True) + "\n").encode()
    ).hexdigest()


def lean_code(text):
    """Remove nested comments and literals while preserving token boundaries."""
    output = []
    index = 0
    while index < len(text):
        if text.startswith("--", index):
            finish = text.find("\n", index + 2)
            index = len(text) if finish < 0 else finish
            output.append(" ")
        elif text.startswith("/-", index):
            depth = 1
            index += 2
            while index < len(text) and depth:
                if text.startswith("/-", index):
                    depth += 1
                    index += 2
                elif text.startswith("-/", index):
                    depth -= 1
                    index += 2
                else:
                    index += 1
            if depth:
                raise ValueError("Unterminated Lean comment")
            output.append(" ")
        elif text[index] == "'" and (
            literal := re.match(
                r"'(?:[^'\\\n]|\\(?:[abfnrtv\\\"']|x[0-9A-Fa-f]{2}|u[0-9A-Fa-f]{4}|u\{[0-9A-Fa-f]+\}))'",
                text[index:],
            )
        ):
            index += len(literal[0])
            output.append(" ")
        elif text[index] == '"':
            index += 1
            while index < len(text):
                if text[index] == "\\":
                    index += 2
                elif text[index] == '"':
                    index += 1
                    break
                else:
                    index += 1
            else:
                raise ValueError("Unterminated Lean string")
            output.append(" ")
        elif text.startswith("r#", index):
            end = index + 1
            while end < len(text) and text[end] == "#":
                end += 1
            if end < len(text) and text[end] == '"':
                terminator = '"' + text[index + 1 : end]
                finish = text.find(terminator, end + 1)
                if finish < 0:
                    raise ValueError("Unterminated Lean raw string")
                index = finish + len(terminator)
                output.append(" ")
            else:
                output.append(text[index])
                index += 1
        else:
            output.append(text[index])
            index += 1
    return "".join(output)


def require_kernel_checking(text, relative):
    tokens = re.findall(r"[\w']+|«[^»]*»|[^\s]", lean_code(text))
    tokens = [token[1:-1] if token.startswith("«") else token for token in tokens]
    for index in range(len(tokens) - 3):
        if tokens[index : index + 4] == ["set_option", "debug", ".", "skipKernelTC"]:
            if index + 4 == len(tokens) or tokens[index + 4] != "false":
                raise ValueError(f"Kernel checking must not be disabled: {relative}")


def load_endpoint_types():
    value = json.loads((PROJECT / "endpoint-types.json").read_text())
    if value.get("format") != "lean-expr-alpha-mdata-v1":
        raise ValueError("Unknown required theorem type representation")
    records = value["theorems"]
    if set(records) != REQUIRED_THEOREMS:
        raise ValueError(
            "Required theorem type policy must cover every endpoint exactly"
        )
    for name, record in records.items():
        if not re.fullmatch(r"[0-9a-f]{64}", record["sha256"]) or not record["pretty"]:
            raise ValueError(f"Malformed required theorem type policy: {name}")
    return records


SPECIFICATION_FILES = (
    "Frontend/LaunchInventory.lean",
    "Frontend/Semantics.lean",
    "FrontendProgram/Construction.lean",
    "FrontendProgram/Digits.lean",
    "FrontendProgram/Grammar.lean",
    "FrontendProgram/RegisterTypes.lean",
    "FrontendProgram/SourceDimensions.lean",
    "FrontendProgram/Text.lean",
    "FrontendProgram/Tokens.lean",
    "MemorySafety/Program.lean",
    "MemorySafety/Semantics.lean",
    "MemorySafety/Soundness.lean",
    "MemorySafety/SourceModel.lean",
    "MemorySafetyExecution/AccessSafety.lean",
    "MemorySafetyExecution/MachineModelSafety.lean",
    "MemorySafetyExecution/Native/ActionAccess.lean",
    "MemorySafetyExecution/Native/ByteEffects.lean",
    "MemorySafetyExecution/Native/ConcreteSafety.lean",
    "MemorySafetyExecution/Native/FullSafety.lean",
    "MemorySafetyExecution/Native/IndependentModel.lean",
    "MemorySafetyExecution/Native/IndependentSemantics.lean",
    "MemorySafetyExecution/Native/InputInitialRelation.lean",
    "MemorySafetyExecution/Native/Instantiation.lean",
    "MemorySafetyExecution/Native/InstructionAccess.lean",
    "MemorySafetyExecution/Native/Owner.lean",
    "MemorySafetyExecution/Native/PlacementResolution.lean",
    "MemorySafetyExecution/Native/Soundness.lean",
    "MemorySafetyExecution/RunSafety.lean",
    "MemorySafetyExecution/SourceMemorySafety.lean",
    "MemorySafetyExecution/TraceSafety.lean",
    "MemorySafetyIR/ExecutionBase.lean",
    "MemorySafetyIR/Initialization.lean",
    "MemorySafetyIR/LaunchExecution.lean",
    "MemorySafetyIR/LaunchModel.lean",
    "MemorySafetyIR/LaunchRestrictions.lean",
    "MemorySafetyIR/Semantics.lean",
    "ValueFlow/Arithmetic.lean",
    "ValueFlow/Provenance.lean",
    "ValueFlow/Semantics.lean",
    "ValueFlow/SourceModel.lean",
    "ValueFlow/Storage/MachineRegistersVirtualMap.lean",
)


def check_specification_inputs():
    policy = json.loads((PROJECT / "specification-inputs.json").read_text())
    if (
        policy.get("version") != 1
        or policy.get("selection") != "reviewed-memory-safety-specification-v1"
    ):
        raise ValueError("Unknown reviewed specification policy")
    records = policy["files"]
    if set(records) != set(SPECIFICATION_FILES):
        raise ValueError(
            "Reviewed specification policy must cover each required source exactly"
        )
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", expected) for expected in records.values()
    ):
        raise ValueError("Malformed reviewed specification fingerprint")
    check_records(PROJECT, records)
    return {
        "files": len(records),
        "sha256": digest(PROJECT / "specification-inputs.json"),
    }


def check_endpoint_types(log, expected):
    observed = {}
    for line in log.read_text().splitlines():
        if "ENDPOINT_TYPE " not in line:
            continue
        value = json.loads(line.split("ENDPOINT_TYPE ", 1)[1])
        name = value["name"]
        if name in observed:
            raise ValueError(f"Duplicate required theorem type: {name}")
        observed[name] = value
    if set(observed) != set(expected):
        raise ValueError("Audit must report every required theorem type exactly")
    for name, record in expected.items():
        if name not in observed:
            raise ValueError(f"Required theorem type was not audited: {name}")
        actual = hashlib.sha256(observed[name]["canonical"].encode()).hexdigest()
        if actual != record["sha256"]:
            raise ValueError(f"Required theorem statement changed: {name}")
    return {name: record["sha256"] for name, record in expected.items()}


def backend_locations(backend):
    locations = {"aeneas": backend.resolve()}
    manifest = json.loads((backend / "lake-manifest.json").read_text())
    for package in manifest["packages"]:
        name = package["name"]
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name):
            raise ValueError(f"Unsupported dependency name: {name}")
        if name == "aeneas":
            continue
        if name in locations:
            raise ValueError(f"Duplicate dependency name: {name}")
        if package["type"] not in {"path", "git"}:
            raise ValueError(f"Unsupported cached dependency type: {package['type']}")
        for field in ("configFile", "manifestFile"):
            relative = Path(
                package.get(field)
                or ("lakefile.lean" if field == "configFile" else "lake-manifest.json")
            )
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    f"Package configuration must be relative: {name}/{relative}"
                )
        if package["type"] == "path":
            location = Path(package["dir"])
            if not location.is_absolute():
                location = backend / location
        else:
            location = backend / ".lake/packages" / name
        location = location.resolve()
        if not (location / package.get("configFile", "lakefile.lean")).is_file():
            raise ValueError(f"Uncached dependency {name}: {location}")
        locations[name] = location
    return locations


def backend_records(backend):
    packages = {}
    manifest = json.loads((backend / "lake-manifest.json").read_text())
    extra_configs = {
        package["name"]: [package.get("configFile"), package.get("manifestFile")]
        for package in manifest["packages"]
    }
    for name, location in sorted(backend_locations(backend).items()):
        files = {
            location / relative
            for relative in extra_configs.get(name, [])
            if relative and (location / relative).is_file()
        }
        for directory, children, names in os.walk(location):
            children[:] = [
                child
                for child in children
                if child
                not in {".git", ".lake", "target", "node_modules", "__pycache__"}
            ]
            for child in children:
                if (Path(directory) / child).is_symlink():
                    raise ValueError(
                        f"Nested source directory symlink is not inventoried: {directory}/{child}"
                    )
            for filename in names:
                path = Path(directory) / filename
                if filename.endswith(".lean") or (
                    path.parent == location
                    and filename
                    in {"lakefile.toml", "lake-manifest.json", "lean-toolchain"}
                ):
                    files.add(path)
        compiled = location / ".lake/build/lib"
        for directory, children, names in os.walk(compiled):
            for child in children:
                if (Path(directory) / child).is_symlink():
                    raise ValueError(
                        f"Nested compiled directory symlink is not inventoried: {directory}/{child}"
                    )
            for filename in names:
                if filename.endswith(
                    (
                        ".olean",
                        ".olean.private",
                        ".olean.server",
                        ".ilean",
                        ".ir",
                        ".so",
                        ".dylib",
                        ".dll",
                    )
                ):
                    files.add(Path(directory) / filename)
        if name == "aeneas" and not any(path.name.endswith(".olean") for path in files):
            raise ValueError(f"No compiled Lean environment for {name}")
        packages[name] = {
            str(path.relative_to(location)): digest(path) for path in sorted(files)
        }
    return {"version": 1, "packages": packages}


def backend_policy(inventory):
    return {
        "version": 1,
        "selection": "lean-package-sources-and-compiled-environments-v1",
        "packages": {
            name: {"files": len(records), "sha256": records_digest(records)}
            for name, records in inventory["packages"].items()
        },
    }


def load_backend_policy():
    value = json.loads((PROJECT / "backend-inputs.json").read_text())
    if (
        value.get("version") != 1
        or value.get("selection") != "lean-package-sources-and-compiled-environments-v1"
    ):
        raise ValueError("Unknown backend inventory selection policy")
    packages = value["packages"]
    if "aeneas" not in packages:
        raise ValueError("Backend inventory must include the Aeneas package")
    for name, record in packages.items():
        if (
            not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]*", name)
            or not isinstance(record["files"], int)
            or record["files"] < 1
            or not re.fullmatch(r"[0-9a-f]{64}", record["sha256"])
        ):
            raise ValueError(f"Malformed backend inventory policy: {name}")
    return value


def check_backend(backend):
    expected = load_backend_policy()
    actual = backend_policy(backend_records(backend))
    if actual != expected:
        raise ValueError("Lean backend source or compiled dependency inventory changed")
    return {
        "inventory_sha256": digest(PROJECT / "backend-inputs.json"),
        "packages": len(expected["packages"]),
        "files": sum(package["files"] for package in expected["packages"].values()),
    }


def check_fresh_extraction(out, manifest):
    receipt_path = out / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["status"] != "pass" or receipt["generated_byte_equal"] != {
        "Funs.lean": True,
        "Types.lean": True,
    }:
        raise ValueError("Fresh extraction must succeed with exact generated code")
    if len(receipt["outcomes"]) != 2 or any(
        item["exit_code"] for item in receipt["outcomes"]
    ):
        raise ValueError("Both fresh extraction commands must succeed")
    before = json.loads((out / "source-before.json").read_text())
    after = json.loads((out / "source-after.json").read_text())
    if before != after or before != manifest["extraction_inputs"]:
        raise ValueError("Fresh extraction must bind all sealed source inputs exactly")
    if receipt["source_manifest_sha256"] != digest(out / "source-before.json"):
        raise ValueError("Fresh extraction source record changed")
    if receipt["specification_sha256"] != digest(PROJECT / "extraction-spec.json"):
        raise ValueError("Fresh extraction specification changed")
    if receipt["llbc_sha256"] != digest(out / "Program.llbc"):
        raise ValueError("Fresh extracted LLBC changed")
    generated = out / "fresh/MemorySafety/Code"
    if sorted(path.name for path in generated.iterdir()) != ["Funs.lean", "Types.lean"]:
        raise ValueError("Fresh extraction has external templates or unexpected output")
    LEAN_EXPORT.verify(generated, receipt["generated_export"], out / "raw-generated")
    for name in ("Funs.lean", "Types.lean"):
        if (generated / name).read_bytes() != (
            PROJECT / "MemorySafety/Code" / name
        ).read_bytes():
            raise ValueError(f"Fresh extraction is stale: {name}")
    for item in receipt["outcomes"]:
        if digest(Path(item["log"])) != item["log_sha256"]:
            raise ValueError("Fresh extraction command log changed")
    return {"receipt": str(receipt_path), "sha256": digest(receipt_path)}


def freshness(root, manifest):
    check_build_mode(root)
    check_specification_inputs()
    required_inputs = {
        "check.py",
        "extract.py",
        "extraction-inputs.json",
        "extraction-spec.json",
        "qualified-extraction.json",
        "lakefile.lean",
        "lean-toolchain",
        "endpoint-types.json",
        "specification-inputs.json",
        "backend-inputs.json",
        "seal.py",
        "test_check.py",
        "test_kernel.py",
    }
    if not required_inputs.issubset(manifest["project_inputs"]):
        raise ValueError("Source seal omits a required checker or trust-policy input")
    inputs = json.loads((PROJECT / "extraction-inputs.json").read_text())
    for destination, source in inputs.items():
        relative_file(root, source)
        if Path(destination).is_absolute() or ".." in Path(destination).parts:
            raise ValueError(f"Extraction destination must be relative: {destination}")
    if set(manifest["extraction_inputs"]) != set(inputs.values()):
        raise ValueError("Source seal must cover every extraction input exactly")
    for category in (
        "rust_closure",
        "cargo_inputs",
        "native_acceptance",
        "extraction_inputs",
    ):
        check_records(root, manifest[category])
    check_records(PROJECT, manifest["project_inputs"])
    qualified = json.loads((PROJECT / "qualified-extraction.json").read_text())
    if qualified["status"] != "pass" or qualified["generated_byte_equal"] != {
        "Funs.lean": True,
        "Types.lean": True,
    }:
        raise ValueError("A successful genuine extraction is required")
    if qualified["source_manifest_sha256"] != records_digest(
        manifest["extraction_inputs"]
    ):
        raise ValueError("Qualified extraction does not match every current input")
    if qualified["specification_sha256"] != digest(PROJECT / "extraction-spec.json"):
        raise ValueError("Qualified extraction uses a different specification")
    LEAN_EXPORT.verify(PROJECT / "MemorySafety/Code", qualified["generated_export"])
    specification = json.loads((PROJECT / "extraction-spec.json").read_text())
    if specification["generated_export"] != LEAN_EXPORT.POLICY:
        raise ValueError("Extraction specification uses a different Lean export policy")
    if inputs.get("lean_model/generated_lean.py") != "lean_model/generated_lean.py":
        raise ValueError(
            "Extraction must seal the generated Lean export implementation"
        )
    generated = set(manifest["generated_files"])
    if qualified["generated_files"] != {
        path: manifest["project_inputs"][path] for path in generated
    }:
        raise ValueError(
            "Qualified extraction does not match the current generated code"
        )
    inventory = {
        str(path.relative_to(PROJECT))
        for namespace in (
            "MemorySafety",
            "MemorySafetyIR",
            "MemorySafetyExecution",
            "MemoryWrite",
            "ValueFlow",
            "ValueFlowDependencies",
            "Frontend",
            "FrontendProgram",
        )
        for path in (PROJECT / namespace).rglob("*.lean")
        if str(path.relative_to(PROJECT)) not in generated
    }
    if set(manifest["proof_files"]) != inventory:
        raise ValueError("Proof seal must cover every authored project module exactly")
    return {
        "rust_files": len(manifest["rust_closure"]),
        "cargo_files": len(manifest["cargo_inputs"]),
        "native_acceptance_files": len(manifest["native_acceptance"]),
        "extraction_input_files": len(manifest["extraction_inputs"]),
        "proof_module_files": len(inventory),
        "generated_files": list(manifest["generated_files"]),
        "manifest_sha256": digest(PROJECT / "source-manifest.json"),
    }


def configure(project, backend):
    manifest_path = backend / "lake-manifest.json"
    if not manifest_path.is_file() or not (backend / "lakefile.lean").is_file():
        raise ValueError(f"Not a cached Aeneas Lean backend: {backend}")
    manifest = json.loads(manifest_path.read_text())
    packages = []
    for package in manifest["packages"]:
        if package["name"] == "aeneas":
            continue
        package = dict(package, inherited=True)
        if package["type"] == "path":
            dependency = Path(package["dir"])
            if not dependency.is_absolute():
                dependency = (backend / dependency).resolve()
            package["dir"] = str(dependency)
        else:
            dependency = backend / ".lake/packages" / package["name"]
        if not (dependency / package.get("configFile", "lakefile.lean")).is_file():
            raise ValueError(
                f"Uncached dependency {package['name']}: {dependency}; no download attempted"
            )
        packages.append(package)
    packages.append(
        {
            "type": "path",
            "name": "aeneas",
            "dir": str(backend),
            "manifestFile": "lake-manifest.json",
            "configFile": "lakefile.lean",
            "inherited": False,
        }
    )
    manifest.update(name="ptxv_memory_safety_joint", packages=packages)
    (project / ".aeneas-path").write_text(str(backend) + "\n")
    write_json(project / "lake-manifest.json", manifest)
    package_dir = project / ".lake/packages"
    package_dir.mkdir(parents=True)
    for package in packages:
        if package["type"] != "path":
            (package_dir / package["name"]).symlink_to(
                (backend / ".lake/packages" / package["name"]).resolve(),
                target_is_directory=True,
            )


def audit_source(modules, required):
    imports = "\n".join(f"import {module}" for module in modules)
    required_names = ", ".join(f"``{name}" for name in required)
    return (
        imports
        + """
import Lean
import Lean.Util.CollectAxioms
import Lean.Replay
open Lean Elab Command
set_option debug.skipKernelTC false
def endpointLevel (parameters : List Name) : Level → Level
  | .zero => .zero
  | .succ level => .succ (endpointLevel parameters level)
  | .max left right => .max (endpointLevel parameters left) (endpointLevel parameters right)
  | .imax left right => .imax (endpointLevel parameters left) (endpointLevel parameters right)
  | .param levelName => .param (Name.mkSimple ("u" ++ toString (parameters.idxOf levelName)))
  | .mvar levelName => .mvar levelName

def endpointExpression (parameters : List Name) : Expr → Expr
  | .mdata _ value => endpointExpression parameters value
  | .sort level => .sort (endpointLevel parameters level)
  | .const constantName levels => .const constantName (levels.map (endpointLevel parameters))
  | .app function argument => .app (endpointExpression parameters function) (endpointExpression parameters argument)
  | .lam _ domain body info => .lam .anonymous (endpointExpression parameters domain) (endpointExpression parameters body) info
  | .forallE _ domain body info => .forallE .anonymous (endpointExpression parameters domain) (endpointExpression parameters body) info
  | .letE _ type value body nondependent => .letE .anonymous (endpointExpression parameters type) (endpointExpression parameters value) (endpointExpression parameters body) nondependent
  | .proj type index value => .proj type index (endpointExpression parameters value)
  | other => other

def projectModule (moduleString : String) : Bool :=
  moduleString.startsWith "MemorySafety" || moduleString.startsWith "MemoryWrite" ||
    moduleString.startsWith "ValueFlow" || moduleString.startsWith "Frontend"

run_cmd do
  let environment ← getEnv
  let mut projectConstants : Std.HashMap Name ConstantInfo := {}
  let mut externalImports : Array Import := #[]
  for index in [:environment.header.moduleNames.size] do
    let moduleName := environment.header.moduleNames[index]!
    if projectModule moduleName.toString then
      for declaration in environment.header.moduleData[index]!.constants do
        projectConstants := projectConstants.insert declaration.name declaration
    else
      externalImports := externalImports.push { module := moduleName }
  let externalEnvironment ← importModules externalImports {} (trustLevel := 0)
  for (constantName, _) in projectConstants.toList do
    if externalEnvironment.contains constantName then
      throwError "Project constant occurs in trusted external environment: {constantName}"
  let _ ← externalEnvironment.toKernelEnv.replay projectConstants
  logInfo m!"KERNEL_REPLAY_PASS {projectConstants.size} project constants"
  let allowed : List Name := [``propext, ``Classical.choice, ``Quot.sound]
  let required : List Name := ["""
        + required_names
        + """]
  for declarationName in required do
    unless environment.contains declarationName do
      throwError "Required theorem missing: {declarationName}"
    unless (environment.find? declarationName).any (fun declaration => declaration.isTheorem) do
      throwError "Required declaration is not a theorem: {declarationName}"
    let some declaration := environment.find? declarationName |
      throwError "Required theorem missing: {declarationName}"
    let canonical := endpointExpression declaration.levelParams declaration.type
    let pretty ← liftTermElabM (Lean.PrettyPrinter.ppExpr declaration.type)
    let record := Json.mkObj [("name",toJson declarationName.toString),
      ("canonical",toJson (reprStr canonical)),("pretty",toJson pretty.pretty)]
    logInfo m!"ENDPOINT_TYPE {record.compress}"
  let mut checked : Nat := 0
  for index in [:environment.header.moduleNames.size] do
    let moduleName := environment.header.moduleNames[index]!.toString
    if projectModule moduleName then
      for declaration in environment.header.moduleData[index]!.constants do
        match declaration with
        | .axiomInfo _ => throwError "Project axiom is not permitted: {declaration.name}"
        | .thmInfo _ =>
          let axioms ← collectAxioms declaration.name
          for axiomName in axioms do
            unless axiomName ∈ allowed do
              throwError "Unexpected axiom {axiomName} in {declaration.name}"
          checked := checked + 1
          logInfo m!"audit {declaration.name}: {axioms}"
        | _ => pure ()
  unless checked > 0 do throwError "No project theorems were audited"
  logInfo m!"AUDIT_PASS {checked} project theorems; only propext, Classical.choice, Quot.sound"
"""
    )


def run(command, cwd, env, log):
    started = time.monotonic()
    with log.open("x") as output:
        result = subprocess.run(
            command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT
        )
    receipt = {
        "argv": list(map(str, command)),
        "exit_code": result.returncode,
        "seconds": time.monotonic() - started,
        "log": str(log),
        "log_sha256": digest(log),
    }
    if result.returncode:
        tail = "\n".join(log.read_text().splitlines()[-35:])
        raise ValueError(f"Verification command failed; retained {log}\n{tail}")
    return receipt


def load_proofs(manifest, development):
    proof_manifest_path = PROJECT / "proof-manifest.json"
    sealed = proof_manifest_path.is_file()
    if sealed:
        proof_manifest = json.loads(proof_manifest_path.read_text())
        if proof_manifest["source_manifest_sha256"] != digest(
            PROJECT / "source-manifest.json"
        ):
            raise ValueError(
                "Proof seal does not match the current Rust/generated-source manifest"
            )
        check_records(PROJECT, proof_manifest["files"])
        paths = list(proof_manifest["files"])
        required = proof_manifest["required_theorems"]
        if set(paths) != set(manifest["proof_files"]):
            raise ValueError(
                "Proof seal does not cover the full declared project module inventory"
            )
        if len(required) != len(set(required)) or set(required) != REQUIRED_THEOREMS:
            raise ValueError(
                "Proof seal must require every final safety and refinement theorem"
            )
    elif development:
        paths = manifest["proof_files"]
        required = []
    else:
        raise ValueError(
            "Authored proofs are not sealed; use --development only for a provisional build"
        )
    if any(not path.endswith(".lean") for path in paths):
        raise ValueError("Proof manifest files must be Lean modules")
    before = {path: digest(relative_file(PROJECT, path)) for path in paths}
    for relative in paths:
        text = relative_file(PROJECT, relative).read_text()
        require_kernel_checking(text, relative)
    return paths, required, sealed, before


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ptxv-root", type=Path, default=PROJECT.parent.parent)
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Check Rust/generated source hashes without running Lean",
    )
    parser.add_argument(
        "--out", type=Path, help="New private output directory for build and audit"
    )
    parser.add_argument(
        "--backend",
        type=Path,
        help="Existing offline Aeneas Lean backend with cached dependencies",
    )
    parser.add_argument(
        "--lake",
        type=Path,
        help="Installed Lake binary (must match the extraction tool fingerprint)",
    )
    parser.add_argument(
        "--tools", type=Path, help="Pinned offline extraction tool paths"
    )
    parser.add_argument(
        "--development",
        action="store_true",
        help="Allow unsealed authored proofs; result is explicitly provisional",
    )
    args = parser.parse_args()
    root = args.ptxv_root.resolve()
    manifest = json.loads((PROJECT / "source-manifest.json").read_text())
    checked = freshness(root, manifest)
    if args.source_only:
        print(
            json.dumps(
                {
                    "status": "pass",
                    "mode": "source-only",
                    "formal_check": False,
                    **checked,
                },
                indent=2,
            )
        )
        return
    if (
        args.out is None
        or args.backend is None
        or args.lake is None
        or args.tools is None
    ):
        parser.error(
            "Formal checking requires --out, --backend, --lake and --tools; use --source-only for hashes"
        )
    out = args.out.resolve()
    if (
        out.exists()
        or out == PROJECT
        or PROJECT in out.parents
        or out == root
        or root in out.parents
    ):
        raise ValueError(
            "--out must be a new directory outside the proof and PTXV source trees"
        )
    paths, required, sealed, before = load_proofs(manifest, args.development)
    modules = [path[:-5].replace("/", ".") for path in paths]
    expected_types = load_endpoint_types() if sealed else {}
    backend_checked = check_backend(args.backend.resolve())
    specification = json.loads((PROJECT / "extraction-spec.json").read_text())
    lake = args.lake.resolve()
    if digest(lake) != specification["tools"]["lake"]["sha256"]:
        raise ValueError("Lake binary does not match the pinned toolchain")
    lean = lake.with_name("lean")
    if digest(lean) != specification["tools"]["lean"]["sha256"]:
        raise ValueError("Lean binary does not match the pinned toolchain")
    out.mkdir(parents=True)
    stage = out / "lean"
    stage.mkdir()
    stage_files = (
        set(paths)
        | set(manifest["generated_files"])
        | {"lakefile.lean", "lean-toolchain"}
    )
    for relative in stage_files:
        if relative.endswith(".lean"):
            require_kernel_checking(
                relative_file(PROJECT, relative).read_text(), relative
            )
        destination = stage / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(relative_file(PROJECT, relative), destination)
    configure(stage, args.backend.resolve())
    (stage / "VerificationAudit.lean").write_text(audit_source(modules, required))
    env = os.environ.copy()
    for name in ("LEAN_PATH", "LEAN_SRC_PATH", "LEAN_SYSROOT"):
        env.pop(name, None)
    env.update(
        LEAN_NUM_THREADS="4",
        LEAN_IMPORT_WORKERS="4",
        LAKE_NO_CACHE="1",
        CARGO_NET_OFFLINE="true",
    )
    env["PATH"] = str(lake.parent) + os.pathsep + env.get("PATH", "")
    outcomes = []
    try:
        outcomes.append(
            run(
                [
                    sys.executable,
                    str(PROJECT / "extract.py"),
                    "--ptxv-root",
                    str(root),
                    "--tools",
                    str(args.tools.resolve()),
                    "--out",
                    str(out / "extraction"),
                ],
                PROJECT,
                env,
                out / "extraction-driver.log",
            )
        )
        fresh_extraction = check_fresh_extraction(out / "extraction", manifest)
        outcomes.append(
            run(
                [
                    str(lake),
                    "--no-cache",
                    "build",
                    *modules,
                ],
                stage,
                env,
                out / "build.log",
            )
        )
        outcomes.append(
            run(
                [
                    str(lake),
                    "--no-cache",
                    "env",
                    "lean",
                    "--trust=0",
                    "VerificationAudit.lean",
                ],
                stage,
                env,
                out / "audit.log",
            )
        )
        endpoint_types = check_endpoint_types(out / "audit.log", expected_types)
        if check_backend(args.backend.resolve()) != backend_checked:
            raise ValueError("Lean backend changed during verification")
        after = freshness(root, manifest)
        check_records(PROJECT, before)
        if after != checked:
            raise ValueError("Inputs changed during verification")
        write_json(
            out / "receipt.json",
            {
                "status": "pass" if sealed else "development-pass",
                "proofs_sealed": sealed,
                "fresh_extraction": fresh_extraction,
                "endpoint_types": endpoint_types,
                "backend": backend_checked,
                "kernel_import_trust": 0,
                "kernel_replay": "all-project-constants-external-only-environment",
                "source_freshness": checked,
                "proof_files": before,
                "required_theorems": required,
                "allowed_axioms": ALLOWED_AXIOMS,
                "outcomes": outcomes,
            },
        )
    except (OSError, ValueError) as error:
        write_json(
            out / "failure.json",
            {"status": "fail", "error": str(error), "outcomes": outcomes},
        )
        raise
    print(
        json.dumps(
            {
                "status": "pass" if sealed else "development-pass",
                "receipt": str(out / "receipt.json"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print(f"memory-safety verification failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
