#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the BSD-style license in the repository root.
"""Compile adversarial Lean modules to test the actual proof audit."""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
THEOREM = "MemorySafety.Source.accepted_rust_program_memory_safe"
SPEC = importlib.util.spec_from_file_location("proof_checker", HERE / "check.py")
checker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(checker)

BYPASS = """import Lean
open Lean Elab Command
set_option debug.skipKernelTC
  true
run_cmd do
  liftCoreM <| addDecl (.thmDecl {
    name := `MemorySafety.kernel_false
    levelParams := []
    type := mkConst ``False
    value := mkConst ``True.intro })
"""

OLD_AUDIT = """import MemorySafety.Claims
import Lean.Util.CollectAxioms
open Lean Elab Command
run_cmd do
  let allowed : List Name := [``propext, ``Classical.choice, ``Quot.sound]
  for declarationName in [``MemorySafety.kernel_false] do
    let axioms ← collectAxioms declarationName
    for axiomName in axioms do
      unless axiomName ∈ allowed do
        throwError "Unexpected axiom {axiomName}"
  logInfo "OLD_AXIOM_AUDIT_ACCEPTED"
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lean", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    lean = args.lean.resolve(strict=True)
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=False)
    environment = dict(os.environ)
    for name in ["LEAN_PATH", "LEAN_SRC_PATH", "LEAN_SYSROOT"]:
        environment.pop(name, None)
    environment.update(
        LEAN_PATH=str(output), LEAN_NUM_THREADS="4", LEAN_IMPORT_WORKERS="4"
    )
    stages = []

    def put(relative, text):
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return path

    def run(label, arguments, expected):
        log = output / (label + ".log")
        started = time.monotonic()
        with log.open("x") as stream:
            result = subprocess.run(
                [str(lean), "-j4", *arguments],
                cwd=output,
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=120,
                check=False,
            )
        stages.append(
            {
                "label": label,
                "argv": [str(lean), "-j4", *arguments],
                "exit_code": result.returncode,
                "seconds": time.monotonic() - started,
                "log_sha256": checker.digest(log),
            }
        )
        if (result.returncode == 0) != expected:
            raise AssertionError(f"Unexpected compiler result for {label}: {log}")
        return log

    def compile_claims(text, label):
        put("MemorySafety/Claims.lean", text)
        run(
            label, ["-o", "MemorySafety/Claims.olean", "MemorySafety/Claims.lean"], True
        )

    audit = put("Audit.lean", checker.audit_source(["MemorySafety.Claims"], [THEOREM]))
    compile_claims(
        f"theorem {THEOREM} : True ∧ True := ⟨.intro,.intro⟩\n", "valid-claims"
    )
    valid = run("valid-audit", [audit.name], True)
    records = [
        json.loads(line.split("ENDPOINT_TYPE ", 1)[1])
        for line in valid.read_text().splitlines()
        if "ENDPOINT_TYPE " in line
    ]
    if len(records) != 1 or records[0]["pretty"] != "True ∧ True":
        raise AssertionError("Valid fixture must expose the reviewed conjunction")
    policy = {
        THEOREM: {
            "sha256": hashlib.sha256(records[0]["canonical"].encode()).hexdigest()
        }
    }
    checker.check_endpoint_types(valid, policy)

    compile_claims(f"theorem {THEOREM} : True := .intro\n", "weakened-claims")
    weakened = run("weakened-audit", [audit.name], True)
    try:
        checker.check_endpoint_types(weakened, policy)
    except ValueError as error:
        if "statement changed" not in str(error):
            raise
    else:
        raise AssertionError("A weakened theorem was accepted")

    put("MemorySafety/KernelBypass.lean", BYPASS)
    run(
        "malformed-module-compiles",
        ["-o", "MemorySafety/KernelBypass.olean", "MemorySafety/KernelBypass.lean"],
        True,
    )
    compile_claims(
        f"import MemorySafety.KernelBypass\ntheorem {THEOREM} : True ∧ True := ⟨.intro,.intro⟩\n",
        "claims-import-malformed-module",
    )
    put("OldAudit.lean", OLD_AUDIT)
    run("axiom-only-control", ["OldAudit.lean"], True)
    run("trust-zero-control", ["--trust=0", "OldAudit.lean"], True)
    rejected = run("kernel-replay-rejects", [audit.name], False)
    if (
        "while replaying declaration 'MemorySafety.kernel_false'"
        not in rejected.read_text()
    ):
        raise AssertionError("Replay must reject the malformed proof itself")
    try:
        checker.require_kernel_checking(BYPASS, "MemorySafety/KernelBypass.lean")
    except ValueError as error:
        if "Kernel checking must not be disabled" not in str(error):
            raise
    else:
        raise AssertionError("Source guard accepted a disabled kernel")
    receipt = {
        "status": "pass",
        "lean_sha256": checker.digest(lean),
        "checker_sha256": checker.digest(HERE / "check.py"),
        "test_sha256": checker.digest(Path(__file__)),
        "stages": stages,
        "controls": [
            "weakened-theorem-type",
            "imported-malformed-proof",
            "disabled-kernel-source",
        ],
        "scope": "Adversarial fixture modules; independent of the actual memory-safety theorem qualification",
    }
    (output / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(f"PASS: theorem-type and kernel-replay controls; {output / 'receipt.json'}")


if __name__ == "__main__":
    main()
