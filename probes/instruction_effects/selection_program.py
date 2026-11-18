#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Verify actual comparison construction, bindings, dispatch, and scratch-buffer delivery."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "lean_model/instruction_effects"
SUBDIR = "Instruction/SelectionProgram/Code"
TARGET = "Instruction.SelectionHardware.Composition"
MUTATIONS = [
    (
        "selection-program-reverses-arguments",
        "args.push(x);\n    args.push(y);",
        "args.push(y);\n    args.push(x);",
    ),
    (
        "selection-program-rebinds-comparison-source",
        "comparison_regs.push((xreg, SVal::Arg(0)));",
        "comparison_regs.push((xreg, SVal::Arg(1)));",
    ),
    (
        "selection-program-rebinds-predicate",
        "selection_regs.push((predicate, SVal::Op(0)));",
        "selection_regs.push((predicate, SVal::Arg(0)));",
    ),
    (
        "selection-program-returns-comparison",
        "(args, ops, SVal::Op(1))",
        "(args, ops, SVal::Op(0))",
    ),
    (
        "selection-program-loses-comparison-pc",
        "pc: comparison_pc,\n        kind: SKind::Inst {",
        "pc,\n        kind: SKind::Inst {",
    ),
]
MUTATIONS += [
    (
        "selection-evaluator-repeats-first-source",
        "let y = fmod(evaluator.read(b), neg[1], abs[1]);",
        "let y = fmod(evaluator.read(a), neg[1], abs[1]);",
    ),
    (
        "selection-evaluator-reuses-first-modifiers",
        "let y = fmod(evaluator.read(b), neg[1], abs[1]);",
        "let y = fmod(evaluator.read(b), neg[0], abs[0]);",
    ),
    (
        "selection-evaluator-changes-comparison-mode",
        "comparison: *cmp,",
        "comparison: FCmp::Eq,",
    ),
    ("selection-evaluator-drops-ftz", "ftz: *ftz,", "ftz: false,"),
    (
        "selection-evaluator-skips-condition-observation",
        "let predicate = evaluator.predicate(c);",
        "let predicate = true;",
    ),
    (
        "selection-evaluator-reads-unselected-before-predicate",
        "let value = if evaluator.predicate(p) {\n                evaluator.read(a)\n            } else {\n                evaluator.read(b)\n            };",
        "let other = evaluator.read(b);\n            let value = if evaluator.predicate(p) {\n                evaluator.read(a)\n            } else {\n                other\n            };",
    ),
]
BINDING_CONTROLS = {
    "selection-bindings-wrong-argument-index": ("binding.read_sval", "Reads"),
    "selection-bindings-wrong-prior-index": ("binding.read_sval", "Reads"),
    "selection-bindings-weakens-register-key": ("binding.register_binding", "Lookup"),
    "selection-bindings-skips-first-entry": ("binding.register_binding", "Lookup"),
    "selection-bindings-drops-predicate-negation": (
        "binding.predicate_binding",
        "Lookup",
    ),
    "selection-bindings-missing-predicate-false": (
        "binding.read_predicate",
        "Wrappers",
    ),
}
MUTATIONS += [
    (
        "selection-bindings-wrong-argument-index",
        "args[index as usize] as u64",
        "args[0] as u64",
    ),
    ("selection-bindings-wrong-prior-index", "prior[index as usize]", "prior[0]"),
    (
        "selection-bindings-weakens-register-key",
        "candidate.file == register.file && candidate.idx == register.idx",
        "candidate.file == register.file || candidate.idx == register.idx",
    ),
    (
        "selection-bindings-skips-first-entry",
        "let mut index = 0;",
        "let mut index = 1;",
    ),
    (
        "selection-bindings-drops-predicate-negation",
        "(source, predicate.neg)",
        "(source, false)",
    ),
    ("selection-bindings-missing-predicate-false", "None => true,", "None => false,"),
]


MUTATIONS += [
    ("selection-driver-retains-old-scratch", "    values.clear();", "    ;"),
    (
        "selection-driver-repeats-first-instruction",
        "&program.ops[index]",
        "&program.ops[0]",
    ),
    ("selection-driver-appends-zero", "values.push(result);", "values.push(0);"),
    (
        "selection-driver-suppresses-callback-error",
        "Err(error) => return Err(error),",
        "Err(error) => return Ok(0),",
    ),
    (
        "selection-driver-clears-callback-prior",
        "evaluator.evaluate(&program.ops[index], program.args, values)",
        "evaluator.evaluate(&program.ops[index], program.args, &[])",
    ),
    (
        "selection-driver-drops-output-arguments",
        "binding::read_sval(program.out, program.args, values)",
        "binding::read_sval(program.out, &[], values)",
    ),
]


BOUND_CONTROLS = {
    "selection-bound-drops-bindings": ("bound.try_evaluate", "Semantics"),
    "selection-bound-forces-register-file": (
        "bound.Sources.Insts.Ptxv_simSegComparisonEvaluator.read",
        "Reads",
    ),
    "selection-bound-drops-predicate-negation": (
        "bound.Sources.Insts.Ptxv_simSegComparisonEvaluator.predicate",
        "Reads",
    ),
    "selection-bound-accepts-constant-bank": ("bound.direct_value", "Eligibility"),
    "selection-bound-replays-direct-in-fallback": (
        "bound.Evaluator.Insts.Ptxv_simSegProgramProgramEvaluator.evaluate",
        "Semantics",
    ),
    "selection-bound-drops-fallback-prior": (
        "bound.Evaluator.Insts.Ptxv_simSegProgramProgramEvaluator.evaluate",
        "Semantics",
    ),
}
MUTATIONS += [
    ("selection-bound-drops-bindings", "bindings: regs,", "bindings: &[],"),
    (
        "selection-bound-forces-register-file",
        "binding::register_bits(*register, self.bindings, self.inputs)",
        "binding::register_bits(Reg { file: ptxv_ir::File::R, idx: register.idx }, self.bindings, self.inputs)",
    ),
    (
        "selection-bound-drops-predicate-negation",
        "binding::read_predicate(predicate, self.bindings, self.inputs)",
        "binding::read_predicate(&Pred { neg: false, ..*predicate }, self.bindings, self.inputs)",
    ),
    (
        "selection-bound-accepts-constant-bank",
        "matches!(value, Val::Reg(_, _) | Val::Imm(_))",
        "!matches!(value, Val::SReg(_))",
    ),
    (
        "selection-bound-replays-direct-in-fallback",
        "Some(bits) => Ok(bits),",
        "Some(_) => self.fallback.evaluate(instruction, inputs),",
    ),
    (
        "selection-bound-drops-fallback-prior",
        "self.fallback.evaluate(instruction, inputs)",
        "self.fallback.evaluate(instruction, binding::Inputs { args: inputs.args, prior: &[] })",
    ),
]


IEEE_CONTROLS = {
    "ieee-comparison-erases-sign": (
        "crates/sim/src/fp/comparison.rs",
        "fp.comparison.classify",
        "Refinement",
    ),
    "ieee-comparison-distinguishes-signed-zero": (
        "crates/sim/src/fp/comparison.rs",
        "fp.comparison.classify",
        "Refinement",
    ),
    "ieee-comparison-treats-infinity-as-nan": (
        "crates/sim/src/fp/comparison.rs",
        "fp.comparison.classify",
        "Refinement",
    ),
    "ieee-comparison-drops-ftz": (
        "crates/sim/src/fp/comparison.rs",
        "fp.comparison.classify",
        "Refinement",
    ),
    "ieee-comparison-ordered-ne-accepts-nan": (
        "crates/sim/src/fp/comparison.rs",
        "fp.comparison.compare",
        "Refinement",
    ),
    "ieee-comparison-loses-inclusive-tie": (
        "crates/sim/src/fp/comparison.rs",
        "fp.comparison.compare",
        "Refinement",
    ),
    "ieee-comparison-wrapper-drops-ftz": (
        "crates/sim/src/fp.rs",
        "fp.f32_cmp",
        "Refinement",
    ),
    "ieee-comparison-primitive-replaces-mode": (
        "crates/sim/src/seg/bound.rs",
        "seg.bound.IeeeComparison.Insts.Ptxv_simSegBoundComparisonPrimitive.compare",
        "Primitive",
    ),
}
MUTATIONS += [
    (
        "ieee-comparison-erases-sign",
        "let key = if negative { -magnitude } else { magnitude };",
        "let key = magnitude;",
    ),
    (
        "ieee-comparison-distinguishes-signed-zero",
        "let key = if negative { -magnitude } else { magnitude };",
        "let key = if negative { -magnitude - 1 } else { magnitude };",
    ),
    (
        "ieee-comparison-treats-infinity-as-nan",
        "let unordered = magnitude > 0x7f80_0000;",
        "let unordered = magnitude >= 0x7f80_0000;",
    ),
    (
        "ieee-comparison-drops-ftz",
        "if flush && magnitude < 0x0080_0000 {",
        "if false {",
    ),
    (
        "ieee-comparison-ordered-ne-accepts-nan",
        "FCmp::Ne => !unordered && first != second,",
        "FCmp::Ne => first != second,",
    ),
    (
        "ieee-comparison-loses-inclusive-tie",
        "FCmp::Le => !unordered && first <= second,",
        "FCmp::Le => !unordered && first < second,",
    ),
    (
        "ieee-comparison-wrapper-drops-ftz",
        "comparison::compare(a, b, cmp, ftz)",
        "comparison::compare(a, b, cmp, false)",
    ),
    (
        "ieee-comparison-primitive-replaces-mode",
        "crate::fp::f32_cmp(request.left, request.right, request.comparison, request.ftz)",
        "crate::fp::f32_cmp(request.left, request.right, ptxv_ir::FCmp::Eq, request.ftz)",
    ),
]


HARDWARE_CONTROLS = {
    name: ("crates/sim/src/seg/hardware.rs", "seg.hardware.evaluate", "Refinement")
    for name in [
        "selection-hardware-replaces-mode",
        "selection-hardware-reverses-observed-words",
        "selection-hardware-drops-negation",
        "selection-hardware-flips-swap",
    ]
}
MUTATIONS += [
    (
        "selection-hardware-replaces-mode",
        "crate::fp::f32_cmp(first, second, comparison, false)",
        "crate::fp::f32_cmp(first, second, FCmp::Eq, false)",
    ),
    (
        "selection-hardware-reverses-observed-words",
        "ptxv_ir::comparison_select::evaluate(plan, first, second, &IeeeComparator)",
        "ptxv_ir::comparison_select::evaluate(plan, second, first, &IeeeComparator)",
    ),
    (
        "selection-hardware-drops-negation",
        "ptxv_ir::comparison_select::evaluate(plan, first, second, &IeeeComparator)",
        "ptxv_ir::comparison_select::evaluate(Plan { negate: false, ..plan }, first, second, &IeeeComparator)",
    ),
    (
        "selection-hardware-flips-swap",
        "ptxv_ir::comparison_select::evaluate(plan, first, second, &IeeeComparator)",
        "ptxv_ir::comparison_select::evaluate(Plan { swap: !plan.swap, ..plan }, first, second, &IeeeComparator)",
    ),
]


HARDWARE_CONTROLS["selection-hardware-restores-old-max-on-gt"] = (
    "crates/sim/src/seg/hardware.rs",
    "seg.hardware.evaluate",
    "Refinement",
)
MUTATIONS.append(
    (
        "selection-hardware-restores-old-max-on-gt",
        "ptxv_ir::comparison_select::evaluate(plan, first, second, &IeeeComparator)",
        "if matches!(plan.comparison, FCmp::Gt) && !plan.negate && !plan.swap {\n        crate::fp::f32_minmax(first, second, false, false, false)\n    } else {\n        ptxv_ir::comparison_select::evaluate(plan, first, second, &IeeeComparator)\n    }",
    )
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_core():
    spec = importlib.util.spec_from_file_location(
        "memory_write_checker", ROOT / "probes/memory_write/check.py"
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import memory-write checker helpers")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_archive(out: Path) -> tuple[Path, list[Path]]:
    workspace = out / "workspace"
    workspace.mkdir()
    shutil.copytree(
        ROOT / "crates", workspace / "crates", ignore=shutil.ignore_patterns("target")
    )
    shutil.copytree(ROOT / "fixtures", workspace / "fixtures")
    shutil.copytree(
        ROOT / "probes",
        workspace / "probes",
        ignore=shutil.ignore_patterns("target", ".lake", "__pycache__"),
    )
    shutil.copyfile(ROOT / "Cargo.toml", workspace / "Cargo.toml")
    shutil.copyfile(
        ROOT / "probes/instruction_effects/selection-Cargo.lock",
        workspace / "Cargo.lock",
    )
    tracked = [
        Path("Cargo.toml"),
        Path("Cargo.lock"),
        Path("env.sh"),
        Path("probes/instruction_effects/selection-Cargo.lock"),
        Path("probes/instruction_effects/selection_program.py"),
        Path("probes/instruction_effects/check.py"),
        Path("probes/memory_write/check.py"),
    ]
    if (ROOT / "fb/env.local.sh").exists():
        tracked.append(Path("fb/env.local.sh"))
    for rust_source in (workspace / "crates").rglob("*.rs"):
        for filename in re.findall(
            r'include_(?:str|bytes)!\(\s*"([^"\n]+)"\s*\)', rust_source.read_text()
        ):
            target = (rust_source.parent / filename).resolve()
            relative = target.relative_to(workspace)
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / relative, target)
                tracked.append(relative)
    tracked.extend(
        p.relative_to(workspace)
        for p in (workspace / "crates").rglob("*")
        if p.is_file()
    )
    tracked.extend(
        p.relative_to(workspace)
        for p in (workspace / "fixtures").rglob("*")
        if p.is_file()
    )
    tracked.extend(
        p.relative_to(workspace)
        for p in (workspace / "probes").rglob("*")
        if p.is_file()
    )
    return workspace, tracked


def hash_sources(tracked: list[Path], proof_files: list[Path]) -> dict[str, str]:
    return {
        str(path): digest(ROOT / path)
        for path in sorted(
            set(tracked + proof_files + [Path("lean_model/generated_lean.py")])
        )
    }


def extract(core, tools, workspace: Path, out: Path, env) -> Path:
    core.run(
        [
            tools["charon"],
            "cargo",
            "--preset=aeneas",
            "--extract-opaque-bodies",
            "--start-from",
            "ptxv_sim::seg::comparison_program_parts",
            "--start-from",
            "ptxv_sim::seg::comparison_instruction",
            "--start-from",
            "ptxv_sim::seg::binding::read_sval",
            "--start-from",
            "ptxv_sim::seg::binding::register_binding",
            "--start-from",
            "ptxv_sim::seg::binding::predicate_binding",
            "--start-from",
            "ptxv_sim::seg::binding::register_bits",
            "--start-from",
            "ptxv_sim::seg::binding::read_predicate",
            "--start-from",
            "ptxv_sim::seg::program::execute",
            "--start-from",
            "ptxv_sim::seg::bound",
            "--start-from",
            "ptxv_sim::fp::f32_cmp",
            "--start-from",
            "ptxv_sim::seg::hardware::evaluate",
            "--start-from",
            "ptxv_sim::machine::apply_thread_flow",
            "--opaque",
            "core",
            "--opaque",
            "std",
            "--opaque",
            "alloc",
            "--include",
            "core::option",
            "--include",
            "core::result",
            "--error-on-warnings",
            "--abort-on-error",
            "--dest-file",
            str(out / "Program.llbc"),
            "--",
            "--offline",
            "--locked",
            "--lib",
            "-p",
            "ptxv-sim",
        ],
        workspace,
        out / "extract.log",
        env,
    )
    fresh = out / "fresh"
    fresh.mkdir()
    core.run(
        [
            tools["aeneas"],
            "-backend",
            "lean",
            "-namespace",
            "PtxvSim",
            "-dest",
            str(fresh),
            "-subdir",
            SUBDIR,
            "-split-files",
            "-abort-on-error",
            "-no-progress-bar",
            str(out / "Program.llbc"),
        ],
        workspace,
        out / "translate.log",
        env,
    )
    generated = fresh / SUBDIR
    if sorted(p.name for p in generated.iterdir()) != ["Funs.lean", "Types.lean"]:
        raise RuntimeError("Unexpected generated files or an external stub")
    core.LEAN_EXPORT.publish(generated, out / "raw-generated")
    return generated


def native(
    core,
    workspace: Path,
    out: Path,
    env,
    filtered: bool = False,
    filter_name: str = "comparison_selection_policy_tests",
) -> int:
    command = [
        "cargo",
        "test",
        "--release",
        "--offline",
        "--locked",
        "-p",
        "ptxv-sim",
        "--lib",
    ]
    if filtered:
        command += [filter_name]
    return core.run(
        command, workspace, out / "native.log", env, expect_success=not filtered
    )


def negative_controls(core, tools, workspace, project, out, env, selected):
    controls = {}
    baseline = (project / SUBDIR / "Funs.lean").read_text()
    for name, old, new in MUTATIONS:
        if selected and name not in selected:
            continue
        work = out / "mutations" / name
        work.mkdir(parents=True)
        mutant = work / "workspace"
        shutil.copytree(workspace, mutant)
        binding_control = name in BINDING_CONTROLS
        bound_control = name in BOUND_CONTROLS
        ieee_control = name in IEEE_CONTROLS
        hardware_control = name in HARDWARE_CONTROLS
        driver_control = name.startswith("selection-driver-")
        source = mutant / (
            "crates/sim/src/seg/bound.rs"
            if bound_control
            else (
                "crates/sim/src/seg/binding.rs"
                if binding_control
                else (
                    "crates/sim/src/seg/program.rs"
                    if driver_control
                    else "crates/sim/src/seg.rs"
                )
            )
        )
        if ieee_control:
            source = mutant / IEEE_CONTROLS[name][0]
        if hardware_control:
            source = mutant / HARDWARE_CONTROLS[name][0]
        original = source.read_text()
        if original.count(old) != 1:
            raise RuntimeError(f"Ambiguous mutation target: {name}")
        source.write_text(original.replace(old, new))
        evaluator_control = name.startswith("selection-evaluator-")
        root = (
            "comparison_instruction"
            if evaluator_control
            else "comparison_program_parts"
        )
        if binding_control:
            root = BINDING_CONTROLS[name][0]
        if driver_control:
            root = "program.execute"
        if bound_control:
            root = BOUND_CONTROLS[name][0]
        header_root = IEEE_CONTROLS[name][1] if ieee_control else "seg." + root
        if hardware_control:
            header_root = HARDWARE_CONTROLS[name][1]
        header_pattern = rf"^def {re.escape(header_root)}\b[\s\S]*?:= do"
        baseline_header = re.search(header_pattern, baseline, re.M)
        if baseline_header is None:
            raise RuntimeError(f"Missing source root signature: {root}")
        status = native(
            core,
            mutant,
            work,
            env,
            filtered=True,
            filter_name=(
                "seg::hardware::tests"
                if hardware_control
                else (
                    "fp::comparison::tests"
                    if ieee_control
                    else (
                        "seg::bound::tests"
                        if bound_control
                        else (
                            "seg::program::tests"
                            if driver_control
                            else (
                                "seg::binding::tests"
                                if binding_control
                                else (
                                    "comparison_evaluator_tests"
                                    if evaluator_control
                                    else "comparison_selection_policy_tests"
                                )
                            )
                        )
                    )
                )
            ),
        )
        if (
            status == 0
            or "test result: FAILED" not in (work / "native.log").read_text()
        ):
            raise RuntimeError(f"Mutation escaped native segment tests: {name}")
        generated = extract(core, tools, mutant, work, env)
        header = re.search(header_pattern, (generated / "Funs.lean").read_text(), re.M)
        if header is None or " ".join(header.group().split()) != " ".join(
            baseline_header.group().split()
        ):
            raise RuntimeError(f"Mutation changed constructor signature: {name}")
        proof = work / "lean"
        shutil.copytree(project, proof, symlinks=True)
        for path in generated.iterdir():
            shutil.copyfile(path, proof / SUBDIR / path.name)
        core.run(
            [tools["lake"], "build", "Instruction.SelectionProgram.Code.Funs"],
            proof,
            work / "code.log",
            env,
        )
        theorem = (
            "Instruction.IEEEComparison." + IEEE_CONTROLS[name][2]
            if ieee_control
            else (
                "Instruction.SelectionBound." + BOUND_CONTROLS[name][1]
                if bound_control
                else (
                    "Instruction.SelectionDriver.Refinement"
                    if driver_control
                    else (
                        "Instruction.SelectionBindings." + BINDING_CONTROLS[name][1]
                        if binding_control
                        else (
                            "Instruction.SelectionEvaluator.Refinement"
                            if evaluator_control
                            else "Instruction.SelectionProgram.Shape"
                        )
                    )
                )
            )
        )
        if hardware_control:
            theorem = "Instruction.SelectionHardware." + HARDWARE_CONTROLS[name][2]
        rejected = core.run(
            [tools["lake"], "build", theorem],
            proof,
            work / "proof.log",
            env,
            expect_success=False,
        )
        if (
            rejected == 0
            or f"error: {theorem.replace(chr(46), chr(47))}.lean:"
            not in (work / "proof.log").read_text()
        ):
            raise RuntimeError(f"Mutation escaped constructor theorem: {name}")
        controls[name] = {
            "source_sha256": digest(source),
            "native_exit": status,
            "generated_build": True,
            "signature_preserved": True,
            "proof_exit": rejected,
            "theorem": theorem,
        }
    return controls


def native_environment(core, tools, out: Path, workspace: Path):
    env = core.verification_environment(out, tools)
    data = subprocess.check_output(
        [
            "bash",
            "-e",
            "-c",
            'source "$1"; env -0',
            "selection-environment",
            str(ROOT / "env.sh"),
        ],
        cwd=ROOT,
        env=env,
    )
    env = dict(entry.split("=", 1) for entry in data.decode().split("\0") if entry)
    env.update(
        CARGO_HOME=os.environ.get(
            "CARGO_HOME", env.get("CARGO_HOME", str(Path.home() / ".cargo"))
        ),
        CARGO_NET_OFFLINE="true",
        CUDA_VISIBLE_DEVICES="",
        PTXV_EXCLUDE_GPUS="0,1,2,3,4,5,6,7",
        CARGO_TARGET_DIR=str(out / "target"),
    )
    env["PTXV_ROOT"] = str(workspace)
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tools", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--negative-controls", action="store_true")
    parser.add_argument("--control", action="append", choices=[m[0] for m in MUTATIONS])
    args = parser.parse_args()
    if args.control and not args.negative_controls:
        parser.error("--control requires --negative-controls")
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    (out / "receipt.json").unlink(missing_ok=True)
    core = load_core()
    tools = json.loads(args.tools.read_text())
    workspace, tracked = source_archive(out)
    env = native_environment(core, tools, out, workspace)
    project = out / "lean"
    snapshot = out / "proof-sources"
    shutil.copytree(
        PROJECT,
        snapshot,
        ignore=shutil.ignore_patterns(".lake", "lake-manifest.json"),
    )
    shutil.copytree(snapshot, project, dirs_exist_ok=True)
    proof_files = [
        Path("lean_model/instruction_effects") / p.relative_to(snapshot)
        for p in snapshot.rglob("*")
        if p.is_file()
    ]
    before = hash_sources(tracked, proof_files)
    for path in tracked:
        archived = workspace / path
        if (
            path != Path("Cargo.lock")
            and archived.is_file()
            and digest(archived) != before[str(path)]
        ):
            raise RuntimeError(f"Rust source changed while archiving: {path}")
    for path in proof_files:
        archived = snapshot / path.relative_to("lean_model/instruction_effects")
        if digest(archived) != before[str(path)]:
            raise RuntimeError(f"Proof source changed while archiving: {path}")
    tool_hashes = {
        name: digest(Path(tools[name])) for name in ["charon", "aeneas", "lake"]
    }
    backend_hash = digest(Path(tools["backend"]) / "lake-manifest.json")
    native(core, workspace, out, env)
    generated = extract(core, tools, workspace, out, env)
    for path in generated.iterdir():
        target = PROJECT / SUBDIR / path.name
        if args.regenerate:
            shutil.copyfile(path, target)
        elif path.read_bytes() != target.read_bytes():
            raise RuntimeError(f"Stale generated file: {target}")
        shutil.copyfile(path, project / SUBDIR / path.name)
    if args.regenerate:
        after = hash_sources(tracked, proof_files)
        allowed = {
            str(Path("lean_model/instruction_effects") / SUBDIR / name)
            for name in ["Funs.lean", "Types.lean"]
        }
        if {k: v for k, v in before.items() if k not in allowed} != {
            k: v for k, v in after.items() if k not in allowed
        }:
            raise RuntimeError("Non-generated sources changed during extraction")
        before = after
    for path in project.rglob("*.lean"):
        if ".lake" not in path.parts and re.search(
            r"\b(sorry|admit|axiom|native_decide|bv_decide)\b", path.read_text()
        ):
            raise RuntimeError(f"Forbidden proof shortcut: {path}")
    core.configure(project, Path(tools["backend"]))
    manifest_path = project / "lake-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["name"] = "ptxv_instruction_effects"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    core.run(
        [tools["lake"], "build", "Instruction"], project, out / "lean-build.log", env
    )
    core.run(
        [tools["lake"], "env", "lean", "-j4", "Audit.lean"],
        project,
        out / "axiom-audit.log",
        env,
    )
    controls = (
        negative_controls(
            core, tools, workspace, project, out, env, set(args.control or [])
        )
        if args.negative_controls
        else {}
    )
    if hash_sources(tracked, proof_files) != before:
        raise RuntimeError("Sources changed during verification")
    if tool_hashes != {name: digest(Path(tools[name])) for name in tool_hashes}:
        raise RuntimeError("Verification tools changed")
    if backend_hash != digest(Path(tools["backend"]) / "lake-manifest.json"):
        raise RuntimeError("Backend dependencies changed")
    receipt = {
        "source_sha256": before,
        "generated_export": json.loads((out / "raw-generated/export.json").read_text()),
        "tool_sha256": tool_hashes,
        "backend_manifest_sha256": backend_hash,
        "fresh_extraction": True,
        "native_tests": True,
        "lean_build": True,
        "axiom_audit": True,
        "negative_controls": controls,
    }
    (out / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    )
    print(f"Verified actual comparison segment execution: {out / 'receipt.json'}")


if __name__ == "__main__":
    main()
