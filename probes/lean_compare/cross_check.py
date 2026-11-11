# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare ptxv's Rust arithmetic with Lean and retained GB300 scalar samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import struct
import subprocess
from pathlib import Path

PROBE = Path(__file__).resolve().parent
REPO = PROBE.parents[1]
ROOT = REPO / "lean_model"
NATIVE = (
    "ex2.ftz",
    "rcp.ftz",
    "rsq.ftz",
    "sqrt.ftz",
    "lg2.ftz",
    "sin.raw",
    "cos.raw",
    "tanh",
    "rcp64h",
    "rsq64h",
    "ex2.f16",
    "ex2.bf16",
    "tanh.f16",
    "tanh.bf16",
)
WRAPPED = ("ex2", "rcp", "rsq", "sqrt", "lg2", "sin", "cos")


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(command, text=True, capture_output=True, **kwargs)


def checked(command: list[str], **kwargs) -> str:
    p = run(command, **kwargs)
    if p.returncode:
        raise RuntimeError(f"{command}:\n{p.stdout}{p.stderr}")
    return p.stdout


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_floatlib(path: Path) -> dict:
    pin = json.loads((ROOT / "floatlib-lock.json").read_text())
    inventory = ROOT / "floatlib-sources.json"
    if sha(inventory) != pin["source_inventory_sha256"]:
        raise ValueError("FloatLib source inventory does not match the lockfile")
    sources = json.loads(inventory.read_text())
    if len(sources) != pin["source_inventory_files"]:
        raise ValueError("FloatLib source inventory has the wrong size")
    for name, expected in sources.items():
        if sha(path / name) != expected["sha256"]:
            raise ValueError(
                f"FloatLib source differs from the pinned revision: {name}"
            )
    if (path / "lean-toolchain").read_text() != (ROOT / "lean-toolchain").read_text():
        raise ValueError("FloatLib and this package must use the same Lean toolchain")
    return {"pin": pin, "source_files_verified": len(sources)}


def pack(values) -> bytes:
    return b"".join(struct.pack("<I", value) for value in values)


def random_input(i: int) -> int:
    mask = (1 << 64) - 1
    z = ((i + 1) * 0x9E3779B97F4A7C15) & mask
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & mask
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & mask
    return (z ^ (z >> 31)) & 0xFFFFFFFF


def corpus() -> set[int]:
    words = set()
    for sign in (0, 1 << 31):
        for exponent in range(256):
            for boundary in range(0, (1 << 23) + 1, 1 << 16):
                for delta in (-1, 0, 1):
                    fraction = boundary + delta
                    if 0 <= fraction < (1 << 23):
                        words.add(sign | exponent << 23 | fraction)
    rng = random.Random(2587683758)
    words.update(rng.getrandbits(32) for _ in range(65536))
    words.update(x << 16 for x in range(65536))
    # High-word FP64 inputs have an 11-bit exponent and a 20-bit fraction.
    for sign in (0, 1 << 31):
        for exponent in range(2048):
            for fraction in (0, 1, 0x7FFFF, 0x80000, 0xFFFFE, 0xFFFFF):
                words.add(sign | exponent << 20 | fraction)
    return words


def reject(command: list[str], reason: str) -> None:
    p = run(command)
    if not p.returncode or reason not in p.stdout + p.stderr:
        raise RuntimeError(f"Expected rejection ({reason}): {p.stdout}{p.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rust", type=Path, default=PROBE / "target/release/ptxv-lean-bridge"
    )
    parser.add_argument(
        "--floatlib",
        type=Path,
        help="Already-built FloatLib checkout matching floatlib-lock.json",
    )
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rust = args.rust.resolve()
    lean = ROOT / ".lake/build/bin/check_ptxv"
    for binary in (rust, lean):
        if not binary.is_file():
            raise FileNotFoundError(f"Build the bridge first: {binary}")
    samples_path = REPO / "fixtures/mufu/samples.json"
    samples = {s["op"]: s for s in json.loads(samples_path.read_text())}
    approx_env = os.environ.copy()
    lean_executable = None
    floatlib_validation = None
    if args.floatlib:
        args.floatlib = args.floatlib.resolve()
        floatlib_validation = validate_floatlib(args.floatlib)
        dependency_path = checked(
            ["lake", "env", "printenv", "LEAN_PATH"], cwd=args.floatlib
        ).strip()
        approx_env["LEAN_PATH"] = (
            str(ROOT / ".lake/build/lib/lean") + os.pathsep + dependency_path
        )
        lean_executable = checked(["lake", "env", "which", "lean"], cwd=ROOT).strip()
    words = corpus()
    for sample in samples.values():
        words.update(x for x, _ in sample["specials"])
        words.update(random_input(i) for i in range(len(sample["random_raw"])))
    common = sorted(words)
    report = {
        "scope": "Rust/Lean word regression and retained GB300 samples; no new GPU capture or universal equivalence proof",
        "ptxv_revision": checked(["git", "rev-parse", "HEAD"], cwd=REPO).strip(),
        "samples_sha256": sha(samples_path),
        "rows": [],
        "negative_controls": [],
        "sources": {},
        "floatlib_validation": floatlib_validation,
    }
    failed = False
    operations = NATIVE + (WRAPPED if args.floatlib else ())
    for op in operations:
        sample = samples[op]
        xs = list(range(65536)) if op.endswith((".f16", ".bf16")) else common
        inp, output = out / f"{op}.inputs", out / f"{op}.outputs"
        inp.write_bytes(pack(xs))
        checked([str(rust), op, str(inp), str(output)])
        raw = output.read_bytes()
        if len(raw) != len(xs) * 4:
            raise RuntimeError(f"{op}: invalid Rust output length")
        ys = dict(zip(xs, struct.unpack(f"<{len(xs)}I", raw)))
        recorded = sample["specials"] + [
            [random_input(i), y] for i, y in enumerate(sample["random_raw"])
        ]
        hardware_bad = sum(ys[x] != y for x, y in recorded)
        hardware_words = len(recorded)
        if op.endswith((".f16", ".bf16")):
            h = 0xCBF29CE484222325
            for byte in raw:
                h = ((h ^ byte) * 0x100000001B3) & ((1 << 64) - 1)
            hardware_bad += h != int(sample["chunk_hashes"][0], 16)
            hardware_words = 65536
        if op in WRAPPED:
            wide_input, wide_output = (
                out / f"{op}.wide-input",
                out / f"{op}.wide-output",
            )
            wide_input.write_bytes(
                b"".join(struct.pack("<8I", x, 0, 0, 0, 0, 0, 0, 0) for x in xs)
            )
            wide_output.write_bytes(b"".join(struct.pack("<2I", ys[x], 0) for x in xs))
            manifest = out / f"{op}.tsv"
            manifest.write_text(f"{op}\tfalse\t{wide_input}\t{wide_output}\n")
            command = [
                lean_executable,
                "--run",
                str(ROOT / "Shared/Approx.lean"),
                str(manifest),
            ]
        else:
            command = [str(lean), op, str(inp), str(output)]
        p = run(command, env=approx_env if op in WRAPPED else None)
        (out / f"{op}.log").write_text(p.stdout + p.stderr)
        row = {
            "operation": op,
            "words": len(xs),
            "lean_passed": p.returncode == 0,
            "recorded_hardware_words": hardware_words,
            "hardware_mismatches": hardware_bad,
            "input_sha256": sha(inp),
            "output_sha256": sha(output),
        }
        report["rows"].append(row)
        print(json.dumps(row), flush=True)
        failed |= p.returncode != 0 or hardware_bad != 0
    inp, output = out / "rcp.ftz.inputs", out / "rcp.ftz.outputs"
    bad = bytearray(output.read_bytes())
    bad[0] ^= 1
    damaged = out / "damaged.outputs"
    damaged.write_bytes(bad)
    reject([str(lean), "rcp.ftz", str(inp), str(damaged)], "Mismatch")
    report["negative_controls"].append("changed output bit rejected")
    reject([str(lean), "rcp", str(inp), str(output)], "Unsupported")
    report["negative_controls"].append("PTX spelling cannot silently select bare MUFU")
    malformed = out / "malformed.inputs"
    malformed.write_bytes(inp.read_bytes()[:-1])
    reject([str(lean), "rcp.ftz", str(malformed), str(output)], "Invalid byte lengths")
    report["negative_controls"].append("partial input word rejected")
    for path in sorted((ROOT / "Shared").rglob("*.lean")) + sorted(
        (ROOT / "Blackwell").rglob("*.lean")
    ):
        report["sources"][str(path.relative_to(REPO))] = sha(path)
    for path in (
        REPO / "crates/sim/src/mufu.rs",
        REPO / "crates/sim/src/mufu_tables.rs",
        REPO / "crates/sim/src/fp.rs",
        PROBE / "Cargo.lock",
        PROBE / "src/main.rs",
        Path(__file__).resolve(),
    ):
        report["sources"][str(path.relative_to(REPO))] = sha(path)
    report["passed"] = not failed
    (out / "cross-check.json").write_text(json.dumps(report, indent=2) + "\n")
    if failed:
        raise SystemExit(
            "Cross-check failed; inspect per-operation logs and cross-check.json"
        )


if __name__ == "__main__":
    main()
