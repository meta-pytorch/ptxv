# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Compare BF16 MMA helpers; optionally extend the domain to signed overflow."""

from __future__ import annotations

import argparse
import json
import random
import struct
from pathlib import Path

from cross_check import checked, PROBE, REPO, ROOT, run, sha


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--rust", type=Path, default=PROBE / "target/release/ptxv-lean-bridge"
    )
    parser.add_argument(
        "--capture",
        type=Path,
        help="Optional original M128/N8/K16 BF16 holdout directory",
    )
    parser.add_argument(
        "--check-overflow",
        action="store_true",
        help="Also compare signed overflow; returns nonzero on disagreement",
    )
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rust = args.rust.resolve()
    lean = ROOT / ".lake/build/bin/check_ptxv"
    rng = random.Random(2591551744)
    rows = [[0, 0x16B2] + [0] * 15 + [0x96E5] + [0] * 15]
    for _ in range(8192):
        ab = [
            rng.randrange(2) << 15 | rng.randrange(80, 171) << 7 | rng.randrange(128)
            for _ in range(32)
        ]
        rows.append([0] + ab)
    for sign in (0, 0x8000):
        for exponent in range(1, 255):
            rows.append([0, sign | exponent << 7] + [0] * 15 + [0x3F80] * 16)
    for c in (0, 0x80000000):
        for sign in (0, 0x8000):
            rows.append([c] + [sign] * 16 + [0x3F80] * 16)
    inp, output = out / "mma.inputs", out / "mma.outputs"
    inp.write_bytes(b"".join(struct.pack("<33I", *row) for row in rows))
    checked([str(rust), "mma.bf16", str(inp), str(output)])
    log = checked([str(lean), "mma.bf16", str(inp), str(output)])
    if output.read_bytes()[:4] != b"\0" * 4:
        raise AssertionError(
            "The previously reported underflow witness must produce +0"
        )
    overflow = []
    for sign in (0, 0x80000000) if args.check_overflow else ():
        x, y, expected = (
            out / f"overflow-{sign}.inputs",
            out / f"overflow-{sign}.rust",
            out / f"overflow-{sign}.lean",
        )
        x.write_bytes(
            struct.pack(
                "<33I", 0, (sign >> 16) | 0x7F00, *([0] * 15), 0x4000, *([0] * 15)
            )
        )
        expected.write_bytes(struct.pack("<I", sign | 0x7F800000))
        checked([str(rust), "mma.bf16", str(x), str(y)])
        checked([str(lean), "mma.bf16", str(x), str(expected)])
        actual = struct.unpack("<I", y.read_bytes())[0]
        wrong = out / f"overflow-{sign}.saturated"
        wrong.write_bytes(struct.pack("<I", sign | 0x7F7FFFFF))
        p = run([str(lean), "mma.bf16", str(x), str(wrong)])
        if not p.returncode or "Mismatch" not in p.stdout + p.stderr:
            raise AssertionError("Lean accepted the old saturation result")
        overflow.append(
            {
                "input": [hex((sign >> 16) | 0x7F00), "0x4000"],
                "rust": hex(actual),
                "lean": hex(sign | 0x7F800000),
                "matches": actual == sign | 0x7F800000,
            }
        )
    report = {
        "scope": "BF16 normal/zero operands, C=+/-0, bounded synthetic sums",
        "ptxv_revision": checked(["git", "rev-parse", "HEAD"], cwd=REPO).strip(),
        "matching_rows": len(rows),
        "underflow_witness_positive_zero": True,
        "disagreements": [case for case in overflow if not case["matches"]],
        "overflow_checks": overflow,
        "input_sha256": sha(inp),
        "output_sha256": sha(output),
        "sources": {
            str(p.relative_to(REPO)): sha(p)
            for p in [
                REPO / "crates/sim/src/fp.rs",
                ROOT / "Blackwell/PtxvCheck.lean",
                ROOT / "Blackwell/GB300Math/TensorAccumulatorBlackwell.lean",
                ROOT / "Shared/GB300Math/TensorCore.lean",
                Path(__file__).resolve(),
            ]
        },
    }
    if args.capture:
        capture = args.capture
        metadata = json.loads((capture / "capture.json").read_text())
        if metadata["sm"] != 103:
            raise ValueError("This replay requires the retained SM103 capture")
        raw = (capture / "bf16.inputs").read_bytes()
        hw = (capture / "dense_bf16_0.outputs").read_bytes()
        if not raw or len(raw) % 12800 or len(hw) != len(raw) // 12800 * 4096:
            raise ValueError("Invalid M128/N8/K16 capture extents")
        replay_rows, expected = [], []
        for matrix, words in enumerate(struct.iter_unpack("<3200I", raw)):
            if not all(
                x & 0x7FFF == 0 or 1 <= (x >> 7 & 255) <= 254 for x in words[:2176]
            ):
                continue
            for cell in range(1024):
                a = words[cell // 8 * 16 : cell // 8 * 16 + 16]
                b = words[2048 + cell % 8 * 16 : 2048 + cell % 8 * 16 + 16]
                replay_rows.append((0,) + a + b)
                expected.append(
                    struct.unpack_from("<I", hw, 4 * (matrix * 1024 + cell))[0]
                )
        if not replay_rows:
            raise ValueError("No finite normal/zero input matrices in capture")
        x, y = out / "retained.inputs", out / "retained.outputs"
        x.write_bytes(b"".join(struct.pack("<33I", *row) for row in replay_rows))
        checked([str(rust), "mma.bf16", str(x), str(y)])
        if y.read_bytes() != struct.pack(f"<{len(expected)}I", *expected):
            raise AssertionError("Rust disagrees with retained GB300 outputs")
        print(checked([str(lean), "mma.bf16", str(x), str(y)]), flush=True)
        report["retained_capture"] = {
            "matching_outputs": len(expected),
            "sm": metadata["sm"],
            "cuda_driver_api_version": metadata["driver_version"],
            "driver_release": metadata.get("driver_release"),
            "capture_sha256": sha(capture / "capture.json"),
            "inputs_sha256": sha(capture / "bf16.inputs"),
            "outputs_sha256": sha(capture / "dense_bf16_0.outputs"),
        }
    report["passed"] = not report["disagreements"]
    (out / "mma-cross-check.json").write_text(json.dumps(report, indent=2) + "\n")
    print(log + json.dumps(report, indent=2))
    return int(not report["passed"])


if __name__ == "__main__":
    raise SystemExit(main())
