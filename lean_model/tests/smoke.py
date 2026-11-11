# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Exercise the standalone word interface and its rejection paths."""

from __future__ import annotations

import json
import struct
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def check(
    binary: str,
    operation: str,
    inputs: bytes,
    outputs: bytes | bytearray,
    expected_error: str | None = None,
) -> dict:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root / "input").write_bytes(inputs)
        (root / "output").write_bytes(outputs)
        result = subprocess.run(
            [
                str(ROOT / ".lake/build/bin" / binary),
                operation,
                str(root / "input"),
                str(root / "output"),
                "4",
                "4",
            ],
            capture_output=True,
            text=True,
        )
        if expected_error is None:
            assert result.returncode == 0, result.stdout + result.stderr
        else:
            assert result.returncode != 0, "Invalid evidence was accepted"
            assert expected_error in result.stdout + result.stderr
        return {
            "binary": binary,
            "operation": operation,
            "expected_error": expected_error,
            "passed": True,
        }


def words(values: list[int]) -> bytes:
    return struct.pack(f"<{len(values)}I", *values)


def main() -> None:
    inputs = words([0, 0x80000000, 0x3F800000, 0x7F800000])
    outputs = words([0x7F800000, 0xFF800000, 0x3F800000, 0])
    records = [check("check_mufu_sm103", "rcp", inputs, outputs)]
    records.append(
        check(
            "check_mufu_sm103",
            "tanh.bf16",
            words([0, 0x8000, 0x3B81, 0x7F80, 0xFF80]),
            words([0, 0x8000, 0x3B80, 0x3F80, 0xBF80]),
        )
    )
    damaged = bytearray(outputs)
    damaged[0] ^= 1
    records.append(check("check_mufu_sm103", "rcp", inputs, damaged, "Mismatch"))
    records.append(check("check_mufu_sm103", "unknown", inputs, outputs, "Unsupported"))
    records.append(
        check("check_mufu_sm103", "rcp", inputs[:-1], outputs, "Invalid byte lengths")
    )
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
