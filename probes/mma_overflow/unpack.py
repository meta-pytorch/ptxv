# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license in the root LICENSE.

"""Retain and validate the MMA_OVERFLOW JSON emitted by the GB300 test."""

import argparse
import base64
import hashlib
import json
import struct
import zlib
from pathlib import Path

import probe


def unpack(item):
    data = zlib.decompress(base64.b64decode(item["zlib_base64"]))
    if len(data) != item["bytes"] or hashlib.sha256(data).hexdigest() != item["sha256"]:
        raise ValueError("Capture blob integrity check failed")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--output", type=Path, help="New directory for unpacked artifacts")
    args = parser.parse_args()
    text = args.log.read_text()
    if text.lstrip().startswith("{"):
        report = json.loads(text)
    else:
        lines = [line for line in text.splitlines() if "MMA_OVERFLOW {" in line]
        if len(lines) != 1:
            raise ValueError(f"Expected one capture, found {len(lines)}")
        report = json.loads(lines[0].split("MMA_OVERFLOW ", 1)[1])
    root = Path(__file__).resolve().parent
    if report["compute_capability"] != [10, 3]:
        raise ValueError("Capture is not from SM103")
    if report["ptx_sha256"] != probe.sha((root.parent / "mma_zero/kernel.ptx").read_bytes()):
        raise ValueError("PTX source differs from captured source")
    output = args.output or root / "expected"
    output.mkdir(exist_ok=False)
    (output / "capture.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "kernel.cubin").write_bytes(unpack(report["cubin"]))
    configs = []
    for cfg in report["configs"]:
        name = cfg["entry"] + ("_enabled" if cfg["enabled"] else "_disabled")
        _, expected_inputs = probe.inputs(cfg["entry"], cfg["enabled"])
        for key, item in cfg["inputs"].items():
            data = unpack(item)
            if data != expected_inputs[key]:
                raise ValueError(f"{name}: {key} differs from regenerated input")
            (output / f"{name}.{key}.bin").write_bytes(data)
        data = unpack(cfg["output"])
        if cfg["repeat_hashes"] != [probe.sha(data)] * 3:
            raise ValueError("Repeat hash mismatch")
        cells = 2048 if cfg["entry"].startswith("tc") else 128
        if data != b"".join(struct.pack("<I", case["output"]) * cells for case in cfg["cases"]):
            raise ValueError("Captured summary does not match full output tiles")
        (output / f"{name}.D.bin").write_bytes(data)
        configs.append({k: v for k, v in cfg.items() if k not in ("inputs", "output") } | {"name": name})
    (output / "configs.json").write_text(json.dumps(configs, indent=2) + "\n")
    print(f"Retained {len(configs)} configurations in {output}")


if __name__ == "__main__":
    main()
