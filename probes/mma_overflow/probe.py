# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the BSD-style license in the root LICENSE.

"""Capture MMA overflow and cancellation using the existing mma_zero kernels."""

from __future__ import annotations

import base64
import ctypes as C
import hashlib
import json
import shutil
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def blob(data: bytes) -> dict:
    return {
        "sha256": sha(data),
        "bytes": len(data),
        "zlib_base64": base64.b64encode(zlib.compress(data)).decode(),
    }


def cases() -> list[dict]:
    def p(e):
        return 2.0**e

    result = []
    for sign in (1, -1):
        for name, terms, c in [
            ("identity", [(1.0, 1.0)], 0.0),
            ("finite", [(p(126), 2.0)], 0.0),
            ("near_max", [(p(127), 2.0 - p(-7))], 0.0),
            ("overflow", [(p(127), 2.0)], 0.0),
            ("two_finite_overflow", [(p(127), 1.0)] * 2, 0.0),
            ("four_finite_overflow", [(p(126), 1.0)] * 4, 0.0),
            ("max_finite", [(p(127), 2.0), (-p(104), 1.0)], 0.0),
            ("half_ulp_below_overflow", [(p(127), 2.0), (-p(103), 1.0)], 0.0),
            ("quarter_ulp_below_overflow", [(p(127), 2.0), (-p(102), 1.0)], 0.0),
            ("cancel_overflow", [(p(127), 2.0), (-p(127), 2.0)], 0.0),
            ("rescue_overflow", [(p(127), 2.0), (-p(127), 1.0)], 0.0),
            ("huge_overflow", [(p(127), p(127))], 0.0),
            ("huge_cancel", [(p(127), p(127)), (-p(127), p(127))], 0.0),
            ("acc_overflow", [(p(127), 1.0)], p(127)),
            ("acc_rescue", [(p(127), 2.0)], -p(127)),
        ]:
            result.append(
                {
                    "name": name + ("_pos" if sign > 0 else "_neg"),
                    "terms": [(sign * a, b) for a, b in terms],
                    "c": sign * c,
                }
            )
    return result


def f32(v: float) -> int:
    return struct.unpack("<I", struct.pack("<f", v))[0]


def inputs(entry: str, enabled: bool) -> tuple[list[dict], dict[str, bytes]]:
    tc, tf = entry.startswith("tc"), entry.startswith("tm")
    rows, columns, k = (128, 16, 16) if tc else (16, 8, 8 if tf else 16)
    selected = [c for c in cases() if enabled or c["c"] == 0]
    data = {key: bytearray() for key in ("A", "B", "C")}
    for case in selected:
        a, b = zip(*(case["terms"] + [(0.0, 0.0)] * (k - len(case["terms"]))))

        def encode(v):
            return f32(v) if tf else f32(v) >> 16

        case["a"] = [encode(v) for v in a]
        case["b"] = [encode(v) for v in b]
        case["c_word"] = f32(case["c"])
        data["A"].extend(struct.pack(f"<{k}{'I' if tf else 'H'}", *case["a"]) * rows)
        data["B"].extend(struct.pack(f"<{k}{'I' if tf else 'H'}", *case["b"]) * columns)
        data["C"].extend(struct.pack("<I", case["c_word"]) * rows * columns)
    return selected, {key: bytes(value) for key, value in data.items()}


class Driver:
    def __init__(self):
        self.lib = C.CDLL("libcuda.so.1")
        self.call("cuInit", 0)
        self.device = C.c_int()
        self.call("cuDeviceGet", C.byref(self.device), 0)
        capability = []
        for attribute in (75, 76):
            v = C.c_int()
            self.call("cuDeviceGetAttribute", C.byref(v), attribute, self.device)
            capability.append(v.value)
        if capability != [10, 3]:
            raise RuntimeError(f"Require GB300 SM103, got {capability}")
        self.context = C.c_void_p()
        self.call("cuDevicePrimaryCtxRetain", C.byref(self.context), self.device)
        self.call("cuCtxSetCurrent", self.context)

    def call(self, name, *args):
        rc = getattr(self.lib, name)(*args)
        if rc:
            raise RuntimeError(f"{name}: CUDA error {rc}")

    def capture(self, cubin: Path, entry: str, enabled: bool) -> dict:
        selected, raw = inputs(entry, enabled)
        tc = entry.startswith("tc")
        cells = 2048 if tc else 128
        output_size = len(selected) * cells * 4
        module = C.c_void_p()
        self.call("cuModuleLoad", C.byref(module), str(cubin).encode())
        allocations = []
        try:
            for key in ("A", "B", "C", "D"):
                pointer = C.c_uint64()
                size = len(raw[key]) if key != "D" else output_size + 16
                self.call("cuMemAlloc_v2", C.byref(pointer), size)
                allocations.append(pointer)
                if key != "D":
                    self.call(
                        "cuMemcpyHtoD_v2",
                        pointer,
                        C.create_string_buffer(raw[key]),
                        size,
                    )
            function = C.c_void_p()
            self.call("cuModuleGetFunction", C.byref(function), module, entry.encode())
            arguments = allocations.copy()
            if tc:
                arguments += [
                    C.c_uint32((8 << 24) | (2 << 17) | 0x490),
                    C.c_uint32(enabled),
                ]
            params = (C.c_void_p * len(arguments))(
                *(C.cast(C.byref(arg), C.c_void_p) for arg in arguments)
            )
            repeats = []
            result = b""
            for fill in (0xA5, 0x5A, 0x3C):
                self.call("cuMemsetD8_v2", allocations[-1], fill, output_size + 16)
                self.call(
                    "cuLaunchKernel",
                    function,
                    len(selected),
                    1,
                    1,
                    128 if tc else 32,
                    1,
                    1,
                    0,
                    None,
                    params,
                    None,
                )
                self.call("cuCtxSynchronize")
                host = C.create_string_buffer(output_size + 16)
                self.call("cuMemcpyDtoH_v2", host, allocations[-1], output_size + 16)
                if host.raw[-16:] != bytes([fill]) * 16:
                    raise AssertionError("Output guard overwritten")
                result = host.raw[:-16]
                repeats.append(sha(result))
            if len(set(repeats)) != 1:
                raise AssertionError("MMA output differs between repeated launches")
            words = struct.unpack(f"<{len(selected) * cells}I", result)
            for i, case in enumerate(selected):
                values = set(words[i * cells : (i + 1) * cells])
                if len(values) != 1:
                    raise AssertionError(
                        f"Nonuniform tile for {entry} {case['name']}: {values}"
                    )
                case["output"] = values.pop()
                if case["name"].startswith("identity"):
                    expected = f32(1.0 if case["name"].endswith("pos") else -1.0)
                    if case["output"] != expected:
                        raise AssertionError("Identity control failed")
            return {
                "entry": entry,
                "enabled": enabled,
                "cases": selected,
                "grid": len(selected),
                "block": 128 if tc else 32,
                "idesc": (8 << 24) | (2 << 17) | 0x490 if tc else None,
                "inputs": {key: blob(value) for key, value in raw.items()},
                "output": blob(result),
                "repeat_hashes": repeats,
            }
        finally:
            for pointer in allocations:
                self.call("cuMemFree_v2", pointer)
            self.call("cuModuleUnload", module)


def capture(ptx: Path) -> dict:
    ptxas = shutil.which("ptxas")
    if ptxas is None:
        raise FileNotFoundError("ptxas must be on PATH")
    with tempfile.TemporaryDirectory(prefix="mma-overflow-") as directory:
        cubin = Path(directory) / "kernel.cubin"
        version = subprocess.check_output([ptxas, "--version"], text=True)
        subprocess.run(
            [ptxas, "-arch=sm_103a", "-O3", str(ptx), "-o", str(cubin)], check=True
        )
        driver = Driver()
        driver_version = C.c_int()
        driver.call("cuDriverGetVersion", C.byref(driver_version))
        uuid = C.create_string_buffer(16)
        driver.call("cuDeviceGetUuid", uuid, driver.device)
        name = C.create_string_buffer(128)
        driver.call("cuDeviceGetName", name, len(name), driver.device)
        nvml = C.CDLL("libnvidia-ml.so.1")
        if nvml.nvmlInit_v2() != 0:
            raise RuntimeError("NVML initialization failed")
        try:
            release = C.create_string_buffer(128)
            if nvml.nvmlSystemGetDriverVersion(release, len(release)) != 0:
                raise RuntimeError("NVML driver-version query failed")
        finally:
            if nvml.nvmlShutdown() != 0:
                raise RuntimeError("NVML shutdown failed")
        configs = [
            driver.capture(cubin, entry, enabled)
            for entry, enabled in [
                ("tc_f16", False),
                ("tc_f16", True),
                ("hm0", False),
                ("hm", True),
                ("tm0", False),
                ("tm", True),
            ]
        ]
        return {
            "ptx_sha256": sha(ptx.read_bytes()),
            "probe_sha256": sha(Path(__file__).read_bytes()),
            "cubin": blob(cubin.read_bytes()),
            "ptxas_version": version,
            "cuda_driver_api_version": driver_version.value,
            "device_uuid": uuid.raw.hex(),
            "driver_release": release.value.decode(),
            "device_name": name.value.decode(),
            "compute_capability": [10, 3],
            "configs": configs,
        }


if __name__ == "__main__":
    print(
        json.dumps(
            capture(Path(__file__).resolve().parent.parent / "mma_zero/kernel.ptx")
        )
    )
