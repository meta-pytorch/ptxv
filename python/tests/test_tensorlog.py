# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Tensor metadata of CUDA-injection captures (ptxv._tensorlog, python/ptxv/_site): the tensors
of each CuTeDSL call are logged with their names, dtypes and shapes, and the staging side gives
a launch the tensors of the call that issued it, so a captured bf16 kernel's buffers are bf16.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
from conftest import GPU

PY = Path(__file__).resolve().parents[1]  # python/
SITE = PY / "ptxv" / "_site"

# A stand-in CuTeDSL host path: from_dlpack wraps a tensor, BaseDSL._func runs a @cute.jit
# function's body (as cutlass.base_dsl.dsl does).
FAKE_RUNTIME = """
class _CuteTensor:
    def __init__(self, t):
        self.t = t

def from_dlpack(t, assumed_align=None):
    return _CuteTensor(t)
"""
FAKE_DSL = """
class BaseDSL:
    def _func(self, funcBody, *args, **kwargs):
        return funcBody(*args, **kwargs)
"""
# A CUDA-looking tensor (what tensor_info reads).
FAKE_TORCH_TENSOR = """
class _Storage:
    def __init__(self, p, n):
        self.p, self.n = p, n
    def data_ptr(self):
        return self.p
    def nbytes(self):
        return self.n

class T:
    is_cuda = True
    def __init__(self, p, dtype, shape, esize):
        self.p, self.dtype, self.shape, self.es = p, dtype, shape, esize
    def untyped_storage(self):
        n = self.es
        for s in self.shape:
            n *= s
        return _Storage(self.p, n)
    def element_size(self):
        return self.es
    def is_contiguous(self):
        return True
    def storage_offset(self):
        return 0
    def numel(self):
        n = 1
        for s in self.shape:
            n *= s
        return n
    def dim(self):
        return len(self.shape)
    def data_ptr(self):
        return self.p
"""


def fake_cutlass(tmp: Path) -> Path:
    root = tmp / "fake"
    for d in ("cutlass", "cutlass/cute", "cutlass/base_dsl"):
        (root / d).mkdir(parents=True, exist_ok=True)
        (root / d / "__init__.py").write_text("")
    (root / "cutlass/cute/runtime.py").write_text(FAKE_RUNTIME)
    (root / "cutlass/base_dsl/dsl.py").write_text(FAKE_DSL)
    (root / "faketorch.py").write_text(FAKE_TORCH_TENSOR)
    return root


def test_cute_calls_log_their_tensors(tmp_path):
    fake = fake_cutlass(tmp_path)
    cap = tmp_path / "cap"
    code = textwrap.dedent(
        f"""
        import sys; sys.path.insert(0, {str(fake)!r})
        from faketorch import T
        from cutlass.cute.runtime import from_dlpack
        from cutlass.base_dsl.dsl import BaseDSL

        class bf16:
            def __str__(self):
                return "torch.bfloat16"

        def launch(a, b, n):
            return n

        x = T(0x10000, bf16(), [4, 64], 2)
        out = T(0x20000, bf16(), [4, 64], 2)
        BaseDSL()._func(launch, from_dlpack(x), from_dlpack(out), 256)
        """
    )
    env = dict(
        os.environ, PYTHONPATH=str(SITE), PTXV_CAPTURE_DIR=str(cap / "%r"), RANK="0"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    calls = [json.loads(ln) for ln in open(cap / "0" / "tensors.jsonl")]
    assert len(calls) == 1 and calls[0]["call"] == "launch", calls
    ts = {t["name"]: t for t in calls[0]["tensors"]}
    assert set(ts) == {"a", "b"}, ts
    assert ts["a"]["dtype"] == "bf16" and ts["a"]["shape"] == [4, 64], ts
    assert (
        ts["b"]["ptr"] == 0x20000 and ts["b"]["nbytes"] == 512 and ts["b"]["whole"]
    ), ts


def test_a_launch_takes_the_tensors_of_the_call_before_it():
    sys.path.insert(0, str(PY))
    from ptxv import _tensorlog

    calls = [
        {"t_us": 10, "tensors": [{"name": "a", "dtype": "bf16"}]},
        {
            "t_us": 20,
            "tensors": [{"name": "a", "dtype": "f32"}, {"name": "a", "dtype": "f16"}],
        },
    ]
    assert _tensorlog.tensors_at(calls, 5) == {}
    assert _tensorlog.tensors_at(calls, 15)["a"]["dtype"] == "bf16"
    got = _tensorlog.tensors_at(calls, 25)
    assert got["a"]["dtype"] == "f32" and got["a_2"]["dtype"] == "f16", got


@pytest.mark.skipif(not GPU, reason="set PTXV_TEST_GPU=<gpu index>")
def test_bf16_cute_kernel_logs_bf16_tensors(tmp_path):
    # A real bf16 CuTeDSL kernel run without `import ptxv` (as under tools/ptxvcap-run): the
    # log names its tensors with dtype bf16 and their shapes.
    pytest.importorskip("cutlass")
    cap = tmp_path / "cap"
    code = textwrap.dedent(
        """
        import torch
        import cutlass
        import cutlass.cute as cute
        from cutlass.cute.runtime import from_dlpack

        @cute.kernel
        def scale(x: cute.Tensor, out: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            out[tidx] = (x[tidx].to(cutlass.Float32) * cutlass.Float32(2.0)).to(cutlass.BFloat16)

        @cute.jit
        def launch(x: cute.Tensor, out: cute.Tensor):
            scale(x, out).launch(grid=[1, 1, 1], block=[64, 1, 1])

        x = torch.randn(64, dtype=torch.bfloat16, device="cuda")
        out = torch.zeros_like(x)
        launch(from_dlpack(x), from_dlpack(out))
        torch.cuda.synchronize()
        assert torch.equal(out, (x.float() * 2).bfloat16())
        """
    )
    # The DSL reads the kernel's source: a file, not `python -c`.
    script = tmp_path / "bf16_scale.py"
    script.write_text(code)
    env = dict(
        os.environ,
        PYTHONPATH=str(SITE),
        PTXV_CAPTURE_DIR=str(cap),
        CUDA_VISIBLE_DEVICES=GPU,
    )
    r = subprocess.run(
        [sys.executable, str(script)], env=env, capture_output=True, text=True
    )
    assert r.returncode == 0, r.stderr
    calls = [json.loads(ln) for ln in open(cap / "tensors.jsonl")]
    ts = {t["name"]: t for c in calls for t in c["tensors"]}
    assert ts["x"]["dtype"] == "bf16" and ts["out"]["dtype"] == "bf16", ts
    assert ts["x"]["shape"] == [64] and ts["x"]["whole"], ts
