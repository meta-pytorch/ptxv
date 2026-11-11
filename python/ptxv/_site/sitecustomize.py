# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Startup hook for processes captured without `import ptxv` (CUDA-injection captures,
tools/ptxvcap-run puts this directory first on PYTHONPATH).

With CuTeDSL keeping PTX (CUTE_DSL_KEEP=ptx,... or CUTE_DSL_KEEP_PTX=1), every compilation's kept
files get a unique name and a collision raises (ptxv._ptxsrc.install_cute_unique_dumps). In a
CUDA-injection capture (PTXV_CAPTURE_DIR), the tensors passed to each CuTeDSL / Triton call are
logged (ptxv._tensorlog). The modules are loaded from their files under private names, so the
ptxv package (its CUPTI import hook) is not imported into the captured process. Then the next
sitecustomize on sys.path, if any, runs as it would have without this one.
"""

import importlib.util
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))


def _keep_ptx() -> bool:
    keep = os.environ.get("CUTE_DSL_KEEP", "").split(",")
    return "ptx" in keep or os.environ.get("CUTE_DSL_KEEP_PTX", "0") not in ("0", "")


def _load(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_ptxv_site_{name}", os.path.join(os.path.dirname(_here), f"{name}.py")
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


if _keep_ptx():
    _load("_ptxsrc").install_cute_unique_dumps()
# CUDA-injection capture (ptxvcap.c writes PTXV_CAPTURE_DIR): the tensors of each CuTeDSL /
# Triton call, so staged cases get their buffers' dtypes and shapes (ptxv._tensorlog).
if os.environ.get("PTXV_CAPTURE_DIR") and os.environ.get("PTXV_TENSOR_LOG", "1") != "0":
    _load("_tensorlog").install(os.environ["PTXV_CAPTURE_DIR"])


def _chain() -> None:
    """Run the sitecustomize this one shadows (the next one on sys.path)."""
    for d in sys.path:
        if not d or os.path.abspath(d) == _here:
            continue
        p = os.path.join(d, "sitecustomize.py")
        if os.path.isfile(p):
            spec = importlib.util.spec_from_file_location(
                "_ptxv_chained_sitecustomize", p
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return


_chain()
