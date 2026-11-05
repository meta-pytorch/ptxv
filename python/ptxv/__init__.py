# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""ptxv from Python: capture kernels, check them, prove equivalence, estimate error
distributions. See docs/python.md.

    import ptxv
    with ptxv.capture() as cap:
        y = torch.softmax(x, -1)
    k = cap.kernels[0]
    k.check()                                  # Verdict
    eq = ptxv.compare(ref, k)                  # Equivalence; eq.bound(x=x) is tau(x)

`import ptxv` enables CUPTI's module-load callback (cubins of kernels launched later), so
import it before the program's first CUDA work. PTXV_NO_IMPORT_HOOK=1 defers that to the
first capture.
"""

from __future__ import annotations

import os as _os

from . import dist
from ._jobs import Job, JobCancelled, LocalBackend, PtxvError
from .api import capture_call, check, compare, errdist, errdist_model, ErrdistModel
from .bound import Bound
from .capture import Capture, capture, Kernel, KernelList, Launch, Launches
from .case import Case
from .results import Equivalence, ErrDist, Finding, Verdict
from .session import Session, session

__version__ = "0.1.0"

__all__ = [
    "Bound",
    "Capture",
    "Case",
    "Equivalence",
    "ErrDist",
    "Finding",
    "Job",
    "JobCancelled",
    "Kernel",
    "KernelList",
    "Launch",
    "Launches",
    "LocalBackend",
    "PtxvError",
    "Session",
    "Verdict",
    "capture",
    "capture_call",
    "check",
    "compare",
    "dist",
    "errdist",
    "errdist_model",
    "ErrdistModel",
    "session",
]


def __getattr__(name: str):
    if name == "DaemonBackend":
        from .daemon import DaemonBackend

        return DaemonBackend
    if name == "SlurmBackend":
        from .slurm import SlurmBackend

        return SlurmBackend
    raise AttributeError(name)


def _import_hook() -> None:
    if _os.environ.get("PTXV_NO_IMPORT_HOOK"):
        return
    try:
        from . import _cupti

        _cupti.init()
    except Exception:  # capture reports the reason when it is used
        pass


_import_hook()
