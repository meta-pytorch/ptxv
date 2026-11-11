# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""libcupti choice (ptxv._cupti.cupti_path): a CUPTI of the driver's major version first
(a lower-major CUPTI subscribes but delivers no launch callbacks), torch's copy before a wheel's
among equals."""

import sys

import pytest
from ptxv import _cupti


@pytest.fixture
def site(tmp_path, monkeypatch):
    with open("/proc/self/maps") as f:
        if "libcupti.so" in f.read():
            pytest.skip("a libcupti is already mapped in this process")
    s = tmp_path / "site"
    (s / "torch" / "lib").mkdir(parents=True)
    (s / "torch" / "__init__.py").write_text("")
    (s / "torch" / "lib" / "libcupti.so.12").write_bytes(b"")
    (s / "nvidia" / "cu13" / "lib").mkdir(parents=True)
    (s / "nvidia" / "cu13" / "lib" / "libcupti.so.13").write_bytes(b"")
    (s / "nvidia" / "cuda_cupti" / "lib").mkdir(parents=True)
    (s / "nvidia" / "cuda_cupti" / "lib" / "libcupti.so.12").write_bytes(b"")
    monkeypatch.delenv("PTXV_CUPTI", raising=False)
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "no_toolkit"))
    monkeypatch.setenv("PTXV_CUDA_HOME", str(tmp_path / "no_toolkit"))
    monkeypatch.setattr(sys, "path", [str(s)] + sys.path)
    monkeypatch.setattr(sys, "prefix", str(tmp_path / "prefix"))
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    return s


def test_driver_major_first(site, monkeypatch):
    monkeypatch.setattr(_cupti, "driver_major", lambda: 13)
    assert _cupti.cupti_path() == str(
        site / "nvidia" / "cu13" / "lib" / "libcupti.so.13"
    )


def test_torch_copy_among_equals(site, monkeypatch):
    monkeypatch.setattr(_cupti, "driver_major", lambda: 12)
    assert _cupti.cupti_path() == str(site / "torch" / "lib" / "libcupti.so.12")


def test_cupti_major():
    assert _cupti.cupti_major("/x/libcupti.so.13") == 13
    assert _cupti.cupti_major("/x/libcupti.so.2025.3.1") is None
