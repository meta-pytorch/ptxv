#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 201 repro: (a * b) * c against a * (b * c) where a * b rounds into the subnormal range.
# Prints GB300's recorded outputs next to ptxv equiv's claims; exits 1 while a claim is
# contradicted (../README.md). On a GPU node every case is first rerun HW_RUNS times (default 5;
# HW_RUNS=0 skips it).   PTXV=<ptxv binary> bash run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
runs=${HW_RUNS:-5}
nvidia-smi -L 2>/dev/null | grep -q GPU || runs=0
exec "${PTXV_PYTHON:-python3}" -I "$here/../check.py" "$here" --ptxv "${PTXV:-ptxv}" --hw "$runs"
