#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 207 repro: ptxv num gives PTX rcp.approx.ftz.f32 and ex2.approx.ftz.f32 results bound 0
# (r5_mufu out[24..27], out[88..91]). Prints every .ftz element whose claimed bound lies below
# |GB300 - f(x)| (GB300's recorded output, expected/) and exits 1 while there is one. Simulator
# only; with PTXV_GPUS set it also replays the kernel on GB300 and compares with expected/ (exit 3
# if GB300 differs).
#   PTXV=<ptxv binary> bash probes/mufu_ftz_bound/run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
exec python3 "$here/check.py" "$(command -v "${PTXV:-ptxv}")" 0,2
