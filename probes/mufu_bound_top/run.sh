#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 206 repro: ptxv num's MUFU bound overflows at the 16 largest finite magnitudes, reached by
# PTX rcp.approx.f32 (no .ftz) of subnormal operands (r5_mufu out[32..47]). Prints every
# rcp.approx.f32 element whose claimed bound lies below |GB300 - 1/x| (GB300's recorded output,
# expected/) and exits 1 while there is one. Simulator only; with PTXV_GPUS set it also replays
# the kernel on GB300 and compares with expected/ (exit 3 if GB300 differs).
#   PTXV=<ptxv binary> bash probes/mufu_bound_top/run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
exec python3 "$here/check.py" "$(command -v "${PTXV:-ptxv}")" 1
