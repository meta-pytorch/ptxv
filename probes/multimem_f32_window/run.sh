#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 225 repro: the simulator's multimem.ld_reduce f32 add against GB300's recorded lanes
# (cancelling sums where GB300's 16-aligned window keeps fewer member bits than the simulator's,
# random lanes where the two windows round differently, and all -0 members, which give +0 on
# GB300). Prints, per member count, test, source and front, how many lanes equal GB300's or
# differ, with one example lane per class, and exits 1 while a lane differs. Simulator only;
# with MM=<mm binary from probes/multimem/mm.cu> on a node with 4 GPUs it also replays the lanes
# on GB300 and compares with expected/ (exit 3 if GB300 differs).
#   PTXV=<ptxv binary> bash probes/multimem_f32_window/run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
exec python3 "$here/check.py" "$(command -v "${PTXV:-ptxv}")"
