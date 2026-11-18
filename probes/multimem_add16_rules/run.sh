#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 208 repro: the simulator's multimem.ld_reduce f16 / bf16 add against GB300's recorded lanes
# (bf16 sums below GB300's 16-aligned window kept where GB300 gives +0; all -0 members give -0
# where GB300 gives +0).
# Prints, per member count, test and front, how many lanes equal GB300's, are its other rounding,
# or differ, with one example lane per class, and exits 1 while a lane differs. Simulator only;
# with MM=<mm binary from probes/multimem/mm.cu> on a node with 4 GPUs it also replays the vectors
# on GB300 and compares with expected/ (exit 3 if GB300 differs).
#   PTXV=<ptxv binary> bash probes/multimem_add16_rules/run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
exec python3 "$here/check.py" "$(command -v "${PTXV:-ptxv}")"
