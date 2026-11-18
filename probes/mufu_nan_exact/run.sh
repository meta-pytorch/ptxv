#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 258 repro (N-NUM-NONFINITE): a MUFU / .approx node whose true function has no real value at
# its operand (sqrt, rsqrt, lg2 of a negative number; here negative subnormals the .ftz forms flush
# to -0) gets the exact value 0 (num.rs num_mufu: `fin` maps a non-finite true value to the empty
# expansion), so ptxv num analyzes the element against a made-up exact value: sqrt.approx.ftz of
# -3 2^-149 (GB300: -0) is reported abs_err 0, untracked=false, nonfinite=false; ex2(lg2(.)) of it
# (GB300: 0) is reported exact 1, abs_err 1. ASSUMPTIONS.md N-NUM-NONFINITE: an element whose exact
# value is not finite is not analyzed (nonfinite, error fields NaN). Exits 1 while such an element
# is analyzed. Simulator only; with PTXV_GPUS set it also replays both cases on GB300
# (ptxv hw --runs 20 --scrub; exit 3 unless bitwise equal to the simulator).
#   PTXV=<ptxv binary> bash probes/mufu_nan_exact/run.sh   (2: tool failure)
set -u
here=$(cd "$(dirname "$0")" && pwd)
exec python3 "$here/check.py" "$(command -v "${PTXV:-ptxv}")"
