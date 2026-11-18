#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 205 repro: st.async writes bypass the numerics shadow. Prints ptxv equiv's level for each
# st.async reference against its candidate and ptxv num's claims on r5_stasync_co_c1, next to
# GB300's recorded outputs (expected/), and exits 1 while GB300 refutes a claim. Simulator only;
# with PTXV_GPUS set it also replays every case on GB300 and compares with expected/ (exit 3 if
# GB300 differs).
#   PTXV=<ptxv binary> bash probes/stasync_num_shadow/run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
exec python3 "$here/check.py" "$(command -v "${PTXV:-ptxv}")"
