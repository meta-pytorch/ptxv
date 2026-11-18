#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 209 repro: st.async in a launch without cluster dimensions traps on GB300 ("an illegal
# instruction was encountered"); ptxv check and --certify run it without a fault. Prints both
# verdicts on both fronts next to GB300's recorded result (expected/), the explicit 1-CTA cluster
# launches as controls, and exits 1 while a trapping case gets no fault. Simulator only; with
# PTXV_GPUS set it also runs the cases on GB300 (exit 3 if GB300's result changed).
#   PTXV=<ptxv binary> bash probes/stasync_no_cluster/run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
exec python3 "$here/check.py" "$(command -v "${PTXV:-ptxv}")"
