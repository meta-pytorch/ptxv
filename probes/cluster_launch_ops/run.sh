#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 209 calibration: `ptxv check` of every case and variant the simulator lifts next to GB300's
# result (expected/<case>.gb300.txt: the variant traps, or runs and gives
# expected/<case>.<variant>.out.bin). Exits 1 while a verdict differs from GB300's (no fault where
# GB300 traps, or not clean where it runs), 2 on a tool failure. The outputs are replayed by
# `cargo test -p ptxv-sim --test fixtures probe_cluster_launch_ops_match_hardware`; `ptxv probe
# probes/cluster_launch_ops/<case>` re-measures a case on GB300. Simulator only.
#   PTXV=<ptxv binary> bash probes/cluster_launch_ops/run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
exec python3 "$here/check.py" "$(command -v "${PTXV:-ptxv}")"
