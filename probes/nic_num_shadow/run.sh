#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 224 repro: `ptxv num` of an output an RDMA write produced (loop/: a loopback IBGDA queue
# pair writes src[i] = x[i] + y[i] over dst, which held a copy of an input with the same bits)
# next to GB300's error (arith/: the payload alone, measured). Exits 1 while a claim is below
# GB300's error. Simulator only; with PTXV_GPUS set it also replays arith/ on GB300.
#   PTXV=<ptxv binary> bash probes/nic_num_shadow/run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
exec python3 "$here/check.py" "$(command -v "${PTXV:-ptxv}")"
