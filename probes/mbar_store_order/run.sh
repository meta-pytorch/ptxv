#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 241 / MISS 242 repro (fixtures/e_mbar_store_order): a plain store over an mbarrier's
# memory, or a re-initialization, unordered with the inval or the first init of the object.
# Either order can happen, so the verdict must not depend on the order the explorer or the
# certificate runs them in. good / SWAP / PREU: violation (MISS 241: a store over a live
# object in one order); REINIT: race (MISS 241, SASS STS.64 re-init vs inval); REINITU: race
# (MISS 242, SYNCS.EXCH.64 vs SYNCS.CCTL.IV); ORD (store after a bar.sync after the inval):
# clean. Runs every variant on both fronts, exhaustive and --certify, prints expected next to
# observed and exits 1 on any difference or missing verdict. Simulator only (no GPU).
#   PTXV=<ptxv binary> probes/mbar_store_order/run.sh
here=$(cd "$(dirname "$0")" && pwd)
CASE=$here/../../fixtures/e_mbar_store_order exec "$here/../mbar_sts_live/run.sh"
