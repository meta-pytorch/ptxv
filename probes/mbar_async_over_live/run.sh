#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 256 repro (fixtures/e_mbar_async_live): a write over a live mbarrier by an atomic
# (ATOM, RED), cp.async (CPASYNC) or a TMA bulk copy (BULK) must be the contract violation MISS
# 218 reports for a plain store (PLAIN, control); NEAR (an atomic next to the mbarrier) is clean.
# Runs every variant on both fronts, exhaustive and --certify, prints expected next to observed
# and exits 1 on any difference or missing verdict. Simulator only (no GPU).
#   PTXV=<ptxv binary> probes/mbar_async_over_live/run.sh
here=$(cd "$(dirname "$0")" && pwd)
CASE=$here/../../fixtures/e_mbar_async_live exec "$here/../mbar_sts_live/run.sh"
