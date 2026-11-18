#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 257 repro (MISS 224 follow-up): a NIC atomic that leaves a word of its 8-byte target
# unchanged (a compare-and-swap that does not swap, a fetch-add of 0, the high word of a fetch-add
# of 1) must not turn the rounded value the word still holds into an exact constant, and a word it
# changes that held a rounding must not be claimed exact (a GPU integer atomic makes such a word
# input data: num_atom_rounding). cas_fail/, fa0/, fa1/: dst[0..2) hold rn(0.1f + 0.2f) and
# rn(1 + 1e-8f) (errors 7.45e-9 and 1.0e-8), then the NIC atomic on that word over a loopback
# queue pair, and the word is copied to out. Exits 1 while a claim contradicts the exact error.
# Simulator only; with PTXV_GPUS set it also replays arith/ (the payload) on GB300 and compares
# with expected/ (exit 3 if GB300 changed).
#   PTXV=<ptxv binary> bash probes/nic_atomic_keeps_word/run.sh   (2: tool failure)
set -u
here=$(cd "$(dirname "$0")" && pwd)
exec python3 "$here/check.py" "$(command -v "${PTXV:-ptxv}")"
