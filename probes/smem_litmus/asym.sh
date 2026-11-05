#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Pure-shared MP/SB/MPS with only one location bank-conflicted (32-way replay on the STS/LDS of
# that location, the other conflict-free), plus the shared-flag/global-payload shape. GPU from $1.
cd "$(dirname "$0")" || exit
export CUDA_VISIBLE_DEVICES=$1
for t in "MP shv cta 0 0" "MPS shv cta 0 0" "SB shv cta 0 0" "LB shv cta 0 0" "IRIW shv cta 0 0" "MP shv cta 0 2" "MP shv cta 2 0"; do
  for lay in "is=1 xis=32 lsep=2048" "is=32 xis=1 lsep=2048"; do
    for st in "nw=32 dmask=63 smask=255" "nw=8 dmask=127 smask=0" "nw=16 dmask=0 smask=63"; do
      # shellcheck disable=SC2086  # each of t/lay/st is several litmus arguments
      timeout 600 ./litmus $t $lay $st iters=500000 seed=$RANDOM
    done
  done
done
