#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Rung 4: capture PyTorch's own softmax / layer_norm / rms_norm / sum / flash SDPA kernels as
# binary-only cases fixtures/r4_<op>_torch (tools/capture.py), then shrink each module cubin
# to the launched kernels (tools/cubin_slim.py: 2-11 MB -> 0.2-1.6 MB, loads and runs alike).
#   CUDA_VISIBLE_DEVICES=0 tools/r4_torch_capture.sh
set -e
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091  # env.sh at the repo root (cwd); not followed
source env.sh
cap() {  # name, capture.py arguments...
    local d=fixtures/$1; shift
    rm -rf "$d"
    $PTXV_PYTHON tools/capture.py "$@" --out "$d"
    for c in "$d"/k*.cubin; do
        # shellcheck disable=SC2046  # one argument per kernel entry name
        python3 tools/cubin_slim.py "$c" "$c.slim" $(grep -E '^entry' "$d/case.toml" | cut -d'"' -f2)
        mv "$c.slim" "$c"
    done
}
cap r4_softmax_torch --op 'torch.softmax(x, -1)' --inputs 'x=randn(4,128,f32)'
cap r4_layernorm_torch --op 'F.layer_norm(x, (128,), w, b, 1e-5)' \
    --inputs 'x=randn(4,128,bf16),w=randn(128,bf16),b=randn(128,bf16)'
cap r4_rmsnorm_torch --op 'F.rms_norm(x, (128,), w, 1e-6)' --inputs 'x=randn(4,128,bf16),w=randn(128,bf16)'
cap r4_sum_torch --op 'torch.sum(x, -1)' --inputs 'x=randn(4,128,f32)'
cap r4_sdpa_flash_torch --file tools/capture_examples/sdpa_flash.py
