# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# shellcheck shell=bash
# Source this to set up the ptxv environment:  source env.sh
# Needs a Rust toolchain (cargo) on PATH and the CUDA 13 toolkit (nvdisasm, cuobjdump).
# Overrides, all optional: PTXV_CUDA_HOME (default /usr/local/cuda-13.0), PTXV_PYTHON (Python
# for helper scripts and plots, never used by the verifier itself; default python3),
# CARGO_HOME / CARGO_TARGET_DIR (Cargo's own defaults unless set; CARGO_TARGET_DIR defaults
# to $PTXV_ROOT/target). A site-local fb/env.local.sh is sourced first when present.
PTXV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PTXV_ROOT
if [ -f "$PTXV_ROOT/fb/env.local.sh" ]; then
  # shellcheck source=/dev/null
  . "$PTXV_ROOT/fb/env.local.sh"
fi
export PTXV_CUDA_HOME="${PTXV_CUDA_HOME:-/usr/local/cuda-13.0}"
case ":$PATH:" in
  *":$PTXV_CUDA_HOME/bin:"*) ;;
  *) export PATH="$PTXV_CUDA_HOME/bin:$PATH" ;;
esac
export PTXV_PYTHON="${PTXV_PYTHON:-python3}"
export CARGO_TARGET_DIR="${CARGO_TARGET_DIR:-$PTXV_ROOT/target}"
# GPU policy: ptxv only uses GPUs with no compute processes it did not start (checked per run).
# PTXV_EXCLUDE_GPUS=<i,j> additionally makes the test harness and helper scripts refuse GPUs.
