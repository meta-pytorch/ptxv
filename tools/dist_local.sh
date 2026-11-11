#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Run `ptxv <args>` as N ranks on this machine (multi-node code path, for testing).
#   tools/dist_local.sh N ptxv-args...
# Rank 0's stdout is printed; every rank's stderr goes to $DIR/err.<rank>.
set -e
N=$1; shift
DIR=$(mktemp -d /tmp/ptxv-dist.XXXX)
BIN=${PTXV_BIN:-$CARGO_TARGET_DIR/release/ptxv}
pids=()
for ((r=1; r<N; r++)); do
  PTXV_DIST_DIR=$DIR PTXV_RANK=$r PTXV_RANKS=$N "$BIN" "$@" 2>"$DIR/err.$r" >/dev/null &
  pids+=($!)
done
PTXV_DIST_DIR=$DIR PTXV_RANK=0 PTXV_RANKS=$N "$BIN" "$@" 2>"$DIR/err.0"
st=$?
for p in "${pids[@]}"; do wait "$p" || st=$?; done
echo "(logs: $DIR)" >&2
exit $st
