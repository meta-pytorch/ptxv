#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 190 repro: the verdict cache key (crates/cli/src/vcache.rs) ignores CUDA_HOME, which
# selects the ptxas and nvdisasm that decide the lifted SASS (crates/sass `tool`). Two
# `ptxv check --certify` runs of fixtures/c_cat's SASS front under two CUDA_HOME directories
# (the second a directory of links to the first's tools) store their verdicts in a fresh cache;
# a later errdist / equiv precondition reads the stored verdict by the same key. Expected: two
# keys (a verdict computed with one toolkit does not serve another); observed: the number of
# stored verdicts. Exits 1 while both runs share one key.
#   PTXV_BIN=<ptxv binary> bash probes/miss190_vcache_cuda_home/run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
fx=$here/../../fixtures/c_cat
bin=${PTXV_BIN:-ptxv}
cuda=${CUDA_HOME:-/usr/local/cuda-13.0}
t=$(mktemp -d)
mkdir -p "$t/cuda/bin"
for f in "$cuda"/bin/*; do ln -s "$f" "$t/cuda/bin/"; done
export PTXV_CACHE=$t/cache
cd "$t"  # ptxv writes ./out/<case>/
unset PTXV_NO_VCACHE
CUDA_HOME=$cuda "$bin" check "$fx" --variant good --front sass --certify > "$t/a.log" 2>&1 || { echo "first run failed: $t/a.log"; exit 2; }
CUDA_HOME=$t/cuda "$bin" check "$fx" --variant good --front sass --certify > "$t/b.log" 2>&1 || { echo "second run failed: $t/b.log"; exit 2; }
n=$(find "$t/cache/check" -name '*.json' | wc -l)
echo "CUDA_HOME $cuda, then $t/cuda (links to the same tools)"
echo "expected: 2 stored verdicts (the key covers the toolkit); observed: $n"
[ "$n" -ge 2 ] && exit 0
exit 1
