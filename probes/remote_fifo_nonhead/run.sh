#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# E-REMOTE-FIFO's delivery before a landing (deliver_fifo_before) and the remote pipe order
# (case.toml). For each variant and front: the stateful, DPOR and --certify verdicts, and the
# DPOR transitions that consumed an op behind its pipe's pending head (the PTXV_DPOR_DEBUG line
# "dpor nonhead transitions: N", summed). Exits 1 while any is taken, and when the binary does
# not print that line (it cannot show the count: ebfacb16 and older).
# The verdicts are printed, not compared: on SASS good DPOR also reaches the outcomes behind
# the race (E-DRF), which the stateful search does not.
# Usage: [PTXV=ptxv] probes/remote_fifo_nonhead/run.sh
set -u
here="$(cd "$(dirname "$0")" && pwd)"
ptxv="${PTXV:-ptxv}"
run() { # <variant> <front> [env...] -- [flags...]
  local v=$1 f=$2; shift 2
  local envs=()
  while [ $# -gt 0 ] && [ "$1" != -- ]; do envs+=("$1"); shift; done
  shift
  env "${envs[@]}" PTXV_THREADS="${PTXV_THREADS:-8}" systemd-run --user --scope -q \
    -p MemoryMax=16G -p MemorySwapMax=0 "$ptxv" check "$here" --variant "$v" --front "$f" \
    --no-gb300 --hw-profile none --no-input-coverage "$@" 2>&1
}
classes() { awk '/^(PASS|FAIL) /{for(i=1;i<=NF;i++) if($i ~ /^\[/){print $i; exit}}'; }
rc=0
for v in good NOST; do
  for f in ptx sass; do
    s=$(run $v $f -- | classes)
    out=$(run $v $f PTXV_EXPLORER=dpor PTXV_DPOR_DEBUG=1 --)
    d=$(echo "$out" | classes)
    nh=$(echo "$out" | awk '/^dpor nonhead transitions: /{n += $4; seen = 1} END {print seen ? n : "missing"}')
    c=$(run $v $f -- --certify | classes)
    [ "$nh" = 0 ] || rc=1
    printf '%-5s %-4s stateful %-34s dpor %-34s certify %-34s nonhead %s\n' \
      "$v" "$f" "$s" "$d" "$c" "$nh"
  done
done
exit $rc
