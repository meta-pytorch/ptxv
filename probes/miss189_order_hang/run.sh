#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# MISS 189 repro: wait_flag spins on a flag that set_flag, the next launch of the list, sets;
# the launch order is unknown. GB300 (`ptxv hw`, which runs the list's order on one stream)
# never starts set_flag and hangs; `ptxv check` under the strict reading of the unknown order
# lets the grids overlap. Prints GB300's result for each variant (CONCURRENT, the grids on two
# streams, is the control that finishes) next to ptxv's verdict for variant good on both
# fronts, and exits 1 while GB300 hangs on good and ptxv does not report a deadlock there.
#   PTXV_BIN=<ptxv> PTXV_GPUS=<idle GPU> bash probes/miss189_order_hang/run.sh
set -u
here=$(cd "$(dirname "$0")" && pwd)
bin=${PTXV_BIN:-ptxv}
t=$(mktemp -d)
cd "$t"  # ptxv writes ./out/<case>/
"$bin" hw "$here" --variant all --no-num --runs 3 --timeout 20 > hw.log 2>&1
classes() {
  python3 -c '
import json, sys
d = None
for ln in open(sys.argv[1]):
    try:
        d = json.loads(ln)
    except ValueError:
        pass
print(",".join(((d or {}).get("rows") or [{}])[0].get("got", [])))
' "$1"
}
for front in ptx sass; do
  env -u PTXV_UNRECORDED_ORDER PTXV_NO_VCACHE=1 "$bin" check "$here" --variant good \
    --front $front --json > "check_$front.json" 2> "check_$front.log"
done
got_ptx=$(classes check_ptx.json)
got_sass=$(classes check_sass.json)
for v in good DECLARED CONCURRENT; do
  row=$(grep -m1 "^hw $v .*distinct outputs" hw.log || echo "hw $v: no result ($(tail -n 1 hw.log))")
  echo "GB300 (ptxv hw): $row"
done
echo "ptxv check good (strict reading of the unknown order): PTX [$got_ptx], SASS [$got_sass]"
echo "expected: deadlock on both fronts where GB300 hangs on good"
echo "logs: $t"
bad=0
case "$(grep -m1 "^hw good .*distinct outputs" hw.log)" in
  *timeout*)
    for g in "$got_ptx" "$got_sass"; do
      [[ ",$g," == *",deadlock,"* ]] || bad=1
    done
    ;;
esac
exit $bad
