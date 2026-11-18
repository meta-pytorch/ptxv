#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# RDP2 peek-distance check: classes and distinct outputs of the unreduced search, the default
# explorer and DPOR for each case dir. MISS when DPOR / default reach fewer outputs or classes
# than the unreduced search on a race-free kernel, or lack a class.
ptxv=${PTXV:?}; F=${FRONT:-ptx}
off=(PTXV_NO_TMA_POR=1 PTXV_NO_OWN_ASYNC=1 PTXV_NO_SYNC_POR=1 PTXV_NO_LOCAL_POR=1 PTXV_NO_CTA_POR=1
     PTXV_NO_BSYNC_POR=1 PTXV_NO_RELEASED=1 PTXV_NO_GATE=1 PTXV_NO_VALTRACK=1 PTXV_NO_PHASE_CLIP=1
     PTXV_NO_DEAD_SYNC=1 PTXV_NO_PEEKSKIP=1 PTXV_NO_PEEKWAIT=1 PTXV_NO_VIP=1)
r() { local d=$1; shift; timeout ${TO:-120} env "$@" PTXV_THREADS=4 "$ptxv" check "$d" --variant $V --front $F --no-gb300 --hw-profile none --no-input-coverage 2>&1 | awk '/^(PASS|FAIL) /{c="";o=""; for(i=1;i<=NF;i++){if($i ~ /^\[/ && c=="") c=$i; if($i ~ /^outputs=/) o=$i}; print c"/"o; exit}'; }
for d in "$@"; do
  u=$(r $d "${off[@]}"); s=$(r $d X=1); p=$(r $d PTXV_EXPLORER=dpor ${DPOR_ENV:-X=1})
  tag=ok; for x in "$s" "$p"; do [ "$x" = "$u" ] || tag=DIFF; done
  printf '%-10s unreduced %-36s default %-36s dpor %-36s %s\n' $(basename $d) "$u" "$s" "$p" "$tag"
done
