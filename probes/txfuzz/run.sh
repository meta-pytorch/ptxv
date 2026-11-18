#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# mbarrier tx fuzz (MISS 273): for each case dir, classes of the unreduced stateful search, the default
# explorer and DPOR on one front. Prints a line per case; MISS where an explorer lacks a class
# of the unreduced search (or reports clean where it does not).
# Usage: PTXV=<ptxv> FRONT=ptx run.sh <case dirs...>
ptxv=${PTXV:?}; F=${FRONT:-ptx}
off=(PTXV_NO_TMA_POR=1 PTXV_NO_OWN_ASYNC=1 PTXV_NO_SYNC_POR=1 PTXV_NO_LOCAL_POR=1 PTXV_NO_CTA_POR=1
     PTXV_NO_BSYNC_POR=1 PTXV_NO_RELEASED=1 PTXV_NO_GATE=1 PTXV_NO_VALTRACK=1 PTXV_NO_PHASE_CLIP=1
     PTXV_NO_DEAD_SYNC=1 PTXV_NO_PEEKSKIP=1 PTXV_NO_PEEKWAIT=1 PTXV_NO_VIP=1)
cls() { local d=$1; shift; timeout ${TO:-40} env "$@" PTXV_THREADS=1 "$ptxv" check "$d" --variant good --front $F --no-gb300 --hw-profile none --no-input-coverage --max-states ${MAXS:-150000} 2>&1 |
  awk '/^(PASS|FAIL) /{c="";inc=""; for(i=1;i<=NF;i++){if($i ~ /^\[/ && c=="") c=$i; if($i ~ /INCOMPLETE/) inc="!INC"}; print c inc; exit}'; }
lacks() { # does $2 lack a class of $1
  local u=$1 x=$2 c
  [[ "$u" == *INC* || "$x" == *INC* || -z "$u" || -z "$x" ]] && return 1
  for c in $(echo "$u" | tr -d '[]' | tr ',' ' '); do [ "$c" = nondeterministic ] && [[ "$u" == *race* ]] && continue; [[ "$x" == *"$c"* ]] || return 0; done
  return 1; }
for d in "$@"; do
  u=$(cls $d "${off[@]}"); s=$(cls $d X=1); p=$(cls $d PTXV_EXPLORER=dpor)
  tag=ok; lacks "$u" "$s" && tag="MISS-default"; lacks "$u" "$p" && tag="$tag MISS-dpor"
  printf '%s unreduced %s default %s dpor %s %s\n' "$(basename $d)" "$u" "$s" "$p" "$tag"
done
