#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# RDP2 check (MISS 282 follow-up; no finding at 94e057e8): E-POLL-ANY against undefined bytes.
# PART (8-byte strong load: one defined word, one never-written), SYS (ld.volatile.global =
# LDG.E.STRONG.SYS of never-written scratch), SUB8 (UST with a one-byte strong store of undefined
# data), TMAU / TMAS (bulk copy / bulk store of undefined bytes over the polled word).
# Expected: default and DPOR = unreduced (PTXV_NO_VIP=1). At b66fd47b the default explorer gave
# PART, SYS [clean] on both fronts and SUB8 SASS [undef-insensitive] against [violation]; TMAU and
# TMAS agree at both revisions: regression rows only, they do not discriminate the fix.
# Usage: PTXV=<ptxv> repro.sh   (exits 1 while any explorer differs)
here="$(cd "$(dirname "$0")" && pwd)"; rc=0
for V in PART SYS SUB8 TMAU TMAS; do for F in ptx sass; do o=$(V=$V FRONT=$F TO=150 "$here/run.sh" "$here"); echo "$V $F $o"; echo "$o" | grep -q DIFF && rc=1; done; done
echo rc=$rc; exit $rc
