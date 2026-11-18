/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RegisterIndexRowProgress
import MemoryWrite.RegisterIndexBuildGraph

open Aeneas Aeneas.Std Result PtxvMemoryWrite
open MemoryWrite.MayWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.RegisterIndex


@[step]
theorem rust_build_row_total (destinations : Slice (alloc.vec.Vec symbolic.RegisterKey))
    (pc : Usize) (before : register_index.RegisterIndex)
    (pcBound : pc.val < destinations.val.length) (valid : Valid before)
    (countFits : before.count.val + (destinations.val[pc.val]!).val.length ≤ Usize.max)
    (pagesFit : before.pages.val.length + (destinations.val[pc.val]!).val.length ≤ Usize.max) :
    register_index.build_writes_loop0_loop0 destinations before pc (alloc.vec.Vec.new Usize) 0#usize
      ⦃ result => Valid result.1 ∧ Preserves before result.1 ∧
        RowRepresents result.1 (destinations.val[pc.val]!).val result.2.val ∧
        result.1.count.val ≤ before.count.val + (destinations.val[pc.val]!).val.length ∧
        result.1.pages.val.length ≤ before.pages.val.length + (destinations.val[pc.val]!).val.length ⦄ := by
  obtain ⟨⟨after, row⟩, executed, bounds⟩ := MemoryWrite.spec_returns _ _
    (rust_build_row_returns destinations pc before pcBound valid countFits pagesFit)
  rw [executed, WP.spec_ok]
  have correct := successful_build_row destinations pc before after (alloc.vec.Vec.new _) row 0#usize pcBound valid
    ⟨by scalar_tac, by simp, by intro earlier bound; scalar_tac⟩ executed
  exact ⟨correct.1, correct.2.1, correct.2.2, bounds.2⟩

end MemoryWrite.RegisterIndex
