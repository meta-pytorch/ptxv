/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RegisterIndexBuildProgress
import MemoryWrite.RegisterIndexBuildGraph

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.RegisterIndex


@[step]
theorem rust_build_writes_total (destinations : Slice (alloc.vec.Vec symbolic.RegisterKey))
    (fits : BuildFits destinations.val) :
    register_index.build_writes destinations ⦃ result => Valid result.1 ∧
      GraphRepresents result.1 destinations.val result.2.val ∧
      result.1.count.val ≤ keyCount destinations.val ∧
      result.1.pages.val.length ≤ keyCount destinations.val ⦄ := by
  obtain ⟨⟨index, writes⟩, executed, bounds⟩ := MemoryWrite.spec_returns _ _
    (rust_build_writes_returns destinations fits)
  rw [executed, WP.spec_ok]
  have correct := successful_build_writes destinations index writes executed
  exact ⟨correct.1, correct.2, bounds.2⟩

end MemoryWrite.RegisterIndex
