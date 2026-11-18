/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteBounds

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false

namespace MemoryWrite.MayWrite

def wordCount (registers : Nat) : Nat := registers / 64 + if registers % 64 = 0 then 0 else 1

theorem register_word_bound {register registers : Nat} (inside : register < registers) :
    register / 64 < wordCount registers := by
  unfold wordCount
  split <;> omega

theorem word_count_fits (registers : Usize) : wordCount registers.val ≤ Usize.max := by
  have division := Nat.div_mul_le_self registers.val 64
  have registerBound : registers.val ≤ Usize.max := by scalar_tac
  have maxLarge : 64 < Usize.max := by scalar_tac
  unfold wordCount
  split <;> omega

theorem graph_write_word_bound (graph : cache_graph.MayWriteGraph) (bounded : GraphBounded graph)
    (pc : Usize) (pcBound : pc.val < graph.reach.val.length) :
    ∀ register ∈ graph.writes.val[pc.val]!.val,
      register.val / 64 < wordCount graph.register_count.val := by
  intro register member
  exact register_word_bound (bounded.2.2.2 pc.val pcBound register member)

end MemoryWrite.MayWrite
