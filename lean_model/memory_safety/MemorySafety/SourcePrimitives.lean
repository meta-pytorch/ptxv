/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.Code.Funs
import MemorySafety.SourceModel

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 3000000
set_option Elab.async false

namespace MemorySafety.Source

@[step]
theorem source_overlaps (a aw b bw : U64) :
    memory_safety.overlaps a aw b bw
      ⦃ result => result = true ↔ Overlap a.val aw.val b.val bw.val ⦄ := by
  unfold memory_safety.overlaps
  split
  · split
    · step with UScalar.cast_inBounds_spec .U128 a (by scalar_tac) as ⟨a128, ha⟩
      step with UScalar.cast_inBounds_spec .U128 b (by scalar_tac) as ⟨b128, hb⟩
      step with UScalar.cast_inBounds_spec .U128 bw (by scalar_tac) as ⟨bw128, hbw⟩
      step as ⟨bend, hbend⟩
      split
      · step with UScalar.cast_inBounds_spec .U128 b (by scalar_tac) as ⟨b128', hb'⟩
        step with UScalar.cast_inBounds_spec .U128 a (by scalar_tac) as ⟨a128', ha'⟩
        step with UScalar.cast_inBounds_spec .U128 aw (by scalar_tac) as ⟨aw128, haw⟩
        step as ⟨aend, haend⟩
        simp only [Overlap]
        scalar_tac
      · simp only [WP.spec_ok, Overlap]
        scalar_tac
    · simp only [WP.spec_ok, Overlap]
      scalar_tac
  · simp only [WP.spec_ok, Overlap]
    scalar_tac

@[step]
theorem source_contains (base width address : U64) :
    memory_safety.contains base width address
      ⦃ result => result = true ↔ Contains base.val width.val address.val ⦄ := by
  unfold memory_safety.contains
  split
  · step with UScalar.cast_inBounds_spec .U128 address (by scalar_tac) as ⟨address128, ha⟩
    step with UScalar.cast_inBounds_spec .U128 base (by scalar_tac) as ⟨base128, hb⟩
    step with UScalar.cast_inBounds_spec .U128 width (by scalar_tac) as ⟨width128, hw⟩
    step as ⟨finish, hf⟩
    simp only [Contains]
    scalar_tac
  · simp only [WP.spec_ok, Contains]
    scalar_tac

end MemorySafety.Source
