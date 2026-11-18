/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Aeneas

open Aeneas Aeneas.Std Result

set_option autoImplicit false

namespace MemoryWrite

theorem spec_returns {α : Type} (computation : Result α) (post : α → Prop)
    (specification : WP.spec computation post) : ∃ value, computation = .ok value ∧ post value := by
  cases computation with
  | ret value => exact ⟨value, rfl, (WP.spec_ok value).mp specification⟩
  | div => simp only [WP.spec_div] at specification
  | vis effect next => simp only [WP.spec_vis] at specification

end MemoryWrite
