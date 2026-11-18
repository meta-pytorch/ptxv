/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.NamesComplete

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false

namespace PtxFrontendProgram

theorem source_names_equal_iff (source : Slice U8) (left right : SourceToken) :
    strict_memory_frontend_parse.names_equal source left right=.ok true ↔ NameEqual source.val left right :=
  ⟨source_names_equal source left right,source_names_equal_complete source left right⟩

theorem source_names_equal_false (source : Slice U8) (left right : SourceToken)
    (computed : strict_memory_frontend_parse.names_equal source left right=.ok false) :
    ¬NameEqual source.val left right := by
  intro meaning
  have positive := source_names_equal_complete source left right meaning
  rw [computed] at positive
  simp at positive

end PtxFrontendProgram
