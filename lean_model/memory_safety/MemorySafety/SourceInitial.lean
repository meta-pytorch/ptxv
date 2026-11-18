/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.SourceLayout

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 3000000
set_option Elab.async false

namespace MemorySafety.Source

@[step]
theorem source_initial (program : memory_safety.Program) :
    memory_safety.initial program
      ⦃ result => ∀ state, result = some state →
        LayoutSafe (programModel program) ∧ stateModel state = initialState (programModel program) ⦄ := by
  unfold memory_safety.initial
  step with source_layout_valid program as ⟨valid, layout⟩
  cases valid with
  | false => simp
  | true =>
    simp only [ite_true]
    step with alloc.vec.from_elem_spec core.clone.CloneUsize 0#usize
      (alloc.vec.Vec.len program.threads) (by rfl) as ⟨pcs, pcsEq, pcsLen⟩
    step with alloc.vec.from_elem_spec core.clone.CloneBool true
      (alloc.vec.Vec.len program.allocations) (by rfl) as ⟨live, liveEq, liveLen⟩
    simp only [Option.some.injEq]
    intro state equal
    subst state
    refine ⟨layout rfl, ?_⟩
    simp [stateModel, initialState, programModel, pcsEq, liveEq, alloc.vec.Vec.new]

end MemorySafety.Source
