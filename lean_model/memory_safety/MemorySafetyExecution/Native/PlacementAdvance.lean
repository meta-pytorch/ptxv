/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.PlacementStep
import MemorySafetyExecution.Native.LayoutPlacement

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option Elab.async false
set_option debug.skipKernelTC false
namespace MemorySafetyExecution.Native

theorem source_prepared_layout_valid (program : CoreProgram) (state : CoreState)
    (actor : Usize) (commit : execution_core.Commit)
    (prepared : execution_core.prepare program state actor=.ok (.Ok commit)) :
    execution_core.layout_valid program state=.ok true := by
  unfold execution_core.prepare at prepared
  obtain ⟨valid,computed,prepared⟩ := MemorySafety.Source.source_bind_success _ _ _ prepared
  cases valid with
  | false => simp at prepared
  | true => exact computed

theorem source_advance_native_placements (program rightProgram : CoreProgram) (left right leftAfter : CoreState)
    (shadow shadowAfter : value_flow_dependencies.State) (actor : Usize) (event : memory_safety.Event)
    (sameCode : program.code=rightProgram.code)
    (placements : ProgramPlacements program rightProgram)
    (related : StateInputsRelated shadow left right)
    (advanced : input_independent_execution.advance_dependencies program left shadow actor=.ok (some shadowAfter))
    (captured : execution_core.execute program left actor=.ok (.Ok event,leftAfter)) :
    ∃ rightEvent rightAfter, execution_core.execute rightProgram right actor=.ok (.Ok rightEvent,rightAfter) ∧
      EventPlacement program.layouts rightProgram.layouts event rightEvent ∧
      StateInputsRelated shadowAfter leftAfter rightAfter := by
  obtain ⟨context⟩ := source_dependency_step_context program left shadow shadowAfter actor advanced
  have leftValid := source_prepared_layout_valid program left actor context.commit context.prepared
  have rightValid := source_layout_valid_placements program rightProgram left right placements
    (input_storage_shapes shadow left right related) leftValid
  exact source_advance_native_placements_valid program rightProgram left right leftAfter shadow shadowAfter actor event
    sameCode placements rightValid related advanced captured

end MemorySafetyExecution.Native
