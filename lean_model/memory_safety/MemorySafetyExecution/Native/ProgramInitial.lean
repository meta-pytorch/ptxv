/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.InputSafety
import MemorySafetyExecution.Native.InputInitialRelation

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
open MemorySafety.Source MemorySafetyExecution.Source

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

def withInitial (program : CoreProgram) (initial : CoreState) : CoreProgram := {program with initial}

theorem source_layout_initial_irrelevant (program : CoreProgram) (initial state : CoreState) :
    execution_core.layout_valid (withInitial program initial) state=execution_core.layout_valid program state := by
  rfl

theorem source_prepare_initial_irrelevant (program : CoreProgram) (initial state : CoreState) (actor : Usize) :
    execution_core.prepare (withInitial program initial) state actor=execution_core.prepare program state actor := by
  rfl

theorem source_resolve_initial_irrelevant (program : CoreProgram) (initial state : CoreState)
    (pointer : value_flow.Pointer) (width : U8) (write : Bool) :
    execution_core.resolve_access (withInitial program initial) state pointer width write=
      execution_core.resolve_access program state pointer width write := rfl

theorem source_commit_initial_irrelevant (program : CoreProgram) (initial state : CoreState)
    (commit : execution_core.Commit) :
    execution_core.commit_action (withInitial program initial) state commit=
      execution_core.commit_action program state commit := by
  rfl

theorem source_execute_initial_irrelevant (program : CoreProgram) (initial state : CoreState) (actor : Usize) :
    execution_core.execute (withInitial program initial) state actor=execution_core.execute program state actor := by
  simp only [execution_core.execute,source_prepare_initial_irrelevant,source_commit_initial_irrelevant]

theorem source_trace_initial_irrelevant (program : CoreProgram) (initial start after : CoreState)
    (events : List memory_safety.Event) (trace : CoreTrace (withInitial program initial) start events after) :
    CoreTrace program start events after := by
  induction trace with
  | initial => exact .initial
  | step prior transition ih =>
    obtain ⟨actor,running,executed⟩ := transition
    exact .step ih ⟨actor,running,by simpa only [source_execute_initial_irrelevant] using executed⟩

theorem source_reachable_initial_irrelevant (program : CoreProgram) (initial start after : CoreState)
    (reachable : CoreReachable (withInitial program initial) start after) : CoreReachable program start after := by
  induction reachable with
  | initial => exact .initial
  | step prior transition ih =>
    obtain ⟨actor,running,executed⟩ := transition
    exact .step ih ⟨actor,running,by simpa only [source_execute_initial_irrelevant] using executed⟩

end MemorySafetyExecution.Native
