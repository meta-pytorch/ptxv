/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.ConcreteSafety
import MemorySafetyExecution.Native.OwnerRefinement

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

theorem native_reachable_step_concrete {L F : Type}
    (program : CoreProgram) (initial : CoreState) (before after : ExecutionMachine L)
    (reachable : NativeReachable (.Strict { program, state := initial, fault := none }) before)
    (dictionary : StepDictionary L F) (actor : Usize) (callback : F) (event : Event)
    (executed : machine_memory_safety.ExecutionMachine.step dictionary before actor callback =
      .ok (.Ok event, after)) :
    ∃ state next,
      before = .Strict { program, state, fault := none } ∧
      after = .Strict { program, state := next, fault := none } ∧
      CoreReachable program initial state ∧ ConcreteMemorySafety program state event next := by
  obtain ⟨state, beforeEq, stateReachable⟩ := native_reachable_projects program initial before reachable
  subst before
  obtain ⟨next, coreRun, afterEq⟩ := source_strict_step_projection dictionary program state actor callback
    event after executed
  exact ⟨state, next, rfl, afterEq, stateReachable,
    source_execute_concrete_safety program state next actor event coreRun⟩

end MemorySafetyExecution.Native
