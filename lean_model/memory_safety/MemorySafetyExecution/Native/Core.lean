/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.Code.Funs

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 3000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

abbrev CoreState := ptxv_memory_write.execution_core.State
abbrev CoreProgram := ptxv_memory_write.execution_core.Program
abbrev Event := ptxv_memory_write.memory_safety.Event
abbrev Fault := ptxv_memory_write.execution_core.Fault
def CoreStep (program : CoreProgram) (state : CoreState) (event : Event)
    (next : CoreState) : Prop :=
  ∃ actor, ptxv_memory_write.execution_core.running state actor = .ok true ∧
    ptxv_memory_write.execution_core.execute program state actor = .ok (.Ok event, next)

inductive CoreReachable (program : CoreProgram) (initial : CoreState) : CoreState → Prop where
  | initial : CoreReachable program initial initial
  | step {before after : CoreState} {event : Event} :
      CoreReachable program initial before → CoreStep program before event after →
      CoreReachable program initial after

end MemorySafetyExecution.Native
