/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.ConcreteSafety

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

def accessRecord (actor : Usize) (request : ptxv_memory_write.memory_safety.Request) :
    MemorySafety.AccessRecord where
  thread := actor.val
  address := request.address.val
  width := request.width.val
  isWrite := match request.kind with
    | .Load => false
    | .Store _ => true

theorem no_race_excludes_common_native_byte
    {program : CoreProgram} {state : CoreState}
    {request : ptxv_memory_write.memory_safety.Request}
    (span : ConcreteSpan program state request) (actor : Usize)
    (monitor : MemorySafety.State)
    (safe : MemorySafety.NoRace monitor (accessRecord actor request))
    (previous : MemorySafety.AccessRecord) (recorded : previous ∈ monitor.history)
    (different : previous.thread ≠ actor.val)
    (writes : previous.isWrite = true ∨ (accessRecord actor request).isWrite = true) :
    ∀ address, ¬(previous.address ≤ address ∧ address < previous.address + previous.width ∧
      touches span address) := by
  intro address both
  have current := (touches_exactly_request span address).1 both.2.2
  apply safe previous recorded
  refine ⟨different, writes, ?_⟩
  change MemorySafety.Overlap previous.address previous.width request.address.val request.width.val
  unfold MemorySafety.Overlap
  omega

end MemorySafetyExecution.Native
