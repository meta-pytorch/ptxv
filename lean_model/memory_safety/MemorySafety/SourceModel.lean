/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.Code.Types
import MemorySafety.Program

open Aeneas Aeneas.Std PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false

namespace MemorySafety.Source

def allocationModel (allocation : memory_safety.Allocation) : Allocation :=
  ⟨allocation.base.val, allocation.bytes.val, allocation.initially_defined⟩

def kindModel : memory_safety.AccessKind → AccessKind
  | .Load => .load
  | .Store defined => .store defined

def requestModel (request : memory_safety.Request) : Request :=
  ⟨request.address.val, request.width.val, request.alignment.val, kindModel request.kind⟩

def instructionModel : memory_safety.Instruction → Instruction
  | .Access request => .access (requestModel request)
  | .Free allocation => .free allocation.val
  | .Choice targets => .choice (targets.val.map UScalar.val)
  | .Skip => .skip
  | .Halt => .halt
  | .Unsupported => .unsupported

def programModel (program : memory_safety.Program) : Program :=
  ⟨program.allocations.val.map allocationModel,
    program.threads.val.map (fun code => code.val.map instructionModel)⟩

def writtenModel (written : memory_safety.WrittenSpan) : WrittenSpan :=
  ⟨written.address.val, written.width.val, written.defined⟩

def recordModel (record : memory_safety.AccessRecord) : AccessRecord :=
  ⟨record.thread.val, record.address.val, record.width.val, record.is_write⟩

def stateModel (state : memory_safety.State) : State :=
  ⟨state.pcs.val.map UScalar.val, state.live.val, state.writes.val.map writtenModel,
    state.history.val.map recordModel, state.bad⟩

def eventModel : memory_safety.Event → Event
  | .Access thread request => .access thread.val (requestModel request)
  | .Free thread allocation => .free thread.val allocation.val
  | .Control => .control

def moveModel (movement : memory_safety.Move) : Move :=
  ⟨eventModel movement.event, movement.pcs.val.map UScalar.val⟩

@[simp] theorem stateModel_bad (state : memory_safety.State) :
    (stateModel state).bad = state.bad := rfl

@[simp] theorem programModel_allocation_length (program : memory_safety.Program) :
    (programModel program).allocations.length = program.allocations.val.length := by
  simp [programModel]

@[simp] theorem programModel_thread_length (program : memory_safety.Program) :
    (programModel program).threads.length = program.threads.val.length := by
  simp [programModel]

@[simp] theorem stateModel_live_length (state : memory_safety.State) :
    (stateModel state).live.length = state.live.val.length := rfl

end MemorySafety.Source
