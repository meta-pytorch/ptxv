/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.Semantics
import MemorySafety.Semantics

set_option autoImplicit false
set_option debug.skipKernelTC false
namespace MemorySafetyExecution.MachineModel

structure Layout where
  base : Nat
  bytes : Nat
  alignment : Nat

structure Memory where
  bytes : List Nat
  defined : List Bool
  live : Bool

structure Thread where
  pc : Nat
  running : Bool
  ids : ValueFlow.ThreadIds
  registers : ValueFlow.RegisterBank

structure State where
  threads : List Thread
  allocations : List Memory

structure Program where
  code : List ValueFlow.GuardedInstruction
  layouts : List Layout
  outputs : List Nat

def render (program : Program) : ValueFlow.Value → Option ValueFlow.Word
  | .scalar bits => some ⟨bits,0,none⟩
  | .pointer pointer => do
      let layout ← program.layouts[pointer.allocation]?
      pure ⟨(layout.base+pointer.offset)%2^64,0,some pointer⟩

def writeVirtual (bank : ValueFlow.RegisterBank) (index : Nat) (word : ValueFlow.Word) :
    ValueFlow.RegisterBank :=
  { bank with virtual := fun query => if query=index then some word else bank.virtual query }

def writePredicate (bank : ValueFlow.RegisterBank) (index : Nat) (value : Bool) :
    ValueFlow.RegisterBank :=
  { bank with predicate := fun query => if query=index then some value else bank.predicate query }

def setThread (state : State) (actor : Nat) (thread : Thread) : State :=
  { state with threads := state.threads.set actor thread }

def resolve (program : Program) (state : State) (pointer : ValueFlow.Pointer)
    (width : Nat) (write : Bool) : Option (MemorySafety.Request × Memory) := do
  let layout ← program.layouts[pointer.allocation]?
  let memory ← state.allocations[pointer.allocation]?
  if ValueFlow.supportedWidth width ∧ layout.alignment≠0 ∧
      layout.alignment%width=0 ∧ pointer.offset%width=0 ∧
      layout.base%width=0 ∧ pointer.offset+width≤layout.bytes ∧
      layout.base+pointer.offset+width≤2^64 ∧
      memory.live=true ∧ pointer.offset+width≤memory.bytes.length ∧
      memory.defined.length=memory.bytes.length ∧
      (write=true ∨ ∀index∈List.range width,memory.defined[pointer.offset+index]?=some true) then
    pure (⟨layout.base+pointer.offset,width,width,if write then .store true else .load⟩,memory)
  else none

def loadFold (memory : Memory) (offset : Nat) : Nat → Nat → Nat → Nat
  | 0,_,value => value
  | remaining+1,index,value =>
      loadFold memory offset remaining (index+1)
        (value ||| ((memory.bytes[offset+index]?.getD 0)*2^(8*index)%2^64))

def signedLoad (value width : Nat) (signed : Bool) : Nat :=
  if signed ∧ width<8 then
    ((BitVec.ofNat 64 value <<< (64-width*8)).sshiftRight (64-width*8)).toNat
  else value

def load (memory : Memory) (offset width : Nat) (signed : Bool) : Nat :=
  signedLoad (loadFold memory offset width 0 0) width signed

def store (memory : Memory) (offset width value : Nat) : Memory :=
  { memory with
    bytes := memory.bytes.zipIdx.map fun (byte,index) =>
      if offset ≤ index ∧ index < offset+width then value/2^(8*(index-offset))%256 else byte
    defined := memory.defined.zipIdx.map fun (flag,index) =>
      if offset ≤ index ∧ index < offset+width then true else flag }

def applyAction (program : Program) (state : State) (actor : Nat)
    (thread : Thread) : ValueFlow.Action → Option (MemorySafety.Event × State)
  | .next => some (.control,setThread state actor {thread with pc:=thread.pc+1})
  | .halt => some (.control,setThread state actor {thread with running:=false})
  | .jump target =>
      if target<program.code.length then
        some (.control,setThread state actor {thread with pc:=target}) else none
  | .writeVirtual destination value => do
      let word ← render program value
      pure (.control,setThread state actor
        {thread with pc:=thread.pc+1,registers:=writeVirtual thread.registers destination word})
  | .writePredicate destination value =>
      some (.control,setThread state actor
        {thread with pc:=thread.pc+1,registers:=writePredicate thread.registers destination value})
  | .load destination bits width signed pointer => do
      let (request,memory) ← resolve program state pointer width false
      let word : ValueFlow.Word := ⟨ValueFlow.truncate bits (load memory pointer.offset width signed),0,none⟩
      pure (.access actor request,setThread state actor
        {thread with pc:=thread.pc+1,registers:=writeVirtual thread.registers destination word})
  | .store width pointer value => do
      let (request,memory) ← resolve program state pointer width true
      let updated := {state with allocations := state.allocations.set pointer.allocation (store memory pointer.offset width value)}
      pure (.access actor request,setThread updated actor {thread with pc:=thread.pc+1})

def step (program : Program) (state : State) (actor : Nat) :
    Option (MemorySafety.Event × State) := do
  let thread ← state.threads[actor]?
  if thread.running then do
    let instruction ← program.code[thread.pc]?
    let action ← ValueFlow.evaluate thread.registers thread.ids instruction
    applyAction program state actor thread action
  else none

def Step (program : Program) (before : State) (event : MemorySafety.Event) (after : State) : Prop :=
  ∃actor,step program before actor=some (event,after)

inductive Reachable (program : Program) (initial : State) : State → Prop
  | initial : Reachable program initial initial
  | next {before after event} : Reachable program initial before → Step program before event after →
      Reachable program initial after

theorem step_deterministic (program : Program) (state : State) (actor : Nat)
    (event otherEvent : MemorySafety.Event) (after otherAfter : State)
    (first : step program state actor=some (event,after))
    (second : step program state actor=some (otherEvent,otherAfter)) :
    event=otherEvent ∧ after=otherAfter := by
  simpa only [Option.some.injEq,Prod.mk.injEq] using first.symm.trans second

end MemorySafetyExecution.MachineModel
