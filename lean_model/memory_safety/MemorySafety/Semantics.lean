/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.Certificate

set_option autoImplicit false
set_option maxHeartbeats 2000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafety

structure Allocation where
  base : Nat
  bytes : Nat
  initiallyDefined : Bool
  deriving DecidableEq, Repr

inductive AccessKind where
  | load
  | store (defined : Bool)
  deriving DecidableEq, Repr

structure Request where
  address : Nat
  width : Nat
  alignment : Nat
  kind : AccessKind
  deriving DecidableEq, Repr

inductive Instruction where
  | access (request : Request)
  | free (allocation : Nat)
  | choice (targets : List Nat)
  | skip
  | halt
  | unsupported
  deriving DecidableEq, Repr

structure Program where
  allocations : List Allocation
  threads : List (List Instruction)
  deriving DecidableEq, Repr

structure WrittenSpan where
  address : Nat
  width : Nat
  defined : Bool
  deriving DecidableEq, Repr

structure AccessRecord where
  thread : Nat
  address : Nat
  width : Nat
  isWrite : Bool
  deriving DecidableEq, Repr

structure State where
  pcs : List Nat
  live : List Bool
  writes : List WrittenSpan
  history : List AccessRecord
  bad : Bool
  deriving DecidableEq, Repr

def addressLimit : Nat := 2 ^ 64

def Overlap (address width otherAddress otherWidth : Nat) : Prop :=
  0 < width ∧ 0 < otherWidth ∧ address < otherAddress + otherWidth ∧ otherAddress < address + width

instance (address width otherAddress otherWidth : Nat) :
    Decidable (Overlap address width otherAddress otherWidth) := by unfold Overlap; infer_instance

def Contains (base width address : Nat) : Prop := base ≤ address ∧ address < base + width

instance (base width address : Nat) : Decidable (Contains base width address) := by
  unfold Contains; infer_instance

def LayoutSafe (program : Program) : Prop :=
  (∀ allocation ∈ program.allocations, 0 < allocation.bytes ∧ allocation.base + allocation.bytes ≤ addressLimit) ∧
  ∀ first ∈ program.allocations.zipIdx, ∀ second ∈ program.allocations.zipIdx,
    first.2 ≠ second.2 → ¬Overlap first.1.base first.1.bytes second.1.base second.1.bytes

instance (program : Program) : Decidable (LayoutSafe program) := by
  unfold LayoutSafe
  infer_instance

def initialState (program : Program) : State :=
  ⟨List.replicate program.threads.length 0, List.replicate program.allocations.length true, [], [], false⟩

def initial (program : Program) : Option State :=
  if LayoutSafe program then some (initialState program) else none

def byteDefined (program : Program) (state : State) (address : Nat) : Bool :=
  match state.writes.reverse.find? (fun written => decide (Contains written.address written.width address)) with
  | some written => written.defined
  | none =>
    match program.allocations.find? (fun allocation => decide (Contains allocation.base allocation.bytes address)) with
    | some allocation => allocation.initiallyDefined
    | none => false

def AccessKind.isWrite : AccessKind → Bool
  | .load => false
  | .store _ => true

def accessRecord (thread : Nat) (request : Request) : AccessRecord :=
  ⟨thread, request.address, request.width, request.kind.isWrite⟩

def Conflict (before after : AccessRecord) : Prop :=
  before.thread ≠ after.thread ∧ (before.isWrite = true ∨ after.isWrite = true) ∧
    Overlap before.address before.width after.address after.width

instance (before after : AccessRecord) : Decidable (Conflict before after) := by
  unfold Conflict; infer_instance

def NoRace (state : State) (access : AccessRecord) : Prop :=
  ∀ before ∈ state.history, ¬Conflict before access

instance (state : State) (access : AccessRecord) : Decidable (NoRace state access) := by
  unfold NoRace; infer_instance

def LiveSpan (program : Program) (state : State) (request : Request) : Prop :=
  ∃ indexed ∈ program.allocations.zipIdx,
    state.live[indexed.2]? = some true ∧ indexed.1.base ≤ request.address ∧
      request.address + request.width ≤ indexed.1.base + indexed.1.bytes

instance (program : Program) (state : State) (request : Request) :
    Decidable (LiveSpan program state request) := by unfold LiveSpan; infer_instance

def Initialized (program : Program) (state : State) (request : Request) : Prop :=
  ∀ offset ∈ List.range request.width, byteDefined program state (request.address + offset) = true

instance (program : Program) (state : State) (request : Request) :
    Decidable (Initialized program state request) := by unfold Initialized; infer_instance

def AccessSafe (program : Program) (state : State) (thread : Nat) (request : Request) : Prop :=
  state.live.length = program.allocations.length ∧
  0 < request.width ∧
  0 < request.alignment ∧ request.address % request.alignment = 0 ∧
  request.address + request.width ≤ addressLimit ∧
  LiveSpan program state request ∧
  NoRace state (accessRecord thread request) ∧
  (request.kind = .load → Initialized program state request)

instance (program : Program) (state : State) (thread : Nat) (request : Request) :
    Decidable (AccessSafe program state thread request) := by unfold AccessSafe; infer_instance

def accessValid (program : Program) (state : State) (thread : Nat) (request : Request) : Bool :=
  decide (AccessSafe program state thread request)

def freeRequest (allocation : Allocation) : Request :=
  ⟨allocation.base, allocation.bytes, 1, .store false⟩

def FreeSafe (program : Program) (state : State) (thread allocation : Nat) : Prop :=
  match program.allocations[allocation]? with
  | none => False
  | some block => state.live[allocation]? = some true ∧ AccessSafe program state thread (freeRequest block)

instance (program : Program) (state : State) (thread allocation : Nat) :
    Decidable (FreeSafe program state thread allocation) := by
  unfold FreeSafe
  cases program.allocations[allocation]? <;> infer_instance

inductive Event where
  | access (thread : Nat) (request : Request)
  | free (thread allocation : Nat)
  | control
  deriving DecidableEq, Repr

def EventSafe (program : Program) (state : State) : Event → Prop
  | .access thread request => AccessSafe program state thread request
  | .free thread allocation => FreeSafe program state thread allocation
  | .control => True

instance (program : Program) (state : State) (event : Event) : Decidable (EventSafe program state event) := by
  cases event <;> unfold EventSafe <;> infer_instance

def updateMemory (program : Program) (state : State) : Event → State
  | .control => state
  | .access thread request =>
    if AccessSafe program state thread request then
      { state with
        writes := match request.kind with
          | .load => state.writes
          | .store defined => state.writes ++ [⟨request.address, request.width, defined⟩]
        history := state.history ++ [accessRecord thread request] }
    else { state with bad := true }
  | .free thread allocation =>
    if FreeSafe program state thread allocation then
      match program.allocations[allocation]? with
      | none => { state with bad := true }
      | some block =>
        { state with
          live := state.live.set allocation false
          history := state.history ++ [accessRecord thread (freeRequest block)] }
    else { state with bad := true }

theorem access_safe_components (program : Program) (state : State) (thread : Nat) (request : Request)
    (safe : AccessSafe program state thread request) :
    0 < request.width ∧ 0 < request.alignment ∧ request.address % request.alignment = 0 ∧
    request.address + request.width ≤ addressLimit ∧ LiveSpan program state request ∧
    NoRace state (accessRecord thread request) ∧ (request.kind = .load → Initialized program state request) :=
  safe.2

theorem safe_read_every_byte (program : Program) (state : State) (thread : Nat) (request : Request)
    (safe : AccessSafe program state thread request) (read : request.kind = .load)
    (offset : Nat) (inside : offset < request.width) :
    byteDefined program state (request.address + offset) = true :=
  safe.2.2.2.2.2.2.2 read offset (List.mem_range.mpr inside)

theorem safe_access_conflicts_with_no_previous_access
    (program : Program) (state : State) (thread : Nat) (request : Request)
    (safe : AccessSafe program state thread request) (before : AccessRecord) (recorded : before ∈ state.history) :
    ¬Conflict before (accessRecord thread request) := safe.2.2.2.2.2.2.1 before recorded

theorem safe_free_conflicts_with_no_previous_access
    (program : Program) (state : State) (thread allocation : Nat)
    (safe : FreeSafe program state thread allocation) (block : Allocation)
    (found : program.allocations[allocation]? = some block)
    (before : AccessRecord) (recorded : before ∈ state.history) :
    ¬Conflict before (accessRecord thread (freeRequest block)) := by
  have both : state.live[allocation]? = some true ∧ AccessSafe program state thread (freeRequest block) := by
    simpa [FreeSafe, found] using safe
  have accepted := both.2
  exact safe_access_conflicts_with_no_previous_access program state thread _ accepted before recorded

theorem memory_update_safe (program : Program) (state : State) (event : Event)
    (clean : (updateMemory program state event).bad = false) : EventSafe program state event := by
  cases event with
  | control => trivial
  | access thread request =>
    by_cases safe : AccessSafe program state thread request
    · exact safe
    · simp [updateMemory, safe] at clean
  | free thread allocation =>
    by_cases safe : FreeSafe program state thread allocation
    · exact safe
    · simp [updateMemory, safe] at clean

end MemorySafety
