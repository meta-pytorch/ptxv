/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.Soundness

set_option autoImplicit false
set_option maxHeartbeats 3000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafety

def PreservesDefinedness : Instruction → Prop
  | .access request => match request.kind with
      | .load => True
      | .store defined => defined = true
  | .free _ => False
  | _ => True

def PreservesAllocation : Event → Prop
  | .access _ request => match request.kind with
      | .load => True
      | .store defined => defined = true
  | .free _ _ => False
  | .control => True

def DefinedWrites (state : State) : Prop :=
  ∀ written ∈ state.writes, written.defined = true

theorem thread_moves_preserve_allocation (program : Program) (state : State) (thread : Nat)
    (choices : List Move) (known : threadMoves program state thread = some choices)
    (preserves : ∀ code ∈ program.threads, ∀ instruction ∈ code, PreservesDefinedness instruction)
    (movement : Move) (member : movement ∈ choices) : PreservesAllocation movement.event := by
  unfold threadMoves at known
  cases codeEq : program.threads[thread]? <;> cases pcEq : state.pcs[thread]? <;>
    simp only [codeEq, pcEq] at known
  all_goals try contradiction
  case some.some code pc =>
    split at known
    · contradiction
    · split at known
      · cases known; cases member
      · cases instructionEq : code[pc]? with
        | none => simp [instructionEq] at known
        | some instruction =>
          have property := preserves code (List.mem_of_getElem? codeEq)
            instruction (List.mem_of_getElem? instructionEq)
          simp only [instructionEq] at known
          cases instruction with
          | free allocation => exact False.elim property
          | unsupported => simp only at known; contradiction
          | access request =>
            simp only at known
            cases known
            simp only [List.mem_singleton] at member
            cases member
            exact property
          | skip =>
            simp only at known
            cases known
            simp only [List.mem_singleton] at member
            cases member
            trivial
          | halt =>
            simp only at known
            cases known
            simp only [List.mem_singleton] at member
            cases member
            trivial
          | choice targets =>
            simp only at known
            split at known
            · contradiction
            · split at known
              · cases known
                obtain ⟨target, _, rfl⟩ := List.mem_map.mp member
                trivial
              · contradiction

theorem gather_moves_preserve_allocation (program : Program) (state : State)
    (threads : List Nat) (choices : List Move)
    (known : gatherMoves program state threads = some choices)
    (preserves : ∀ code ∈ program.threads, ∀ instruction ∈ code, PreservesDefinedness instruction)
    (movement : Move) (member : movement ∈ choices) : PreservesAllocation movement.event := by
  induction threads generalizing choices with
  | nil => simp [gatherMoves] at known; cases known; cases member
  | cons thread threads ih =>
    simp only [gatherMoves] at known
    cases first : threadMoves program state thread with
    | none => simp [first] at known
    | some firstChoices =>
      cases rest : gatherMoves program state threads with
      | none => simp [first, rest] at known
      | some restChoices =>
        simp [first, rest] at known
        cases known
        cases List.mem_append.mp member with
        | inl firstMember => exact thread_moves_preserve_allocation program state thread _ first preserves movement firstMember
        | inr restMember => exact ih _ rest restMember

theorem memory_safe_moves_known (program : Program) (safe : MemorySafe program)
    (before : State) (reachable : Reachable program before) : ∃ choices, moves program before = some choices := by
  cases known : moves program before with
  | some choices => exact ⟨choices, rfl⟩
  | none =>
    have transition : Step program before .control { before with bad := true } := by simp [Step, known]
    have clean := safe.2.1 _ (.step reachable transition)
    contradiction

theorem step_preserves_allocation (program : Program) (safe : MemorySafe program)
    (preserves : ∀ code ∈ program.threads, ∀ instruction ∈ code, PreservesDefinedness instruction)
    (before after : State) (event : Event) (reachable : Reachable program before)
    (transition : Step program before event after) : PreservesAllocation event := by
  obtain ⟨choices, known⟩ := memory_safe_moves_known program safe before reachable
  obtain ⟨movement, member, eventEq, _⟩ : ∃ movement ∈ choices,
      movement.event = event ∧ applyMove program before movement = after := by
    simpa only [Step, known] using transition
  rw [←eventEq]
  unfold moves at known
  split at known
  · contradiction
  · split at known
    · cases known; cases member
    · exact gather_moves_preserve_allocation program before _ choices known preserves movement member

theorem memory_update_preserves_definedness (program : Program) (state : State) (event : Event)
    (preserves : PreservesAllocation event) (defined : DefinedWrites state) :
    DefinedWrites (updateMemory program state event) ∧ (updateMemory program state event).live = state.live := by
  cases event with
  | control => exact ⟨defined, rfl⟩
  | free thread allocation => contradiction
  | access thread request =>
    by_cases accepted : AccessSafe program state thread request
    · simp only [updateMemory, accepted, ite_true]
      cases kindEq : request.kind with
      | load =>
        simpa only [kindEq, DefinedWrites] using And.intro defined (Eq.refl state.live)
      | store payload =>
        have payloadEq : payload = true := by simpa [PreservesAllocation, kindEq] using preserves
        subst payload
        refine ⟨?_, by trivial⟩
        intro written member
        simp only [List.mem_append, List.mem_singleton] at member
        rcases member with old | rfl
        · exact defined written old
        · rfl
    · simpa [updateMemory, accepted, DefinedWrites] using defined

theorem reachable_definedness (program : Program) (safe : MemorySafe program)
    (preserves : ∀ code ∈ program.threads, ∀ instruction ∈ code, PreservesDefinedness instruction)
    (state : State) (reachable : Reachable program state) :
    DefinedWrites state ∧ state.live = List.replicate program.allocations.length true := by
  induction reachable with
  | initial =>
    constructor
    · intro written member
      cases member
    · rfl
  | @step before event after reachable transition ih =>
    have preserved := step_preserves_allocation program safe preserves before after event reachable transition
    obtain ⟨choices, known⟩ := memory_safe_moves_known program safe before reachable
    obtain ⟨movement, _, eventEq, result⟩ : ∃ movement ∈ choices,
        movement.event = event ∧ applyMove program before movement = after := by
      simpa only [Step, known] using transition
    have updated := memory_update_preserves_definedness program before event preserved ih.1
    rw [←result, applyMove]
    split <;> simpa only [eventEq, DefinedWrites] using And.intro updated.1 (updated.2.trans ih.2)

theorem initially_defined_byte_remains_defined (program : Program) (state : State) (address : Nat)
    (defined : DefinedWrites state)
    (initial : byteDefined program (initialState program) address = true) :
    byteDefined program state address = true := by
  unfold byteDefined
  cases found : state.writes.reverse.find? (fun written => decide (Contains written.address written.width address)) with
  | some written =>
    exact defined written (List.mem_reverse.mp (List.mem_of_find?_eq_some found))
  | none => simpa only [byteDefined, initialState, List.reverse_nil, List.find?_nil] using initial

theorem layout_unique_containing (program : Program) (layout : LayoutSafe program)
    (first second : Allocation) (firstMember : first ∈ program.allocations)
    (secondMember : second ∈ program.allocations) (address : Nat)
    (firstContains : Contains first.base first.bytes address)
    (secondContains : Contains second.base second.bytes address) : first = second := by
  by_contra different
  obtain ⟨firstIndex, firstBound, firstEq⟩ := List.mem_iff_getElem.mp firstMember
  obtain ⟨secondIndex, secondBound, secondEq⟩ := List.mem_iff_getElem.mp secondMember
  have firstIndexed : (first, firstIndex) ∈ program.allocations.zipIdx := by
    have member := List.getElem_mem (l := program.allocations.zipIdx) (n := firstIndex)
      (by simpa using firstBound)
    simpa [List.getElem_zipIdx, firstEq] using member
  have secondIndexed : (second, secondIndex) ∈ program.allocations.zipIdx := by
    have member := List.getElem_mem (l := program.allocations.zipIdx) (n := secondIndex)
      (by simpa using secondBound)
    simpa [List.getElem_zipIdx, secondEq] using member
  have indicesDifferent : firstIndex ≠ secondIndex := by
    intro same
    subst secondIndex
    exact different (firstEq.symm.trans secondEq)
  have disjoint := layout.2 _ firstIndexed _ secondIndexed indicesDifferent
  apply disjoint
  change Overlap first.base first.bytes second.base second.bytes
  have firstPositive := (layout.1 first firstMember).1
  have secondPositive := (layout.1 second secondMember).1
  unfold Contains at firstContains secondContains
  unfold Overlap
  omega

theorem allocation_initially_defined_byte (program : Program) (layout : LayoutSafe program)
    (allocation : Allocation) (member : allocation ∈ program.allocations)
    (defined : allocation.initiallyDefined = true) (address : Nat)
    (inside : Contains allocation.base allocation.bytes address) :
    byteDefined program (initialState program) address = true := by
  simp only [byteDefined, initialState, List.reverse_nil, List.find?_nil]
  cases found : program.allocations.find? (fun allocation =>
      decide (Contains allocation.base allocation.bytes address)) with
  | none =>
    have absent := List.find?_eq_none.mp found allocation member
    exact False.elim (absent (by simpa using inside))
  | some containing =>
    have containingMember := List.mem_of_find?_eq_some found
    have contains : Contains containing.base containing.bytes address := by
      simpa using List.find?_some found
    have same := layout_unique_containing program layout containing allocation
      containingMember member address contains inside
    simpa only [same] using defined

end MemorySafety
