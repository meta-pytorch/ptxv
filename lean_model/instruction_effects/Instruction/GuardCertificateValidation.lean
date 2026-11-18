/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GuardCertificateCheck

open Aeneas Aeneas.Std Result PtxvIr
open Instruction.Partial

set_option autoImplicit false
set_option maxHeartbeats 5000000

namespace Instruction.GuardCertificate

def effectiveGuard (guard : Pred) (known : Option Bool) : Pred :=
  match known with
  | some true => Pred.TRUE
  | _ => guard

def returnsAt (rows : Slice (alloc.vec.Vec Usize)) (index : Nat) : Option (Slice Usize) :=
  (rows.val[index]?).map alloc.vec.Vec.deref

def NodeChecked {C : Type} (reader : guard_certificate.Constants C)
    (front : Front) (code : Slice Inst) (states : Slice (Option (alloc.vec.Vec Fact)))
    (returns : Slice (alloc.vec.Vec Usize)) (constants : C) (index : Nat)
    (observation : Option (Option Bool)) : Prop :=
  (states.val[index]? = some none ∧ observation = none) ∨
  ∃ (facts : alloc.vec.Vec Fact) (instruction : Inst) (pc : Usize) (known : Option Bool)
    (targets : alloc.vec.Vec Usize),
    states.val[index]? = some (some facts) ∧ code.val[index]? = some instruction ∧ pc.val = index ∧
    observation = some known ∧
    guard_certificate.decision front (alloc.vec.Vec.deref facts) instruction.guard = .ok known ∧
    control_flow.successors instruction.op (effectiveGuard instruction.guard known)
      { pc, count := Slice.len code, cut := 0#u8, fold := known,
        returns := returnsAt returns index } = .ok targets ∧
    ∀ position, (inside : position < targets.val.length) →
      TargetChecked reader
        { front, instruction, facts := alloc.vec.Vec.deref facts, constants }
        states targets.val[position]

def RowsChecked {C : Type} (reader : guard_certificate.Constants C)
    (front : Front) (code : Slice Inst) (states : Slice (Option (alloc.vec.Vec Fact)))
    (returns : Slice (alloc.vec.Vec Usize)) (constants : C)
    (observations : alloc.vec.Vec (Option (Option Bool))) : Prop :=
  observations.val.length = code.val.length ∧
  ∀ index, (inside : index < observations.val.length) →
    NodeChecked reader front code states returns constants index observations.val[index]

def CertificateChecked {C : Type} (reader : guard_certificate.Constants C)
    (certificate : guard_certificate.Certificate C)
    (observations : alloc.vec.Vec (Option (Option Bool))) : Prop :=
  certificate.code.val.length = certificate.states.val.length ∧
  (∃ facts, certificate.states.val[certificate.entry.val]? = some (some facts) ∧ facts.val = []) ∧
  RowsChecked reader certificate.front certificate.code certificate.states certificate.returns
    certificate.constants observations

theorem successful_effective_guard (known : Option Bool) (guard : Pred) (test : Bool)
    (actual : Pred)
    (tested : (match known with | none => .ok false | some value => if value then .ok true else .ok false : Result Bool) = .ok test)
    (selected : (if test then .ok Pred.TRUE else .ok guard : Result Pred) = .ok actual) :
    actual = effectiveGuard guard known := by
  cases known with
  | none =>
      have equal := Result.ok_injective tested
      subst test
      apply Result.ok_injective
      simpa only [effectiveGuard, Bool.false_eq_true, ite_false] using selected.symm
  | some value =>
      cases value <;> simp only [ite_true, ite_false] at tested
      all_goals
        have equal := Result.ok_injective tested
        subst test
        apply Result.ok_injective
        simpa only [effectiveGuard, Bool.false_eq_true, ite_true, ite_false] using selected.symm

theorem successful_returns (rows : Slice (alloc.vec.Vec Usize)) (pc : Usize)
    (actual : Option (Slice Usize))
    (success : (if pc < Slice.len rows then do
        let row ← Slice.index_usize rows pc
        .ok (some (alloc.vec.Vec.deref row))
      else .ok none : Result (Option (Slice Usize))) = .ok actual) :
    actual = returnsAt rows pc.val := by
  split at success
  · rename_i test
    have inside : pc.val < rows.val.length := by
      simpa only [UScalar.lt_equiv, Slice.len_val, Slice.length] using test
    obtain ⟨row, rowOk, returned⟩ := bind_success _ _ _ success
    have rowAt := Slice.index_usize_spec rows pc inside
    simp only [rowOk, WP.spec_ok] at rowAt
    have equal := Result.ok_injective returned
    subst actual
    simp only [returnsAt, List.getElem?_eq_getElem inside, Option.map_some, rowAt]
  · rename_i test
    have outside : ¬pc.val < rows.val.length := by
      simpa only [UScalar.lt_equiv, Slice.len_val, Slice.length] using test
    have equal := Result.ok_injective success
    subst actual
    simp only [returnsAt, List.getElem?_eq_none (by omega : rows.val.length ≤ pc.val), Option.map_none]


def PrefixChecked {C : Type} (reader : guard_certificate.Constants C)
    (front : Front) (code : Slice Inst) (states : Slice (Option (alloc.vec.Vec Fact)))
    (returns : Slice (alloc.vec.Vec Usize)) (constants : C)
    (observations : alloc.vec.Vec (Option (Option Bool))) (pc : Usize) : Prop :=
  pc.val ≤ code.val.length ∧ observations.val.length = pc.val ∧
  ∀ index, (inside : index < observations.val.length) →
    NodeChecked reader front code states returns constants index observations.val[index]

theorem successful_row_advance {C : Type} (reader : guard_certificate.Constants C)
    (front : Front) (code : Slice Inst) (states : Slice (Option (alloc.vec.Vec Fact)))
    (returns : Slice (alloc.vec.Vec Usize)) (constants : C)
    (observations : alloc.vec.Vec (Option (Option Bool))) (pc : Usize)
    (prior : PrefixChecked reader front code states returns constants observations pc)
    (inside : pc.val < code.val.length) (observation : Option (Option Bool))
    (current : NodeChecked reader front code states returns constants pc.val observation)
    (flow : ControlFlow (alloc.vec.Vec (Option (Option Bool)) × Usize)
      (Option (alloc.vec.Vec (Option (Option Bool)))))
    (success : (do
      let updated ← alloc.vec.Vec.push observations observation
      let next ← pc + 1#usize
      .ok (ControlFlow.cont (updated, next))) = .ok flow) :
    ∃ updated next, flow = .cont (updated, next) ∧
      PrefixChecked reader front code states returns constants updated next := by
  obtain ⟨updated, updatedOk, incrementOk⟩ := bind_success _ _ _ success
  have updatedAt := alloc.vec.Vec.push_spec observations observation (by
    have := code.property
    obtain ⟨_, length, _⟩ := prior
    omega)
  simp only [updatedOk, WP.spec_ok] at updatedAt
  obtain ⟨next, nextOk, returned⟩ := bind_success _ _ _ incrementOk
  have nextAt := Usize.add_spec (x := pc) (y := 1#usize) (by scalar_tac)
  simp only [nextOk, WP.spec_ok] at nextAt
  refine ⟨updated, next, (Result.ok_injective returned).symm, ?_⟩
  obtain ⟨pcBound, rowsLength, priorRows⟩ := prior
  refine ⟨by scalar_tac, ?_, ?_⟩
  · rw [updatedAt, List.length_append, List.length_singleton]
    scalar_tac
  · intro index indexInside
    have updatedLength : updated.val.length = pc.val + 1 := by simp [updatedAt, rowsLength]
    by_cases old : index < observations.val.length
    · simpa only [updatedAt, List.getElem_append_left old] using priorRows index old
    · have currentIndex : index = observations.val.length := by omega
      subst index
      have currentAt : NodeChecked reader front code states returns constants observations.val.length observation := by
        simpa only [rowsLength] using current
      simpa only [updatedAt, List.getElem_append_right (Nat.le_refl observations.val.length), Nat.sub_self,
        List.getElem_singleton] using currentAt

theorem successful_validate_loop {C : Type} (reader : guard_certificate.Constants C)
    (front : Front) (code : Slice Inst) (states : Slice (Option (alloc.vec.Vec Fact)))
    (returns : Slice (alloc.vec.Vec Usize)) (constants : C)
    (sameLength : code.val.length = states.val.length)
    (observations : alloc.vec.Vec (Option (Option Bool)))
    (success : guard_certificate.validate_loop reader front code states returns constants
      (alloc.vec.Vec.new _) 0#usize = .ok (some observations)) :
    RowsChecked reader front code states returns constants observations := by
  unfold guard_certificate.validate_loop at success
  have execution := successful_loop_has_execution _ _ _ success
  have post := execution.preserves
    (fun state : alloc.vec.Vec (Option (Option Bool)) × Usize =>
      PrefixChecked reader front code states returns constants state.1 state.2)
    (fun result : Option (alloc.vec.Vec (Option (Option Bool))) =>
      ∀ produced, result = some produced → RowsChecked reader front code states returns constants produced)
    (by refine ⟨by scalar_tac, by rfl, ?_⟩; intro index inside; simp [alloc.vec.Vec.new] at inside) ?_
  · exact post observations rfl
  rintro ⟨rows, pc⟩ invariant flow bodyOk
  unfold guard_certificate.validate_loop.body at bodyOk
  dsimp only at bodyOk
  split at bodyOk
  · rename_i test
    have inside : pc.val < code.val.length := by
      simpa only [UScalar.lt_equiv, Slice.len_val, Slice.length] using test
    have stateInside : pc.val < states.val.length := by omega
    obtain ⟨present, presentOk, continuedOk⟩ := bind_success _ _ _ bodyOk
    have presentAt := Slice.index_usize_spec states pc stateInside
    simp only [presentOk, WP.spec_ok] at presentAt
    cases present with
    | none =>
        simp only at continuedOk
        have current : NodeChecked reader front code states returns constants pc.val none :=
          .inl ⟨by rw [List.getElem?_eq_getElem stateInside, ←presentAt], rfl⟩
        obtain ⟨updated, next, same, established⟩ := successful_row_advance reader front code states
          returns constants rows pc invariant inside none current flow continuedOk
        subst flow
        exact established
    | some facts =>
        simp only at continuedOk
        obtain ⟨instruction, instructionOk, knownOk⟩ := bind_success _ _ _ continuedOk
        have instructionAt := Slice.index_usize_spec code pc inside
        simp only [instructionOk, WP.spec_ok] at instructionAt
        obtain ⟨known, knownRead, testedOk⟩ := bind_success _ _ _ knownOk
        obtain ⟨tested, testedRead, guardOk⟩ := bind_success _ _ _ testedOk
        obtain ⟨guard, guardRead, returnsOk⟩ := bind_success _ _ _ guardOk
        have guardAt := successful_effective_guard known instruction.guard tested guard testedRead guardRead
        obtain ⟨destinations, destinationsRead, successorsOk⟩ := bind_success _ _ _ returnsOk
        have destinationsAt := successful_returns returns pc destinations destinationsRead
        obtain ⟨targets, targetsRead, checkedOk⟩ := bind_success _ _ _ successorsOk
        obtain ⟨accepted, acceptedRead, advancedOk⟩ := bind_success _ _ _ checkedOk
        cases accepted with
        | false =>
            simp only [Bool.false_eq_true, ite_false] at advancedOk
            have equal := Result.ok_injective advancedOk
            subst flow
            intro produced same
            cases same
        | true =>
            simp only [ite_true] at advancedOk
            have current : NodeChecked reader front code states returns constants pc.val (some known) := by
              apply Or.inr
              refine ⟨facts, instruction, pc, known, targets, ?_, ?_, rfl, rfl, knownRead, ?_, ?_⟩
              · rw [List.getElem?_eq_getElem stateInside, ←presentAt]
              · rw [List.getElem?_eq_getElem inside, ←instructionAt]
              · simpa only [guardAt, destinationsAt] using targetsRead
              · simpa only [alloc.vec.Vec.deref, Slice.from_val] using successful_check_successors reader
                  { front, instruction, facts := alloc.vec.Vec.deref facts, constants }
                  states (alloc.vec.Vec.deref targets) acceptedRead
            obtain ⟨updated, next, same, established⟩ := successful_row_advance reader front code states
              returns constants rows pc invariant inside (some known) current flow advancedOk
            subst flow
            exact established
  · rename_i test
    have stopped : ¬pc.val < code.val.length := by
      simpa only [UScalar.lt_equiv, Slice.len_val, Slice.length] using test
    have equal := Result.ok_injective bodyOk
    subst flow
    intro produced same
    cases same
    obtain ⟨pcBound, rowsLength, priorRows⟩ := invariant
    dsimp only at pcBound rowsLength priorRows
    exact ⟨by omega, priorRows⟩


theorem successful_validate {C : Type} (reader : guard_certificate.Constants C)
    (certificate : guard_certificate.Certificate C)
    (observations : alloc.vec.Vec (Option (Option Bool)))
    (success : guard_certificate.validate reader certificate = .ok (some observations)) :
    CertificateChecked reader certificate observations := by
  unfold guard_certificate.validate at success
  dsimp only at success
  split at success
  · have impossible := Result.ok_injective success
    cases impossible
  · rename_i notDifferent
    have sameCount : Slice.len certificate.code = Slice.len certificate.states := by
      simpa only [bne_iff_ne, not_not] using notDifferent
    have sameLength : certificate.code.val.length = certificate.states.val.length := by
      simpa only [Slice.len_val, Slice.length] using congrArg UScalar.val sameCount
    split at success
    · have impossible := Result.ok_injective success
      cases impossible
    · rename_i notPast
      have entryInside : certificate.entry.val < certificate.states.val.length := by
        simp only [ge_iff_le, UScalar.le_equiv, Slice.len_val, Slice.length] at notPast
        omega
      obtain ⟨present, presentOk, restOk⟩ := bind_success _ _ _ success
      have presentAt := Slice.index_usize_spec certificate.states certificate.entry entryInside
      simp only [presentOk, WP.spec_ok] at presentAt
      cases present with
      | none =>
          simp only at restOk
          have impossible := Result.ok_injective restOk
          cases impossible
      | some facts =>
          simp only [guard_certificate.empty, core.slice.Slice.is_empty, bind_ok] at restOk
          split at restOk
          · rename_i empty
            have factsEmpty : facts.val = [] := by
              apply List.length_eq_zero_iff.mp
              simpa only [alloc.vec.Vec.deref, Slice.from_val, Slice.length, decide_eq_true_eq] using empty
            refine ⟨sameLength, ⟨facts, ?_, factsEmpty⟩, ?_⟩
            · rw [List.getElem?_eq_getElem entryInside, ←presentAt]
            · exact successful_validate_loop reader certificate.front certificate.code certificate.states
                certificate.returns certificate.constants sameLength observations restOk
          · have impossible := Result.ok_injective restOk
            cases impossible

end Instruction.GuardCertificate
