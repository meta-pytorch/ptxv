/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.SourceModel
import MemorySafety.Code.Funs
import MemorySafety.Program
import MemoryWrite.TotalPrimitives

open Aeneas Aeneas.Std Result PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 3000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafety.Source

theorem scalar_val_injective {ty : UScalarTy} (left right : UScalar ty) :
    left.val = right.val ↔ left = right :=
  ⟨UScalar.eq_of_val_eq, UScalar.val_eq_of_eq⟩

theorem result_bind_success {α β : Type} (computation : Result α) (next : α → Result β)
    (value : β) (accepted : Aeneas.Std.bind computation next = .ok value) :
    ∃ input, computation = .ok input ∧ next input = .ok value := by
  cases computation with
  | ret input => exact ⟨input, rfl, by simpa only [bind_ok] using accepted⟩
  | vis effect continuation => simp only [bind_vis, vis_not_ok] at accepted
  | div => simp only [bind_div, div_not_ok] at accepted

theorem allM_pure {α : Type} (predicate : α → Bool) (values : List α) :
    List.allM (fun value => (.ok (predicate value) : Result Bool)) values = .ok (values.all predicate) := by
  induction values with
  | nil => rfl
  | cons value values ih =>
    cases found : predicate value with
    | false => simp [List.allM, List.all_cons, ih, found]; rfl
    | true => simp [List.allM, List.all_cons, ih, found]

theorem zipped_model_equality {α β : Type} [DecidableEq β] (model : α → β)
    (left right : List α) :
    (if left.length = right.length then
      (left.zip right).all (fun pair => decide (model pair.1 = model pair.2)) else false) =
      decide (left.map model = right.map model) := by
  induction left generalizing right with
  | nil => cases right <;> simp
  | cons first rest ih =>
    cases right with
    | nil => simp
    | cons other others =>
      by_cases lengthEq : rest.length = others.length
      · have tail := ih others
        simp only [lengthEq, ite_true] at tail
        simp [lengthEq, tail]
      · have different : rest.map model ≠ others.map model := by
          intro equal
          exact lengthEq (by simpa using congrArg List.length equal)
        simp [lengthEq, different]

theorem vector_model_equality {α β : Type} [DecidableEq β]
    (implementation : core.cmp.PartialEq α α) (model : α → β)
    (element : ∀ left right, implementation.eq left right = .ok (decide (model left = model right)))
    (left right : alloc.vec.Vec α) :
    alloc.vec.partial_eq.PartialEqVec.eq implementation left right =
      .ok (decide (left.val.map model = right.val.map model)) := by
  unfold alloc.vec.partial_eq.PartialEqVec.eq
  simp_rw [element]
  rw [allM_pure]
  change (if left.val.length = right.val.length then
      Result.ok ((left.val.zip right.val).all (fun pair => decide (model pair.1 = model pair.2)))
    else Result.ok false) = _
  rw [← apply_ite Result.ok]
  rw [zipped_model_equality]

theorem source_written_equality (left right : memory_safety.WrittenSpan) :
    memory_safety.WrittenSpan.Insts.CoreCmpPartialEqWrittenSpan.eq left right =
      .ok (decide (writtenModel left = writtenModel right)) := by
  rcases left with ⟨address, width, defined⟩
  rcases right with ⟨otherAddress, otherWidth, otherDefined⟩
  by_cases a : address = otherAddress <;> by_cases w : width = otherWidth <;>
    cases defined <;> cases otherDefined <;>
      simp_all [memory_safety.WrittenSpan.Insts.CoreCmpPartialEqWrittenSpan.eq, writtenModel,
        scalar_val_injective]

theorem source_record_equality (left right : memory_safety.AccessRecord) :
    memory_safety.AccessRecord.Insts.CoreCmpPartialEqAccessRecord.eq left right =
      .ok (decide (recordModel left = recordModel right)) := by
  rcases left with ⟨thread, address, width, write⟩
  rcases right with ⟨otherThread, otherAddress, otherWidth, otherWrite⟩
  by_cases t : thread = otherThread <;> by_cases a : address = otherAddress <;>
    by_cases w : width = otherWidth <;> cases write <;> cases otherWrite <;>
      simp_all [memory_safety.AccessRecord.Insts.CoreCmpPartialEqAccessRecord.eq, recordModel,
        scalar_val_injective]

theorem source_state_equality (left right : memory_safety.State) :
    memory_safety.State.Insts.CoreCmpPartialEqState.eq left right =
      .ok (decide (stateModel left = stateModel right)) := by
  have scalarEq (x y : Usize) : core.cmp.PartialEqUsize.eq x y =
      .ok (decide (x.val = y.val)) := by
    change Result.ok (decide (x = y)) = _
    simp only [scalar_val_injective]
  have pcsEq := vector_model_equality core.cmp.PartialEqUsize UScalar.val scalarEq left.pcs right.pcs
  have liveEq := vector_model_equality core.cmp.PartialEqBool id (by intros; rfl) left.live right.live
  have writesEq := vector_model_equality memory_safety.WrittenSpan.Insts.CoreCmpPartialEqWrittenSpan
    writtenModel source_written_equality left.writes right.writes
  have historyEq := vector_model_equality memory_safety.AccessRecord.Insts.CoreCmpPartialEqAccessRecord
    recordModel source_record_equality left.history right.history
  by_cases b : left.bad = right.bad <;>
    by_cases p : left.pcs.val.map UScalar.val = right.pcs.val.map UScalar.val <;>
    by_cases l : left.live.val = right.live.val <;>
    by_cases w : left.writes.val.map writtenModel = right.writes.val.map writtenModel <;>
    by_cases h : left.history.val.map recordModel = right.history.val.map recordModel <;>
    simp [memory_safety.State.Insts.CoreCmpPartialEqState.eq, pcsEq, liveEq, writesEq, historyEq,
      stateModel, b, p, l, w, h]

theorem usize_increment (index : Usize) (bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits) :
    (index + 1#usize : Result Usize) = .ok (Usize.ofNatCore (index.val + 1) bound) := by
  have platformBound : index.val + 1 < 2 ^ System.Platform.numBits := bound
  simp [HAdd.hAdd, UScalar.add, UScalar.tryMk, UScalar.tryMkOpt,
    UScalar.check_bounds, platformBound, Result.ofOption]
  rfl

theorem source_present_loop (states : Slice memory_safety.State) (state : memory_safety.State)
    (found : Bool) (index : Usize) :
    memory_safety.state_present_loop states state found index = .ok
      (found || (states.val.drop index.val).any (fun candidate => decide (stateModel candidate = stateModel state))) := by
  generalize fuelEq : states.val.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing found index with
  | h fuel ih =>
    have unroll : memory_safety.state_present_loop states state found index = (do
        let result ← memory_safety.state_present_loop.body states state found index
        match result with
        | .cont pair => memory_safety.state_present_loop states state pair.1 pair.2
        | .done output => .ok output) := by
      unfold memory_safety.state_present_loop
      rw [loop.eq_def]
      congr 1
      funext result
      cases result with
      | cont pair => cases pair; rfl
      | done output => rfl
    rw [unroll]
    by_cases inside : index.val < states.val.length
    · have bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := Slice.length_ineq states
        scalar_tac
      let next : Usize := Usize.ofNatCore (index.val + 1) bound
      have nextVal : next.val = index.val + 1 := by simp [next]
      have increment := usize_increment index bound
      have lookup : Slice.index_usize states index = .ok states.val[index.val] := by
        simp [Slice.index_usize, Slice.getElem?_Usize_eq, List.getElem?_eq_getElem inside]
      have test : index < Slice.len states := by scalar_tac
      cases found with
      | true => simp [memory_safety.state_present_loop.body, test]
      | false =>
        have body : memory_safety.state_present_loop.body states state false index = .ok
            (.cont (decide (stateModel states.val[index.val] = stateModel state), next)) := by
          simp only [memory_safety.state_present_loop.body, test, ite_true, Bool.false_eq_true,
            ite_false, lookup, bind_ok, source_state_equality]
          split
          · rename_i same
            simp only [increment, bind_ok]
            congr 2
            exact Prod.ext same.symm rfl
          · rename_i different
            simp only [increment, bind_ok]
            congr 2
            exact Prod.ext (Bool.eq_false_iff.mpr different).symm rfl
        rw [body]
        simp only [bind_ok]
        rw [ih (states.val.length - next.val) (by omega) _ next rfl]
        rw [List.drop_eq_getElem_cons inside]
        simp only [List.any_cons, Bool.false_or, nextVal]
        rfl
    · have test : ¬index < Slice.len states := by scalar_tac
      simp [memory_safety.state_present_loop.body, test,
        List.drop_eq_nil_of_le (show states.val.length ≤ index.val by omega)]

theorem source_present (states : Slice memory_safety.State) (state : memory_safety.State) :
    memory_safety.state_present states state =
      .ok (decide (stateModel state ∈ states.val.map stateModel)) := by
  rw [memory_safety.state_present, source_present_loop]
  change Result.ok ((states.val.drop 0).any (fun candidate => decide (stateModel candidate = stateModel state))) = _
  simp only [List.drop_zero]
  congr 1
  apply Bool.eq_iff_iff.mpr
  simp only [decide_eq_true_eq, List.any_eq_true, List.mem_map]

theorem source_closed_loop (certificate : Slice memory_safety.State)
    (next : alloc.vec.Vec memory_safety.State) (closed : Bool) (index : Usize) :
    memory_safety.state_closed_loop certificate next closed index = .ok
      (closed && (next.val.drop index.val).all (fun state =>
        decide (stateModel state ∈ certificate.val.map stateModel))) := by
  generalize fuelEq : next.val.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing closed index with
  | h fuel ih =>
    have unroll : memory_safety.state_closed_loop certificate next closed index = (do
        let result ← memory_safety.state_closed_loop.body certificate next closed index
        match result with
        | .cont pair => memory_safety.state_closed_loop certificate next pair.1 pair.2
        | .done output => .ok output) := by
      unfold memory_safety.state_closed_loop
      rw [loop.eq_def]
      congr 1
      funext result
      cases result with
      | cont pair => cases pair; rfl
      | done output => rfl
    rw [unroll]
    by_cases inside : index.val < next.val.length
    · have bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := alloc.vec.Vec.len_ineq next
        scalar_tac
      let after : Usize := Usize.ofNatCore (index.val + 1) bound
      have afterVal : after.val = index.val + 1 := by simp [after]
      have increment := usize_increment index bound
      have lookup : alloc.vec.Vec.index_usize next index = .ok next.val[index.val] := by
        simp [alloc.vec.Vec.index_usize, List.getElem?_eq_getElem inside]
      have test : index < alloc.vec.Vec.len next := by scalar_tac
      cases closed with
      | false => simp [memory_safety.state_closed_loop.body, test]
      | true =>
        have body : memory_safety.state_closed_loop.body certificate next true index = .ok
            (.cont (decide (stateModel next.val[index.val] ∈ certificate.val.map stateModel), after)) := by
          simp only [memory_safety.state_closed_loop.body, test, ite_true,
            alloc.vec.Vec.index_slice_index, lookup, bind_ok, source_present, increment]
          rfl
        rw [body]
        simp only [bind_ok]
        rw [ih (next.val.length - after.val) (by omega) _ after rfl]
        rw [List.drop_eq_getElem_cons inside]
        simp only [List.all_cons, Bool.true_and, afterVal]
        rfl
    · have test : ¬index < alloc.vec.Vec.len next := by scalar_tac
      simp [memory_safety.state_closed_loop.body, test,
        List.drop_eq_nil_of_le (show next.val.length ≤ index.val by omega)]

def SourceClosed (program : memory_safety.Program) (certificate : Slice memory_safety.State)
    (state : memory_safety.State) : Prop :=
  state.bad = false ∧ ∃ next,
    memory_safety.successors program state = .ok (some next) ∧
    ∀ after ∈ next.val, stateModel after ∈ certificate.val.map stateModel

theorem source_closed_acceptance (program : memory_safety.Program)
    (certificate : Slice memory_safety.State) (state : memory_safety.State)
    (accepted : memory_safety.state_closed program certificate state = .ok true) :
    SourceClosed program certificate state := by
  unfold memory_safety.state_closed at accepted
  cases bad : state.bad with
  | true => simp [bad] at accepted
  | false =>
    simp only [bad, Bool.false_eq_true, ite_false] at accepted
    obtain ⟨result, computed, remaining⟩ := result_bind_success _ _ _ accepted
    cases result with
    | none => simp at remaining
    | some next =>
      refine ⟨bad, next, computed, ?_⟩
      change memory_safety.state_closed_loop certificate next true 0#usize = .ok true at remaining
      rw [source_closed_loop] at remaining
      have all : next.val.all (fun state => decide (stateModel state ∈ certificate.val.map stateModel)) = true := by
        simpa using remaining
      simpa only [List.all_eq_true, decide_eq_true_eq] using all

def checkAll {α : Type} (check : α → Result Bool) : List α → Result Bool
  | [] => .ok true
  | state :: states => do
    let accepted ← check state
    if accepted then checkAll check states else .ok false

theorem checkAll_acceptance {α : Type} (check : α → Result Bool) (states : List α)
    (accepted : checkAll check states = .ok true) :
    ∀ state ∈ states, check state = .ok true := by
  induction states with
  | nil => simp
  | cons state states ih =>
    unfold checkAll at accepted
    obtain ⟨result, checked, remaining⟩ := result_bind_success _ _ _ accepted
    cases result with
    | false => simp at remaining
    | true =>
      simp only [ite_true] at remaining
      intro value member
      cases member with
      | head => exact checked
      | tail _ member => exact ih remaining value member

theorem source_verify_loop (program : memory_safety.Program) (certificate : Slice memory_safety.State)
    (accepted : Bool) (index : Usize) :
    memory_safety.verify_loop program certificate accepted index =
      (if accepted then checkAll (memory_safety.state_closed program certificate)
        (certificate.val.drop index.val) else .ok false) := by
  generalize fuelEq : certificate.val.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing accepted index with
  | h fuel ih =>
    have unroll : memory_safety.verify_loop program certificate accepted index = (do
        let result ← memory_safety.verify_loop.body program certificate accepted index
        match result with
        | .cont pair => memory_safety.verify_loop program certificate pair.1 pair.2
        | .done output => .ok output) := by
      unfold memory_safety.verify_loop
      rw [loop.eq_def]
      congr 1
      funext result
      cases result with
      | cont pair => cases pair; rfl
      | done output => rfl
    rw [unroll]
    by_cases inside : index.val < certificate.val.length
    · have bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := Slice.length_ineq certificate
        scalar_tac
      let next : Usize := Usize.ofNatCore (index.val + 1) bound
      have nextVal : next.val = index.val + 1 := by simp [next]
      have increment := usize_increment index bound
      have lookup : Slice.index_usize certificate index = .ok certificate.val[index.val] := by
        simp [Slice.index_usize, Slice.getElem?_Usize_eq, List.getElem?_eq_getElem inside]
      have test : index < Slice.len certificate := by scalar_tac
      cases accepted with
      | false => simp [memory_safety.verify_loop.body, test]
      | true =>
        have body : memory_safety.verify_loop.body program certificate true index = (do
            let checked ← memory_safety.state_closed program certificate certificate.val[index.val]
            .ok (.cont (checked, next))) := by
          simp only [memory_safety.verify_loop.body, test, ite_true, lookup, bind_ok]
          congr 1
          funext checked
          rw [increment]
          simp only [bind_ok]
          rfl
        rw [body]
        simp only [Std.bind_assoc, bind_ok]
        rw [List.drop_eq_getElem_cons inside]
        simp only [ite_true, checkAll]
        congr 1
        funext checked
        rw [ih (certificate.val.length - next.val) (by omega) checked next rfl]
        simp only [nextVal]
    · have test : ¬index < Slice.len certificate := by scalar_tac
      cases accepted <;>
        simp [memory_safety.verify_loop.body, test, checkAll,
          List.drop_eq_nil_of_le (show certificate.val.length ≤ index.val by omega)]

theorem source_verify_acceptance (program : memory_safety.Program)
    (certificate : Slice memory_safety.State)
    (accepted : memory_safety.verify program certificate = .ok true) :
    (∃ start, memory_safety.initial program = .ok (some start) ∧
      stateModel start ∈ certificate.val.map stateModel) ∧
    ∀ state ∈ certificate.val, SourceClosed program certificate state := by
  unfold memory_safety.verify at accepted
  obtain ⟨result, initialized, remaining⟩ := result_bind_success _ _ _ accepted
  cases result with
  | none => simp at remaining
  | some start =>
    change Aeneas.Std.bind (memory_safety.state_present certificate start) (fun b =>
      if b then memory_safety.verify_loop program certificate true 0#usize else .ok false) = .ok true at remaining
    rw [source_present] at remaining
    simp only [bind_ok] at remaining
    by_cases present : stateModel start ∈ certificate.val.map stateModel
    · rw [ite_eq_left (by simpa using present)] at remaining
      refine ⟨⟨start, initialized, present⟩, ?_⟩
      rw [source_verify_loop] at remaining
      change checkAll (memory_safety.state_closed program certificate) certificate.val = .ok true at remaining
      intro state member
      exact source_closed_acceptance program certificate state (checkAll_acceptance _ _ remaining state member)
    · rw [ite_eq_right (by simpa using present)] at remaining
      simp at remaining

theorem list_model_injective {α β : Type} (model : α → β) (injective : Function.Injective model) :
    Function.Injective (fun values : List α => values.map model) := by
  intro left
  induction left with
  | nil => intro right equal; cases right <;> simp_all
  | cons first rest ih =>
    intro right equal
    cases right with
    | nil => simp at equal
    | cons other others =>
      simp only [List.map_cons, List.cons.injEq] at equal
      exact congrArg₂ List.cons (injective equal.1) (ih equal.2)

theorem written_model_injective : Function.Injective writtenModel := by
  rintro ⟨address, width, defined⟩ ⟨otherAddress, otherWidth, otherDefined⟩ equal
  simp only [writtenModel, WrittenSpan.mk.injEq] at equal
  rcases equal with ⟨addressEq, widthEq, rfl⟩
  exact congrArg₂ (fun a w => memory_safety.WrittenSpan.mk a w defined)
    (UScalar.eq_of_val_eq addressEq) (UScalar.eq_of_val_eq widthEq)

theorem record_model_injective : Function.Injective recordModel := by
  rintro ⟨thread, address, width, write⟩ ⟨otherThread, otherAddress, otherWidth, otherWrite⟩ equal
  simp only [recordModel, AccessRecord.mk.injEq] at equal
  rcases equal with ⟨threadEq, addressEq, widthEq, rfl⟩
  have : thread = otherThread := UScalar.eq_of_val_eq threadEq
  have : address = otherAddress := UScalar.eq_of_val_eq addressEq
  have : width = otherWidth := UScalar.eq_of_val_eq widthEq
  subst_vars
  rfl

theorem state_model_injective : Function.Injective stateModel := by
  intro left right equal
  have fields := State.mk.inj equal
  have pcs : left.pcs = right.pcs := alloc.vec.Vec.ext _ _
    (list_model_injective UScalar.val (fun _ _ => UScalar.eq_of_val_eq) fields.1)
  have live : left.live = right.live := alloc.vec.Vec.ext _ _ fields.2.1
  have writes : left.writes = right.writes := alloc.vec.Vec.ext _ _
    (list_model_injective writtenModel written_model_injective fields.2.2.1)
  have history : left.history = right.history := alloc.vec.Vec.ext _ _
    (list_model_injective recordModel record_model_injective fields.2.2.2.1)
  cases left
  cases right
  simp_all

theorem source_verify_exact_closure (program : memory_safety.Program)
    (certificate : Slice memory_safety.State)
    (accepted : memory_safety.verify program certificate = .ok true) :
    (∃ start, memory_safety.initial program = .ok (some start) ∧ start ∈ certificate.val) ∧
    ∀ state ∈ certificate.val, state.bad = false ∧ ∃ next,
      memory_safety.successors program state = .ok (some next) ∧
      ∀ after ∈ next.val, after ∈ certificate.val := by
  obtain ⟨⟨start, initialized, present⟩, closed⟩ := source_verify_acceptance program certificate accepted
  have rawPresent {state : memory_safety.State}
      (member : stateModel state ∈ certificate.val.map stateModel) : state ∈ certificate.val := by
    obtain ⟨representative, member, equal⟩ := List.mem_map.mp member
    have same : representative = state := state_model_injective equal
    simpa only [same] using member
  refine ⟨⟨start, initialized, rawPresent present⟩, ?_⟩
  intro state member
  obtain ⟨good, next, computed, nextPresent⟩ := closed state member
  exact ⟨good, next, computed, fun after member => rawPresent (nextPresent after member)⟩

inductive SourceReachable (program : memory_safety.Program) : memory_safety.State → Prop
  | initial {state} : memory_safety.initial program = .ok (some state) → SourceReachable program state
  | step {before after next} : SourceReachable program before →
      memory_safety.successors program before = .ok (some next) → after ∈ next.val →
      SourceReachable program after

theorem source_verify_reachable (program : memory_safety.Program)
    (certificate : Slice memory_safety.State)
    (accepted : memory_safety.verify program certificate = .ok true)
    {state : memory_safety.State} (reachable : SourceReachable program state) :
    state ∈ certificate.val ∧ state.bad = false := by
  obtain ⟨⟨start, initialized, initialMember⟩, closed⟩ := source_verify_exact_closure program certificate accepted
  induction reachable with
  | @initial reached other =>
    have same : start = reached := by simpa only [initialized, ok.injEq, Option.some.injEq] using other
    subst_vars
    exact ⟨initialMember, (closed _ initialMember).1⟩
  | @step before after actualNext reachable computed member ih =>
    obtain ⟨_, next, selected, allPresent⟩ := closed _ ih.1
    have same : next = actualNext := by simpa only [selected, ok.injEq, Option.some.injEq] using computed
    subst_vars
    have present := allPresent _ member
    exact ⟨present, (closed _ present).1⟩

end MemorySafety.Source
