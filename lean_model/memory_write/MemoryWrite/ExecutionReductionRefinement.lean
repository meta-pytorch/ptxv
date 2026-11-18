/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.ExecutionReduction

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.ExecutionReduction

@[step]
theorem rust_same_words (before after : Slice U64) :
    execution.same_words before after
      ⦃ result => result = decide (before.val = after.val) ⦄ := by
  unfold execution.same_words
  by_cases hlength : before.val.length = after.val.length
  · have hlen : Slice.len before = Slice.len after := by scalar_tac
    simp only [hlen, bne_self_eq_false, Bool.false_eq_true, ite_false]
    unfold execution.same_words_loop
    apply loop.spec_decr_nat
      (fun index : Usize => before.val.length - index.val)
      (fun index : Usize => index.val ≤ before.val.length ∧
        ∀ position < index.val, before.val[position]? = after.val[position]?)
    · intro index hindex
      obtain ⟨hbound, hprefix⟩ := hindex
      unfold execution.same_words_loop.body
      by_cases hlt : index.val < before.val.length
      · have htest : index < Slice.len before := by scalar_tac
        simp only [htest, ite_true]
        step with Slice.index_usize_spec before index hlt as ⟨left, hleft⟩
        step with Slice.index_usize_spec after index (by scalar_tac) as ⟨right, hright⟩
        simp only [bne_iff_ne]
        split
        · rename_i hne
          have hdiff : before.val ≠ after.val := by
            intro heq
            apply hne
            rw [hleft, hright]
            simp only [heq]
          simp [hdiff]
        · rename_i heq
          have heq : left = right := not_not.mp heq
          step as ⟨next, hnext⟩; try scalar_tac
          refine ⟨?_, ?_, ?_⟩
          · scalar_tac
          · intro position hposition
            by_cases hsame : position = index.val
            · subst position
              rw [List.getElem?_eq_getElem hlt,
                List.getElem?_eq_getElem (by omega : index.val < after.val.length),
                ←hleft, ←hright, heq]
            · exact hprefix position (by scalar_tac)
          · scalar_tac
      · have htest : ¬index < Slice.len before := by scalar_tac
        simp only [htest, ite_false, WP.spec_ok]
        have heq : before.val = after.val := by
          apply List.ext_getElem?
          intro position
          by_cases hp : position < before.val.length
          · exact hprefix position (by omega)
          · simp [hp, ←hlength]
        simp [heq]
    · exact ⟨by scalar_tac, by intro position hposition; scalar_tac⟩
  · have hlen : Slice.len before ≠ Slice.len after := by
      intro h
      apply hlength
      have := congrArg UScalar.val h
      simpa [Slice.len] using this
    have hdiff : before.val ≠ after.val := fun h => hlength (congrArg List.length h)
    simp [hlen, hdiff]

@[step]
theorem rust_same_undefined (before after : Slice (U16 × U64)) :
    execution.same_undefined before after
      ⦃ result => result = decide (before.val = after.val) ⦄ := by
  unfold execution.same_undefined
  by_cases hlength : before.val.length = after.val.length
  · have hlen : Slice.len before = Slice.len after := by scalar_tac
    simp only [hlen, bne_self_eq_false, Bool.false_eq_true, ite_false]
    unfold execution.same_undefined_loop
    apply loop.spec_decr_nat
      (fun index : Usize => before.val.length - index.val)
      (fun index : Usize => index.val ≤ before.val.length ∧
        ∀ position < index.val, before.val[position]? = after.val[position]?)
    · intro index hindex
      obtain ⟨hbound, hprefix⟩ := hindex
      unfold execution.same_undefined_loop.body
      by_cases hlt : index.val < before.val.length
      · have htest : index < Slice.len before := by scalar_tac
        simp only [htest, ite_true]
        step with Slice.index_usize_spec before index hlt as ⟨leftIndex, leftMask, hleft⟩
        step with Slice.index_usize_spec after index (by scalar_tac) as ⟨rightIndex, rightMask, hright⟩
        simp only [bne_iff_ne]
        split
        · rename_i hne
          have hdiff : before.val ≠ after.val := by
            intro heq
            have heqAt : (Prod.mk leftIndex leftMask) = (Prod.mk rightIndex rightMask) := by
              rw [hleft, hright]
              simp only [heq]
            exact hne (congrArg Prod.fst heqAt)
          simp [hdiff]
        · rename_i hfirst
          split
          · rename_i hne
            have hdiff : before.val ≠ after.val := by
              intro heq
              have heqAt : (Prod.mk leftIndex leftMask) = (Prod.mk rightIndex rightMask) := by
                rw [hleft, hright]
                simp only [heq]
              exact hne (congrArg Prod.snd heqAt)
            simp [hdiff]
          · rename_i hsecond
            have heq : (Prod.mk leftIndex leftMask) = (Prod.mk rightIndex rightMask) := by
              have hf : leftIndex = rightIndex := not_not.mp hfirst
              have hs : leftMask = rightMask := not_not.mp hsecond
              rw [hf, hs]
            step as ⟨next, hnext⟩; try scalar_tac
            refine ⟨?_, ?_, ?_⟩
            · scalar_tac
            · intro position hposition
              by_cases hsame : position = index.val
              · subst position
                rw [List.getElem?_eq_getElem hlt,
                  List.getElem?_eq_getElem (by omega : index.val < after.val.length),
                  ←hleft, ←hright, heq]
              · exact hprefix position (by scalar_tac)
            · scalar_tac
      · have htest : ¬index < Slice.len before := by scalar_tac
        simp only [htest, ite_false, WP.spec_ok]
        have heq : before.val = after.val := by
          apply List.ext_getElem?
          intro position
          by_cases hp : position < before.val.length
          · exact hprefix position (by omega)
          · simp [hp, ←hlength]
        simp [heq]
    · exact ⟨by scalar_tac, by intro position hposition; scalar_tac⟩
  · have hlen : Slice.len before ≠ Slice.len after := by
      intro h
      apply hlength
      have := congrArg UScalar.val h
      simpa [Slice.len] using this
    have hdiff : before.val ≠ after.val := fun h => hlength (congrArg List.length h)
    simp [hlen, hdiff]

deriving instance DecidableEq for execution.MemoryByte

@[step]
theorem rust_same_memory (before after : Slice execution.MemoryByte) :
    execution.same_memory before after
      ⦃ result => result = decide (before.val = after.val) ⦄ := by
  unfold execution.same_memory
  by_cases hlength : before.val.length = after.val.length
  · have hlen : Slice.len before = Slice.len after := by scalar_tac
    simp only [hlen, bne_self_eq_false, Bool.false_eq_true, ite_false]
    unfold execution.same_memory_loop
    apply loop.spec_decr_nat
      (fun index : Usize => before.val.length - index.val)
      (fun index : Usize => index.val ≤ before.val.length ∧
        ∀ position < index.val, before.val[position]? = after.val[position]?)
    · intro index hindex
      obtain ⟨hbound, hprefix⟩ := hindex
      unfold execution.same_memory_loop.body
      by_cases hlt : index.val < before.val.length
      · have htest : index < Slice.len before := by scalar_tac
        simp only [htest, ite_true]
        step with Slice.index_usize_spec before index hlt as ⟨left, hleft⟩
        step with Slice.index_usize_spec after index (by scalar_tac) as ⟨right, hright⟩
        rcases left with ⟨leftBits, leftDefined⟩
        rcases right with ⟨rightBits, rightDefined⟩
        simp only [bne_iff_ne]
        split
        · rename_i hne
          have hdiff : before.val ≠ after.val := by
            intro heq
            have heqAt : (execution.MemoryByte.mk leftBits leftDefined) = (execution.MemoryByte.mk rightBits rightDefined) := by
              rw [hleft, hright]
              simp only [heq]
            exact hne (congrArg execution.MemoryByte.bits heqAt)
          simp [hdiff]
        · rename_i hfirst
          split
          · rename_i hne
            have hdiff : before.val ≠ after.val := by
              intro heq
              have heqAt : (execution.MemoryByte.mk leftBits leftDefined) = (execution.MemoryByte.mk rightBits rightDefined) := by
                rw [hleft, hright]
                simp only [heq]
              exact hne (congrArg execution.MemoryByte.defined heqAt)
            simp [hdiff]
          · rename_i hsecond
            have heq : (execution.MemoryByte.mk leftBits leftDefined) = (execution.MemoryByte.mk rightBits rightDefined) := by
              have hf : leftBits = rightBits := not_not.mp hfirst
              have hs : leftDefined = rightDefined := not_not.mp hsecond
              rw [hf, hs]
            step as ⟨next, hnext⟩; try scalar_tac
            refine ⟨?_, ?_, ?_⟩
            · scalar_tac
            · intro position hposition
              by_cases hsame : position = index.val
              · subst position
                rw [List.getElem?_eq_getElem hlt,
                  List.getElem?_eq_getElem (by omega : index.val < after.val.length),
                  ←hleft, ←hright, heq]
              · exact hprefix position (by scalar_tac)
            · scalar_tac
      · have htest : ¬index < Slice.len before := by scalar_tac
        simp only [htest, ite_false, WP.spec_ok]
        have heq : before.val = after.val := by
          apply List.ext_getElem?
          intro position
          by_cases hp : position < before.val.length
          · exact hprefix position (by omega)
          · simp [hp, ←hlength]
        simp [heq]
    · exact ⟨by scalar_tac, by intro position hposition; scalar_tac⟩
  · have hlen : Slice.len before ≠ Slice.len after := by
      intro h
      apply hlength
      have := congrArg UScalar.val h
      simpa [Slice.len] using this
    have hdiff : before.val ≠ after.val := fun h => hlength (congrArg List.length h)
    simp [hlen, hdiff]

theorem rust_same_words_eq (before after : Slice U64) :
    execution.same_words before after = .ok (decide (before.val = after.val)) := by
  obtain ⟨result, hrun, hresult⟩ := spec_returns _ _ (rust_same_words before after)
  simpa only [hresult] using hrun

theorem rust_same_undefined_eq (before after : Slice (U16 × U64)) :
    execution.same_undefined before after = .ok (decide (before.val = after.val)) := by
  obtain ⟨result, hrun, hresult⟩ := spec_returns _ _ (rust_same_undefined before after)
  simpa only [hresult] using hrun

theorem rust_same_memory_eq (before after : Slice execution.MemoryByte) :
    execution.same_memory before after = .ok (decide (before.val = after.val)) := by
  obtain ⟨result, hrun, hresult⟩ := spec_returns _ _ (rust_same_memory before after)
  simpa only [hresult] using hrun

def registerModel (state : execution.RegisterState) : RegisterState :=
  ⟨state.scalar.val, state.predicates.val, state.predicate_defined.val,
    state.scalar_defined.val, state.vector_undefined.val, state.uniform.val,
    state.uniform_defined.val, state.uniform_predicates, state.uniform_predicate_defined⟩

theorem rust_same_registers_eq (before after : execution.RegisterState) :
    execution.same_registers before after =
      .ok (decide (registerModel before = registerModel after)) := by
  unfold execution.same_registers
  simp only [rust_same_words_eq, rust_same_undefined_eq, bind_ok]
  cases before
  cases after
  simp only [registerModel, RegisterState.mk.injEq]
  repeat split <;> simp_all

@[step]
theorem rust_record_store (audit : execution.StutterAudit)
    (before after : Slice execution.MemoryByte) :
    execution.record_store audit before after
      ⦃ next => next.memory_unchanged =
        (audit.memory_unchanged && decide (before.val = after.val)) ⦄ := by
  unfold execution.record_store
  rw [rust_same_memory_eq]
  cases audit with
  | mk unchanged => cases unchanged <;> simp

@[step]
theorem rust_accepts (audit : execution.StutterAudit)
    (before after : execution.RegisterState) :
    execution.accepts audit before after
      ⦃ accepted => accepted =
        (audit.memory_unchanged && decide (registerModel before = registerModel after)) ⦄ := by
  unfold execution.accepts
  rw [rust_same_registers_eq]
  cases audit with
  | mk unchanged => cases unchanged <;> simp

def memoryModel (byte : execution.MemoryByte) : MemoryByte := ⟨byte.bits, byte.defined⟩

private theorem memoryModel_injective : Function.Injective memoryModel := by
  intro first second heq
  cases first
  cases second
  simpa only [memoryModel, MemoryByte.mk.injEq, execution.MemoryByte.mk.injEq] using heq

private theorem memoryListModel_injective : Function.Injective (List.map memoryModel) := by
  intro first second heq
  induction first generalizing second with
  | nil => cases second <;> simp_all
  | cons byte rest ih =>
    cases second with
    | nil => simp at heq
    | cons other others =>
      simp only [List.map_cons, List.cons.injEq] at heq
      exact congrArg₂ List.cons (memoryModel_injective heq.1) (ih heq.2)

abbrev StoreSlices := Slice execution.MemoryByte × Slice execution.MemoryByte

def storeModel (sample : StoreSlices) : StoreSample :=
  ⟨sample.1.val.map memoryModel, sample.2.val.map memoryModel⟩

def auditSourceStores : execution.StutterAudit → List StoreSlices → Result execution.StutterAudit
  | audit, [] => .ok audit
  | audit, (before, after) :: rest => do
      let next ← execution.record_store audit before after
      auditSourceStores next rest

theorem rust_record_store_eq (audit : execution.StutterAudit)
    (before after : Slice execution.MemoryByte) :
    execution.record_store audit before after =
      .ok ⟨audit.memory_unchanged && decide (before.val = after.val)⟩ := by
  unfold execution.record_store
  rw [rust_same_memory_eq]
  cases audit with
  | mk unchanged => cases unchanged <;> simp

theorem source_store_equality (sample : StoreSlices) :
    decide (sample.1.val = sample.2.val) =
      decide ((storeModel sample).current = (storeModel sample).initial) := by
  have heq : sample.1.val = sample.2.val ↔
      (storeModel sample).current = (storeModel sample).initial := by
    constructor
    · intro h
      simp [storeModel, h]
    · intro h
      exact (memoryListModel_injective h).symm
  by_cases h : sample.1.val = sample.2.val
  · simp [h, heq.mp h]
  · have hne : (storeModel sample).current ≠ (storeModel sample).initial :=
      fun equal => h (heq.mpr equal)
    simp [h, hne]

theorem auditSourceStores_eq (audit : execution.StutterAudit) (samples : List StoreSlices) :
    auditSourceStores audit samples =
      .ok ⟨audit.memory_unchanged && auditStores (samples.map storeModel)⟩ := by
  induction samples generalizing audit with
  | nil => simp [auditSourceStores, auditStores]
  | cons sample rest ih =>
    rcases sample with ⟨before, after⟩
    simp only [auditSourceStores, rust_record_store_eq, bind_ok, ih, List.map_cons, auditStores]
    rw [source_store_equality (before, after)]
    simp [Bool.and_assoc]

def checkSourcePoll (before after : execution.RegisterState) (samples : List StoreSlices) :
    Result Bool := do
  let empty ← execution.begin_audit
  let audited ← auditSourceStores empty samples
  execution.accepts audited before after

def evidenceModel (before after : execution.RegisterState) (samples : List StoreSlices) :
    PollEvidence :=
  ⟨registerModel before, registerModel after, samples.map storeModel⟩

theorem checkSourcePoll_eq (before after : execution.RegisterState) (samples : List StoreSlices) :
    checkSourcePoll before after samples = .ok (accepts (evidenceModel before after samples)) := by
  unfold checkSourcePoll execution.begin_audit
  simp only [bind_ok, auditSourceStores_eq, Bool.true_and]
  unfold execution.accepts
  rw [rust_same_registers_eq]
  by_cases hstores : auditStores (samples.map storeModel) = true
  · have hsafe : storesUnchanged (samples.map storeModel) := (auditStores_iff _).mp hstores
    simp [hstores, accepts, evidenceModel, hsafe]
  · have hunsafe : ¬storesUnchanged (samples.map storeModel) := by
      intro h
      exact hstores ((auditStores_iff _).mpr h)
    simp [hstores, accepts, evidenceModel, hunsafe]

theorem rust_checked_poll_preserves_observation {Event : Type}
    (before after : execution.RegisterState) (samples : List StoreSlices)
    (reads priorEvents : List Event)
    (hrun : checkSourcePoll before after samples = .ok true) :
    let segment : PollSegment Event := ⟨evidenceModel before after samples, reads⟩
    executeSegment segment priorEvents = suppressSegment segment priorEvents ∧
      storesUnchanged segment.evidence.stores := by
  rw [checkSourcePoll_eq] at hrun
  have haccept := Result.ok_injective hrun
  exact checked_poll_preserves_observation
    ⟨evidenceModel before after samples, reads⟩ priorEvents haccept

structure SourcePollSegment (Event : Type) where
  before : execution.RegisterState
  after : execution.RegisterState
  stores : List StoreSlices
  reads : List Event

def SourcePollSegment.model {Event : Type} (segment : SourcePollSegment Event) :
    PollSegment Event :=
  ⟨evidenceModel segment.before segment.after segment.stores, segment.reads⟩

def SourceConnected {Event : Type} : RegisterState → List (SourcePollSegment Event) → Prop
  | _, [] => True
  | state, segment :: rest =>
      registerModel segment.before = state ∧
      checkSourcePoll segment.before segment.after segment.stores = .ok true ∧
      SourceConnected (registerModel segment.after) rest

theorem sourceConnected_derives_model {Event : Type}
    (segments : List (SourcePollSegment Event)) (state : RegisterState)
    (h : SourceConnected state segments) :
    Connected state (segments.map SourcePollSegment.model) := by
  induction segments generalizing state with
  | nil => trivial
  | cons segment rest ih =>
    obtain ⟨hbefore, hrun, hrest⟩ := h
    rw [checkSourcePoll_eq] at hrun
    exact ⟨hbefore, Result.ok_injective hrun, ih _ hrest⟩

theorem rust_checked_poll_trace_hides_no_counterexample {Event : Type}
    (segments : List (SourcePollSegment Event)) (state : RegisterState) (events : List Event)
    (h : SourceConnected state segments) (safe : Observation Event → Prop)
    (hreduced : safe (suppressPolls state events (segments.map SourcePollSegment.model))) :
    safe (executePolls state events (segments.map SourcePollSegment.model)) := by
  exact checked_poll_trace_hides_no_counterexample _ state events
    (sourceConnected_derives_model segments state h) safe hreduced

end MemoryWrite.ExecutionReduction
