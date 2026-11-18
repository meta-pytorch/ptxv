/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RustSparseClock

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite.SparseClock

def rawMerge : List (U32 × U32) → List (U32 × U32) → List (U32 × U32)
  | [], right => right
  | left, [] => left
  | first :: left, second :: right =>
    if first.1 = second.1 then
      (first.1, core.cmp.impls.OrdU32.max first.2 second.2) :: rawMerge left right
    else if first.1 < second.1 then
      first :: rawMerge left (second :: right)
    else second :: rawMerge (first :: left) right
termination_by left right => left.length + right.length

@[simp] theorem raw_merge_nil_right (left : List (U32 × U32)) : rawMerge left [] = left := by
  cases left <;> simp [rawMerge]

theorem raw_merge_model (left right : List (U32 × U32)) :
    baseModel (rawMerge left right) = mergeEntries (baseModel left) (baseModel right) := by
  fun_induction rawMerge left right with
  | case1 right => simp [baseModel, mergeEntries]
  | case2 left h => simp [baseModel]
  | case3 first left second right heq ih =>
    have hv : first.1.val = second.1.val := by scalar_tac
    simpa [baseModel, entryModel, mergeEntries, hv] using
      congrArg (List.cons (entryModel (first.1, core.cmp.impls.OrdU32.max first.2 second.2))) ih
  | case4 first left second right hne hlt ih =>
    have hvne : first.1.val ≠ second.1.val := by scalar_tac
    have hvlt : first.1.val < second.1.val := by scalar_tac
    simpa [baseModel, entryModel, mergeEntries, hvne, hvlt] using
      congrArg (List.cons (entryModel first)) ih
  | case5 first left second right hne hnlt ih =>
    have hvne : first.1.val ≠ second.1.val := by scalar_tac
    have hvnlt : ¬first.1.val < second.1.val := by scalar_tac
    simpa [baseModel, entryModel, mergeEntries, hvne, hvnlt] using
      congrArg (List.cons (entryModel second)) ih

def MergePrefix (left right output result : List (U32 × U32)) (i j : Nat) : Prop :=
  i ≤ left.length ∧ j ≤ right.length ∧ result.length ≤ i + j ∧
    result ++ rawMerge (left.drop i) (right.drop j) = output

@[step] theorem rust_merge_loop (left right : Slice (U32 × U32))
    (output : List (U32 × U32)) (result : alloc.vec.Vec (U32 × U32)) (i j : Usize)
    (hsize : left.val.length + right.val.length < Usize.max)
    (hp : MergePrefix left.val right.val output result.val i.val j.val) :
    clock.merge_entries_loop left right result i j ⦃ done => done.val = output ⦄ := by
  unfold clock.merge_entries_loop
  apply loop.spec_decr_nat
    (fun current : alloc.vec.Vec (U32 × U32) × Usize × Usize =>
      left.val.length - current.2.1.val + (right.val.length - current.2.2.val))
    (fun current : alloc.vec.Vec (U32 × U32) × Usize × Usize =>
      MergePrefix left.val right.val output current.1.val current.2.1.val current.2.2.val)
  · intro current hcurrent
    rcases current with ⟨result, i, j⟩
    change MergePrefix left.val right.val output result.val i.val j.val at hcurrent
    obtain ⟨hib, hjb, hlength, htail⟩ := hcurrent
    unfold clock.merge_entries_loop.body
    by_cases hi : i.val < left.val.length
    · have hit : i < Slice.len left := by scalar_tac
      simp only [hit, ite_true]
      by_cases hj : j.val < right.val.length
      · have hjt : j < Slice.len right := by scalar_tac
        simp only [hjt, ite_true]
        step with Slice.index_usize_spec left i hi as ⟨a, hleft⟩
        rcases a with ⟨p, cp⟩
        step with Slice.index_usize_spec right j hj as ⟨b, hright⟩
        rcases b with ⟨q, cq⟩
        change (if p = q then do
            let maximum ← lift (core.cmp.impls.OrdU32.max cp cq)
            let updated ← alloc.vec.Vec.push result (p, maximum)
            let ni ← i + 1#usize
            let nj ← j + 1#usize
            pure (ControlFlow.cont (updated, ni, nj))
          else if p < q then do
            let updated ← alloc.vec.Vec.push result (p, cp)
            let ni ← i + 1#usize
            pure (ControlFlow.cont (updated, ni, j))
          else do
            let updated ← alloc.vec.Vec.push result (q, cq)
            let nj ← j + 1#usize
            pure (ControlFlow.cont (updated, i, nj))) ⦃ _ ⦄
        have hdropL : left.val.drop i.val = (p, cp) :: left.val.drop (i.val + 1) := by
          rw [List.drop_eq_getElem_cons hi, ←hleft]
        have hdropR : right.val.drop j.val = (q, cq) :: right.val.drop (j.val + 1) := by
          rw [List.drop_eq_getElem_cons hj, ←hright]
        have hcapacity : result.val.length < Usize.max := by omega
        by_cases heq : p = q
        · simp only [heq, ite_true]
          step as ⟨maximum, hmaximum⟩
          step with alloc.vec.Vec.push_spec result (q, maximum) hcapacity as ⟨updated, hupdated⟩
          step as ⟨ni, hni⟩
          step as ⟨nj, hnj⟩
          have hniVal : ni.val = i.val + 1 := by scalar_tac
          have hnjVal : nj.val = j.val + 1 := by scalar_tac
          refine ⟨⟨by omega, by omega, ?_, ?_⟩, by omega⟩
          · simp only [hupdated, List.length_append, List.length_singleton]; omega
          · rw [hdropL, hdropR, rawMerge, ite_eq_left heq] at htail
            simpa [hupdated, hniVal, hnjVal, List.append_assoc, hmaximum, heq] using htail
        · simp only [heq, ite_false]
          by_cases hlt : p < q
          · simp only [hlt, ite_true]
            step with alloc.vec.Vec.push_spec result (p, cp) hcapacity as ⟨updated, hupdated⟩
            step as ⟨ni, hni⟩
            have hniVal : ni.val = i.val + 1 := by scalar_tac
            refine ⟨⟨by omega, hjb, ?_, ?_⟩, by omega⟩
            · simp only [hupdated, List.length_append, List.length_singleton]; omega
            · rw [hdropL, hdropR, rawMerge, ite_eq_right heq, ite_eq_left hlt] at htail
              simpa [hupdated, hniVal, hdropR, List.append_assoc] using htail
          · simp only [hlt, ite_false]
            step with alloc.vec.Vec.push_spec result (q, cq) hcapacity as ⟨updated, hupdated⟩
            step as ⟨nj, hnj⟩
            have hnjVal : nj.val = j.val + 1 := by scalar_tac
            refine ⟨⟨hib, by omega, ?_, ?_⟩, by omega⟩
            · simp only [hupdated, List.length_append, List.length_singleton]; omega
            · rw [hdropL, hdropR, rawMerge, ite_eq_right heq, ite_eq_right hlt] at htail
              simpa [hupdated, hnjVal, hdropL, List.append_assoc] using htail
      · have hjt : ¬j < Slice.len right := by scalar_tac
        simp only [hjt, ite_false]
        step with Slice.index_usize_spec left i hi as ⟨a, hleft⟩
        rcases a with ⟨p, cp⟩
        step with alloc.vec.Vec.push_spec result (p, cp) (by omega) as ⟨updated, hupdated⟩
        step as ⟨ni, hni⟩
        have hniVal : ni.val = i.val + 1 := by scalar_tac
        have hdropL : left.val.drop i.val = (p, cp) :: left.val.drop (i.val + 1) := by
          rw [List.drop_eq_getElem_cons hi, ←hleft]
        have hdropR : right.val.drop j.val = [] := List.drop_eq_nil_iff.mpr (by omega)
        refine ⟨⟨by omega, hjb, ?_, ?_⟩, by omega⟩
        · simp only [hupdated, List.length_append, List.length_singleton]; omega
        · simpa [hdropR, hdropL, hupdated, hniVal, List.append_assoc] using htail
    · have hit : ¬i < Slice.len left := by scalar_tac
      simp only [hit, ite_false]
      by_cases hj : j.val < right.val.length
      · have hjt : j < Slice.len right := by scalar_tac
        simp only [hjt, ite_true]
        step with Slice.index_usize_spec right j hj as ⟨b, hright⟩
        rcases b with ⟨q, cq⟩
        step with alloc.vec.Vec.push_spec result (q, cq) (by omega) as ⟨updated, hupdated⟩
        step as ⟨nj, hnj⟩
        have hnjVal : nj.val = j.val + 1 := by scalar_tac
        have hdropL : left.val.drop i.val = [] := List.drop_eq_nil_iff.mpr (by omega)
        have hdropR : right.val.drop j.val = (q, cq) :: right.val.drop (j.val + 1) := by
          rw [List.drop_eq_getElem_cons hj, ←hright]
        refine ⟨⟨hib, by omega, ?_, ?_⟩, by omega⟩
        · simp only [hupdated, List.length_append, List.length_singleton]; omega
        · simpa [hdropL, hdropR, hupdated, hnjVal, rawMerge, List.append_assoc] using htail
      · have hjt : ¬j < Slice.len right := by scalar_tac
        simp only [hjt, ite_false, WP.spec_ok]
        have hdropL : left.val.drop i.val = [] := List.drop_eq_nil_iff.mpr (by omega)
        have hdropR : right.val.drop j.val = [] := List.drop_eq_nil_iff.mpr (by omega)
        simpa [hdropL, hdropR, rawMerge] using htail
  · exact hp

@[step] theorem rust_merge_entries_bounded (left right : Slice (U32 × U32))
    (hsize : left.val.length + right.val.length < Usize.max) :
    clock.merge_entries left right ⦃ result =>
      baseModel result.val = mergeEntries (baseModel left.val) (baseModel right.val) ⦄ := by
  unfold clock.merge_entries
  step as ⟨capacity, hcapacity⟩
  step with rust_merge_loop left right (rawMerge left.val right.val)
    (alloc.vec.Vec.with_capacity (U32 × U32) capacity) 0#usize 0#usize hsize
    (by simp [MergePrefix, alloc.vec.Vec.with_capacity, alloc.vec.Vec.new,
      alloc.vec.Vec.from, alloc.vec.Vec.val, Slice.from_val]) as ⟨result, hresult⟩
  rw [hresult]
  exact raw_merge_model left.val right.val

end MemoryWrite.SparseClock
