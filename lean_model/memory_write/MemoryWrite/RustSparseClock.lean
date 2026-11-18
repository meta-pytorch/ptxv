/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.SparseClock
import MemoryWrite.RustCapture

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite.SparseClock

def entryModel (entry : U32 × U32) : Entry := (entry.1.val, entry.2.val)

def baseModel (base : List (U32 × U32)) : List Entry := base.map entryModel

@[simp] theorem baseModel_length (base : List (U32 × U32)) :
    (baseModel base).length = base.length := by simp [baseModel]

@[simp] theorem baseModel_getElem (base : List (U32 × U32)) (i : Nat)
    (hi : i < base.length) : (baseModel base)[i]'(by simpa using hi) = entryModel base[i] := by
  simp [baseModel]

def SearchWindow (base : List Entry) (actor low high : Nat) : Prop :=
  low ≤ high ∧ high ≤ base.length ∧
    (∀ k (hk : k < base.length), k < low → base[k].1 < actor) ∧
    (∀ k (hk : k < base.length), high ≤ k → actor < base[k].1)

@[step] theorem rust_locate_loop (base : Slice (U32 × U32)) (actor : U32)
    (low high : Usize) (hs : Sorted (baseModel base.val))
    (hw : SearchWindow (baseModel base.val) actor.val low.val high.val) :
    clock.locate_loop base actor low high ⦃ result =>
      Located (baseModel base.val) actor.val result.1.val result.2 ⦄ := by
  unfold clock.locate_loop
  apply loop.spec_decr_nat
    (fun current : Usize × Usize => current.2.val - current.1.val)
    (fun current : Usize × Usize =>
      SearchWindow (baseModel base.val) actor.val current.1.val current.2.val)
  · intro current hcurrent
    rcases current with ⟨low, high⟩
    change SearchWindow (baseModel base.val) actor.val low.val high.val at hcurrent
    obtain ⟨horder, hhigh, hbefore, hafter⟩ := hcurrent
    unfold clock.locate_loop.body
    by_cases hlt : low.val < high.val
    · have htest : low < high := by scalar_tac
      simp only [htest, ite_true]
      step as ⟨difference, hdifference⟩ ; try scalar_tac
      step as ⟨half, hhalf⟩ ; try scalar_tac
      have hdiff : difference.val = high.val - low.val := by scalar_tac
      have hhalfVal : half.val = (high.val - low.val) / 2 := by scalar_tac
      have hhalfLt : half.val < high.val - low.val := by
        rw [hhalfVal]
        exact Nat.div_lt_self (by omega) (by omega)
      step as ⟨middle, hmiddle⟩ ; try scalar_tac
      have hm : middle.val = low.val + half.val := by scalar_tac
      have hmhigh : middle.val < high.val := by omega
      have hmbound : middle.val < base.val.length := by simp only [baseModel_length] at hhigh; omega
      have hmidmodel : ((baseModel base.val)[middle.val]'(by simpa using hmbound)).1 =
          base.val[middle.val].1.val :=
        congrArg Prod.fst (baseModel_getElem base.val middle.val hmbound)
      step with Slice.index_usize_spec base middle hmbound as ⟨key, stamp, hentry⟩
      have hkey : key = base.val[middle.val].1 := congrArg Prod.fst hentry
      rw [hkey]
      by_cases heq : base.val[middle.val].1 = actor
      · simp only [heq, ite_true, WP.spec_ok]
        refine ⟨by omega, ?_, by simpa using hmbound, ?_, ?_⟩
        · intro k hk hki
          have hkLt := sorted_index_lt (baseModel base.val) hs k middle.val hk
            (by simpa using hmbound) hki
          simpa only [hmidmodel, heq] using hkLt
        · simp only [hmidmodel, heq]
        · intro k hk hki
          have hkLt := sorted_index_lt (baseModel base.val) hs middle.val k
            (by simpa using hmbound) hk hki
          simpa only [hmidmodel, heq] using hkLt
      · simp only [heq, ite_false]
        by_cases hless : base.val[middle.val].1 < actor
        · simp only [hless, ite_true]
          step as ⟨next, hnext⟩ ; try scalar_tac
          have hn : next.val = middle.val + 1 := by scalar_tac
          refine ⟨⟨by omega, hhigh, ?_, hafter⟩, by omega⟩
          intro k hk hki
          by_cases hkm : k = middle.val
          · subst k
            rw [hmidmodel]; scalar_tac
          · have hkLt := sorted_index_lt (baseModel base.val) hs k middle.val hk
              (by simpa using hmbound) (by omega)
            have hlessVal : base.val[middle.val].1.val < actor.val := by scalar_tac
            rw [hmidmodel] at hkLt
            omega
        · simp only [hless, ite_false, WP.spec_ok]
          refine ⟨⟨by omega, by omega, hbefore, ?_⟩, by omega⟩
          intro k hk hki
          have hgreater : actor.val < base.val[middle.val].1.val := by scalar_tac
          by_cases hkm : k = middle.val
          · subst k
            rw [hmidmodel]; exact hgreater
          · have hkLt := sorted_index_lt (baseModel base.val) hs middle.val k
              (by simpa using hmbound) hk (by omega)
            rw [hmidmodel] at hkLt
            omega
    · have htest : ¬low < high := by scalar_tac
      simp only [htest, ite_false, WP.spec_ok]
      refine ⟨by omega, hbefore, ?_⟩
      intro k hk hki
      exact hafter k hk (by omega)
  · exact hw

@[step] theorem rust_locate (base : Slice (U32 × U32)) (actor : U32)
    (hs : Sorted (baseModel base.val)) :
    clock.locate base actor ⦃ result =>
      Located (baseModel base.val) actor.val result.1.val result.2 ⦄ := by
  unfold clock.locate
  apply rust_locate_loop base actor 0#usize (Slice.len base) hs
  refine ⟨?_, ?_, ?_, ?_⟩
  · scalar_tac
  · simp [Slice.len]
  · intro k hk hki; scalar_tac
  · intro k hk hki
    simp [Slice.len] at hki
    simpa using (show False from (by simpa using hk : k < base.val.length).not_ge hki)

@[step] theorem rust_lookup (base : Slice (U32 × U32)) (actor : U32)
    (hs : Sorted (baseModel base.val)) :
    clock.lookup base actor ⦃ result => result.val = entriesValue (baseModel base.val) actor.val ⦄ := by
  unfold clock.lookup
  step with rust_locate base actor hs as ⟨index, found, hlocated⟩
  have hv := located_value (baseModel base.val) actor.val index.val found hlocated
  cases found with
  | false => simp only [Bool.false_eq_true, ite_false, WP.spec_ok]; simpa using hv.symm
  | true =>
    obtain ⟨hi, _, _⟩ := hlocated.2.2
    have hib : index.val < base.val.length := by simpa using hi
    simp only [ite_true]
    step with Slice.index_usize_spec base index hib as ⟨key, stamp, hread⟩
    have hstamp : stamp = base.val[index.val].2 := congrArg Prod.snd hread
    rw [hstamp, hv]
    simp only [ite_true, List.getElem?_eq_getElem hi, Option.getD_some]
    exact (congrArg Prod.snd (baseModel_getElem base.val index.val hib)).symm

@[step] theorem rust_get (base : Slice (U32 × U32)) (own : U32 × U32) (actor : U32)
    (hs : Sorted (baseModel base.val)) :
    clock.get base own actor ⦃ result =>
      result.val = value (baseModel base.val) (entryModel own) actor.val ⦄ := by
  rcases own with ⟨owner, time⟩
  unfold clock.get
  step with rust_lookup base actor hs as ⟨result, hresult⟩
  by_cases heq : owner = actor
  · simp only [heq, ite_true, WP.spec_ok]
    simp [value, entryModel, hresult]
  · simp only [heq, ite_false, WP.spec_ok]
    have hv : owner.val ≠ actor.val := by intro h; apply heq; scalar_tac
    simp [value, entryModel, hv, hresult]

@[step] theorem rust_empty : clock.empty ⦃ result =>
    result.1.val = [] ∧ result.2 = (0#u32, 0#u32) ⦄ := by
  simp [clock.empty, alloc.vec.Vec.new, alloc.vec.Vec.from, alloc.vec.Vec.val, Slice.from_val]

theorem source_empty_projection :
    ∃ base own, clock.empty = .ok (base, own) ∧
      value (baseModel base.val) (entryModel own) = fun _ => 0 := by
  obtain ⟨⟨base, own⟩, hrun, hbase, hown⟩ := spec_returns _ _ rust_empty
  refine ⟨base, own, hrun, ?_⟩
  change own = (0#u32, 0#u32) at hown
  rw [hown]
  funext actor
  simp [hbase, baseModel, entryModel, value, entriesValue]

theorem baseModel_modify_own (base : List (U32 × U32)) (own : U32 × U32) (index : Nat) :
    baseModel (base.modify index (fun e => (e.1, core.cmp.impls.OrdU32.max e.2 own.2))) =
      (baseModel base).modify index (fun e => (e.1, max e.2 own.2.val)) := by
  induction base generalizing index with
  | nil => simp [baseModel]
  | cons head tail ih =>
    cases index with
    | zero => simp [baseModel, entryModel]
    | succ index => simpa [baseModel] using congrArg (List.cons (entryModel head)) (ih index)

@[step] theorem rust_overlay_entry_bounded (result : alloc.vec.Vec (U32 × U32))
    (owner time : U32) (hs : Sorted (baseModel result.val)) (hsize : result.val.length < Usize.max) :
    clock.overlay_entry result owner time ⦃ updated =>
      baseModel updated.val = overlay (baseModel result.val) (entryModel (owner, time)) ⦄ := by
  let base : Slice (U32 × U32) := alloc.vec.Vec.deref result
  have hresult : result.val = base.val := by simp only [base, alloc.vec.Vec.deref, Slice.from_val]
  have hs : Sorted (baseModel base.val) := by simpa only [hresult] using hs
  have hsize : base.val.length < Usize.max := by simpa only [hresult] using hsize
  rw [hresult]
  unfold clock.overlay_entry
  by_cases hzero : time = 0#u32
  · simp only [hzero, bne_self_eq_false, Bool.false_eq_true, ite_false, WP.spec_ok]
    simp [overlay, entryModel, hresult]
  · have hz : time.val ≠ 0 := by intro h; apply hzero; scalar_tac
    simp only [hzero, bne_iff_ne, ne_eq, not_false_eq_true, ite_true]
    step with rust_locate base owner hs as ⟨index, found, hlocated⟩
    have hover := merge_singleton_located (baseModel base.val) (entryModel (owner, time))
      index.val found hlocated
    have hov : overlay (baseModel base.val) (entryModel (owner, time)) =
        installOwn (baseModel base.val) (entryModel (owner, time)) index.val found := by
      simpa only [overlay, entryModel, hz, ite_false] using hover
    rw [hov]
    cases found with
    | true =>
      obtain ⟨hi, _, _⟩ := hlocated.2.2
      have hir : index.val < result.val.length := by simpa [hresult] using hi
      simp only [ite_true]
      step as ⟨key, stamp, hread⟩
      step as ⟨maximum, hmaximum⟩
      step as ⟨keyAgain, stampAgain, back, hreadAgain, hback⟩
      rw [hback]
      simp only [alloc.vec.Vec.set, alloc.vec.Vec.from, alloc.vec.Vec.val,
        Slice.from_val]
      change baseModel (result.val.set index.val (keyAgain, maximum)) = _
      have hmodify := baseModel_modify_own result.val (owner, time) index.val
      rw [List.modify_eq_set] at hmodify
      simp only [List.getElem?_eq_getElem hir, Option.getD_some] at hmodify
      have hk : keyAgain = result.val[index.val].1 := congrArg Prod.fst hreadAgain
      have ht : stamp = result.val[index.val].2 := congrArg Prod.snd hread
      have hmmax : maximum = core.cmp.impls.OrdU32.max stamp time := by scalar_tac
      rw [hmmax, hk, ht]
      simpa only [installOwn, ite_true, hresult, entryModel] using hmodify
    | false =>
      have hi : index.val ≤ result.val.length := by simpa [hresult] using hlocated.1
      simp only [Bool.false_eq_true, ite_false]
      by_cases hend : index.val = result.val.length
      · have htest : index = alloc.vec.Vec.len result := by scalar_tac
        simp only [htest, ite_true]
        step with alloc.vec.Vec.push_spec result (owner, time) (by simpa [hresult] using hsize)
          as ⟨updated, hupdated⟩
        rw [hupdated]
        simp only [installOwn, Bool.false_eq_true, ite_false, baseModel, List.map_append,
          List.map_cons, List.map_nil, hresult]
        have hm : index.val = (base.val.map entryModel).length := by
          simpa only [List.length_map, hresult] using hend
        rw [←htest, hm, List.insertIdx_length_self]
      · have htest : ¬index = alloc.vec.Vec.len result := by intro h; apply hend; scalar_tac
        simp only [htest, ite_false]
        step with alloc.vec.Vec.insert_spec result index (owner, time) (by change index.val < result.val.length; omega)
          (by simpa [hresult] using hsize) as ⟨updated, hupdated⟩
        simp [hupdated, installOwn, baseModel, List.map_insertIdx, hresult, entryModel]


@[step] theorem rust_entries_bounded (base : Slice (U32 × U32)) (own : U32 × U32)
    (hs : Sorted (baseModel base.val)) (hsize : base.val.length < Usize.max) :
    clock.entries base own ⦃ result =>
      baseModel result.val = overlay (baseModel base.val) (entryModel own) ⦄ := by
  rcases own with ⟨owner, time⟩
  unfold clock.entries
  step with alloc.slice.Slice.to_vec_spec (BuiltinClone (U32 × U32)) base
    (by intro x hx; rfl) as ⟨result, hclone⟩
  have hresult : result.val = base.val := congrArg Slice.val hclone.symm
  step with rust_overlay_entry_bounded result owner time (by simpa only [hresult] using hs)
      (by simpa only [hresult] using hsize) as ⟨updated, projected⟩
  simpa only [hresult] using projected

end MemoryWrite.SparseClock
