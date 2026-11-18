/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Storage

open Aeneas Aeneas.Std Result ControlFlow PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace MemoryWrite

@[step]
theorem rust_same_key_spec (left right : Key) :
    checked.same_key left right ⦃ result => result = true ↔ left = right ⦄ := by
  cases left with
  | mk ls la lo =>
    cases right with
    | mk rs ra ro =>
      unfold checked.same_key
      split
      · split <;> simp_all
      · simp_all

def findCellModel (cells : Slice Cell) (key : Key) : Option Cell :=
  cells.val.find? (fun cell => decide (cell.key = key))

theorem findCellModel_some (cells : Slice Cell) (key : Key) (cell : Cell)
    (hfind : findCellModel cells key = some cell) : cell ∈ cells.val ∧ cell.key = key := by
  exact ⟨List.mem_of_find?_eq_some hfind, by simpa using List.find?_some hfind⟩

theorem findCellModel_none (cells : Slice Cell) (key : Key) :
    findCellModel cells key = none ↔ ∀ cell ∈ cells.val, cell.key ≠ key := by
  simp [findCellModel, List.find?_eq_none]

theorem key_eq_of_location_eq (left right : Key) (heq : location left = location right) : left = right := by
  cases left with
  | mk ls la lo =>
    cases right with
    | mk rs ra ro =>
      simp only [location, Nat.add_zero, Address.mk.injEq] at heq
      obtain ⟨rfl, rfl, hoffset⟩ := heq
      have : lo = ro := by scalar_tac
      subst ro
      rfl

theorem validSpan_findCellModel_some (cells : Slice Cell) (start : Key) (width : Usize) (index : Nat)
    (hspan : ValidWriteSpan cells start width.val) (hindex : index < width.val) :
    ∃ cell, findCellModel cells (Storage.offsetKey start index) = some cell := by
  obtain ⟨cell, hcell, hloc⟩ := hspan.coverage index hindex
  have hkey : cell.key = Storage.offsetKey start index :=
    key_eq_of_location_eq cell.key _
      (hloc.trans (Storage.offsetKey_location start index (by have := hspan.no_wrap; omega)).symm)
  cases hfind : findCellModel cells (Storage.offsetKey start index) with
  | none => exact False.elim ((findCellModel_none cells _).mp hfind cell hcell hkey)
  | some found => exact ⟨found, rfl⟩

theorem drop_cons_getElem {α : Type} (values : List α) (index : Nat) (hindex : index < values.length) :
    values.drop index = values[index] :: values.drop (index + 1) := by
  apply List.ext_getElem?
  intro position
  cases position with
  | zero => simp [List.getElem?_drop, List.getElem?_eq_getElem hindex]
  | succ position => simp [List.getElem?_drop]

@[step]
theorem rust_find_cell_spec (cells : Slice Cell) (key : Key) :
    checked.find_cell cells key ⦃ result => result = findCellModel cells key ⦄ := by
  unfold checked.find_cell checked.find_cell_loop
  apply loop.spec_decr_nat
    (fun cursor : Usize => cells.val.length - cursor.val)
    (fun cursor : Usize => cursor.val ≤ cells.val.length ∧
      (cells.val.drop cursor.val).find? (fun cell => decide (cell.key = key)) = findCellModel cells key)
  · intro cursor hcursor
    obtain ⟨hbound, hsearch⟩ := hcursor
    unfold checked.find_cell_loop.body
    by_cases hlt : cursor.val < cells.val.length
    · have htest : cursor < Slice.len cells := by scalar_tac
      simp only [htest, ite_true]
      step with Slice.index_usize_spec cells cursor hlt as ⟨cell, hcell⟩
      step with rust_same_key_spec cell.key key as ⟨matched, hmatched⟩
      have hdrop : cells.val.drop cursor.val = cell :: cells.val.drop (cursor.val + 1) := by
        simpa only [hcell] using drop_cons_getElem cells.val cursor.val hlt
      by_cases heq : cell.key = key
      · have htrue : matched = true := hmatched.mpr heq
        simp only [htrue, ite_true, WP.spec_ok]
        rw [hdrop] at hsearch
        simpa only [List.find?_cons, heq, decide_true] using hsearch
      · have hfalse : matched = false := by simpa using (show matched ≠ true from fun h => heq (hmatched.mp h))
        simp only [hfalse, Bool.false_eq_true, ite_false]
        step as ⟨next, hnext⟩
        have hnextVal : next.val = cursor.val + 1 := by scalar_tac
        refine ⟨by omega, ?_, by omega⟩
        rw [hdrop] at hsearch
        simpa only [hnextVal, List.find?_cons, heq, decide_false] using hsearch
    · have htest : ¬cursor < Slice.len cells := by scalar_tac
      simp only [htest, ite_false, WP.spec_ok]
      have hdrop : cells.val.drop cursor.val = [] := List.drop_eq_nil_of_le (by omega)
      simpa only [hdrop, List.find?_nil] using hsearch
  · exact ⟨by scalar_tac, by simp [findCellModel]⟩

def SpanCoverage (cells : Slice Cell) (start : Key) (width : Nat) : Prop :=
  ∀ index, index < width → ∃ cell ∈ cells.val, location cell.key = location start index

theorem rust_span_fits_loop_spec (cells : Slice Cell) (start : Key) (width : Usize)
    (hwrap : start.offset.val + width.val ≤ U64.max + 1) :
    checked.span_fits_loop cells start.space start.allocation start.offset width 0#usize
      ⦃ result => result = true ↔ SpanCoverage cells start width.val ⦄ := by
  unfold checked.span_fits_loop
  apply loop.spec_decr_nat
    (fun cursor : Usize => width.val - cursor.val)
    (fun cursor : Usize => cursor.val ≤ width.val ∧ SpanCoverage cells start cursor.val)
  · intro cursor hcursor
    obtain ⟨hbound, hprefix⟩ := hcursor
    unfold checked.span_fits_loop.body
    by_cases hlt : cursor.val < width.val
    · have htest : cursor < width := by scalar_tac
      simp only [htest, ite_true]
      have hcastBound : cursor.val ≤ UScalar.max .U64 := by
        have := Usize.bounds_eq
        scalar_tac
      step with UScalar.cast_inBounds_spec .U64 cursor hcastBound as ⟨delta, hdelta⟩
      step as ⟨offset, hoffset⟩
      have hlocation : location (Key.mk start.space start.allocation offset) = location start cursor.val := by
        simp only [location, Nat.add_zero, Address.mk.injEq]
        exact ⟨trivial, trivial, by scalar_tac⟩
      step with rust_find_cell_spec cells ⟨start.space, start.allocation, offset⟩ as ⟨found, hfound⟩
      cases found with
      | none =>
        change (ok (done false) : Result (ControlFlow Usize Bool)) ⦃ output => _ ⦄
        simp only [WP.spec_ok, Bool.false_eq_true, false_iff]
        intro hcoverage
        obtain ⟨cell, hcell, hloc⟩ := hcoverage cursor.val hlt
        have hkey : cell.key = ⟨start.space, start.allocation, offset⟩ :=
          key_eq_of_location_eq cell.key _ (hloc.trans hlocation.symm)
        exact (findCellModel_none cells _).mp hfound.symm cell hcell hkey
      | some cell =>
        have hcell := findCellModel_some cells _ cell hfound.symm
        change (do let next ← cursor + 1#usize; ok (cont next) : Result (ControlFlow Usize Bool))
          ⦃ output => _ ⦄
        step as ⟨next, hnext⟩
        have hnextVal : next.val = cursor.val + 1 := by scalar_tac
        refine ⟨by omega, ?_, by omega⟩
        intro index hindex
        by_cases hbefore : index < cursor.val
        · exact hprefix index hbefore
        · have heq : index = cursor.val := by omega
          subst index
          exact ⟨cell, hcell.1, by simpa only [hcell.2] using hlocation⟩
    · have htest : ¬cursor < width := by scalar_tac
      simp only [htest, ite_false, WP.spec_ok, true_iff]
      intro index hindex
      exact hprefix index (by omega)
  · refine ⟨by scalar_tac, ?_⟩
    intro index hindex
    simp at hindex

@[step]
theorem rust_span_fits_spec (cells : Slice Cell) (start : Key) (width : Usize) :
    checked.span_fits cells start width ⦃ result => result = true ↔ ValidWriteSpan cells start width.val ⦄ := by
  unfold checked.span_fits
  have hwidthBound : width.val ≤ U64.max := by
    have := Usize.bounds_eq
    scalar_tac
  have hmaximumValue : core.num.U64.MAX.val = U64.max := by
    simp [core.num.U64.MAX, U64.max_eq, U64.rMax]
  step with UScalar.cast_inBounds_spec .U128 start.offset (by scalar_tac) as ⟨base, hbase⟩
  step with UScalar.cast_inBounds_spec .U128 width (by scalar_tac) as ⟨count, hcount⟩
  step as ⟨stop, hstop⟩
  step with UScalar.cast_inBounds_spec .U128 core.num.U64.MAX
    (by rw [hmaximumValue]; scalar_tac) as ⟨maximum, hmaximum⟩
  step as ⟨limit, hlimit⟩
  split
  · rename_i hoverflow
    simp only [WP.spec_ok, Bool.false_eq_true, false_iff]
    intro hspan
    have := hspan.no_wrap
    scalar_tac
  · rename_i hfits
    have hwrap : start.offset.val + width.val ≤ U64.max + 1 := by scalar_tac
    step with rust_span_fits_loop_spec cells start width hwrap as ⟨result, hresult⟩
    constructor
    · intro htrue
      exact ⟨hwrap, hresult.mp htrue⟩
    · intro hspan
      exact hresult.mpr hspan.coverage

end MemoryWrite
