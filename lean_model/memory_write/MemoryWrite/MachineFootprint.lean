/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Code.Funs
import MemoryWrite.TotalPrimitives

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 4000000

namespace MemoryWrite.MachineFootprint

abbrev Range := U8 × U32 × U64 × U64

def joins (last range : Range) : Prop :=
  last.1 = range.1 ∧ last.2.1 = range.2.1 ∧
    range.2.2.1 ≤ last.2.2.2 ∧ last.2.2.1 ≤ range.2.2.2

instance (last range : Range) : Decidable (joins last range) := inferInstanceAs (Decidable
  (last.1 = range.1 ∧ last.2.1 = range.2.1 ∧
    range.2.2.1 ≤ last.2.2.2 ∧ last.2.2.1 ≤ range.2.2.2))

def merged (last range : Range) : Range :=
  (last.1, last.2.1, min last.2.2.1 range.2.2.1, max last.2.2.2 range.2.2.2)

/-- The original last_mut update, stated using a list's final element. -/
def nativeAppend (ranges : List Range) (range : Range) : List Range :=
  match ranges.getLast? with
  | none => ranges ++ [range]
  | some last =>
    if joins last range then ranges.dropLast ++ [merged last range]
    else ranges ++ [range]

def covers (ranges : List Range) (space : U8) (owner : U32) (point : U64) : Prop :=
  ∃ range ∈ ranges, range.1 = space ∧ range.2.1 = owner ∧
    range.2.2.1 ≤ point ∧ point < range.2.2.2

def valid (range : Range) : Prop := range.2.2.1 ≤ range.2.2.2

def lastIndex (ranges : alloc.vec.Vec Range) : Usize :=
  Usize.ofNatCore (ranges.val.length - 1) (by have := ranges.property; scalar_tac)

def appendRef (ranges : alloc.vec.Vec Range) (range : Range) : Result (alloc.vec.Vec Range) :=
  match ranges.val.getLast? with
  | none => alloc.vec.Vec.push ranges range
  | some last =>
    if joins last range then .ok (ranges.set (lastIndex ranges) (merged last range))
    else alloc.vec.Vec.push ranges range

theorem source_append_empty (ranges : alloc.vec.Vec Range) (range : Range)
    (empty : ranges.val = []) :
    machine_footprint.append_range ranges range = alloc.vec.Vec.push ranges range := by
  simp [machine_footprint.append_range, empty]

theorem source_append_snoc (ranges : alloc.vec.Vec Range) (prior : List Range)
    (last range : Range) (shape : ranges.val = prior ++ [last]) :
    machine_footprint.append_range ranges range =
      if joins last range then .ok (ranges.set (lastIndex ranges) (merged last range))
      else alloc.vec.Vec.push ranges range := by
  have positive : (alloc.vec.Vec.len ranges) != 0#usize := by simp [shape]
  have indexValue : (lastIndex ranges).val = prior.length := by simp [lastIndex, shape]
  have subtract : (alloc.vec.Vec.len ranges - 1#usize : Result Usize) = .ok (lastIndex ranges) := by
    simp [HSub.hSub, UScalar.sub, lastIndex, shape]
    simp [Usize.ofNatCore, UScalar.ofNatCore, BitVec.ofFin_eq_ofNat]
  have inside : (lastIndex ranges).val < ranges.val.length := by simp [indexValue, shape]
  have lookup : alloc.vec.Vec.index_usize ranges (lastIndex ranges) = .ok last := by
    simp [alloc.vec.Vec.index_usize, indexValue, shape]
  rcases last with ⟨space, owner, lo, hi⟩
  rcases range with ⟨space', owner', lo', hi'⟩
  simp only [machine_footprint.append_range, positive, ↓reduceIte,
    subtract, bind_ok, alloc.vec.Vec.index_slice_index, lookup]
  simp only [alloc.vec.Vec.index_mut_slice_index, alloc.vec.Vec.index_mut_usize, lookup, bind_ok, lift]
  have minimum : core.cmp.impls.OrdU64.min lo lo' = min lo lo' := by
    unfold core.cmp.impls.OrdU64.min
    by_cases h : lo' < lo
    · simp [h, min_eq_right (le_of_lt h)]
    · simp [h, min_eq_left (le_of_not_gt h)]
  have maximum : core.cmp.impls.OrdU64.max hi hi' = max hi hi' := by
    unfold core.cmp.impls.OrdU64.max
    by_cases h : hi' < hi
    · simp [h, max_eq_left (le_of_lt h)]
    · simp [h, max_eq_right (le_of_not_gt h)]
  simp only [joins, merged]
  by_cases h1 : space = space' <;> by_cases h2 : owner = owner' <;>
    by_cases h3 : lo' ≤ hi <;> by_cases h4 : lo ≤ hi' <;>
      simp [h1, h2, h3, h4, minimum, maximum, -UScalar.le_equiv]

theorem source_append (ranges : alloc.vec.Vec Range) (range : Range) :
    machine_footprint.append_range ranges range = appendRef ranges range := by
  cases h : ranges.val.getLast? with
  | none =>
    have empty : ranges.val = [] := by simpa using h
    simp [appendRef, h, source_append_empty ranges range empty]
  | some last =>
    obtain ⟨prior, shape⟩ := List.getLast?_eq_some_iff.mp h
    simpa [appendRef, h] using source_append_snoc ranges prior last range shape

theorem covers_append (first second : List Range) (space : U8) (owner : U32) (point : U64) :
    covers (first ++ second) space owner point ↔
      covers first space owner point ∨ covers second space owner point := by
  simp only [covers, List.mem_append]
  aesop


theorem merged_coverage (last range : Range) (space : U8) (owner : U32) (point : U64)
    (lastValid : valid last) (rangeValid : valid range) (joined : joins last range) :
    covers [merged last range] space owner point ↔ covers [last, range] space owner point := by
  rcases last with ⟨ls, li, lo, hi⟩
  rcases range with ⟨rs, ri, ro, rh⟩
  rcases joined with ⟨rfl, rfl, overlap, overlap'⟩
  rw [show [(ls, li, lo, hi), (ls, li, ro, rh)] =
    [(ls, li, lo, hi)] ++ [(ls, li, ro, rh)] by rfl, covers_append]
  simp [covers, merged]
  simp only [valid] at lastValid rangeValid
  by_cases hs : ls = space <;> by_cases ho : li = owner <;>
    simp_all only [and_false, false_and, true_and, or_self]
  simp only [min_def]
  split_ifs <;> simp_all [UScalar.le_equiv] <;> omega

theorem native_append_coverage (ranges : List Range) (range : Range)
    (allValid : ∀ old ∈ ranges, valid old) (rangeValid : valid range)
    (space : U8) (owner : U32) (point : U64) :
    covers (nativeAppend ranges range) space owner point ↔
      covers ranges space owner point ∨ covers [range] space owner point := by
  cases h : ranges.getLast? with
  | none =>
    have empty : ranges = [] := by simpa using h
    simp [nativeAppend, empty, covers]
  | some last =>
    obtain ⟨prior, rfl⟩ := List.getLast?_eq_some_iff.mp h
    have lastValid : valid last := allValid last (by simp)
    simp only [nativeAppend, List.getLast?_concat]
    by_cases joined : joins last range
    · simp only [joined, ite_true, List.dropLast_concat, covers_append]
      rw [merged_coverage last range space owner point lastValid rangeValid joined]
      rw [show [last, range] = [last] ++ [range] by rfl, covers_append]; tauto
    · simp only [joined, ite_false, covers_append]

theorem source_append_value (ranges : alloc.vec.Vec Range) (range : Range)
    (room : ranges.val.length < Usize.max) :
    machine_footprint.append_range ranges range ⦃ updated =>
      updated.val = nativeAppend ranges.val range ⦄ := by
  rw [source_append]
  cases h : ranges.val.getLast? with
  | none =>
    have empty : ranges.val = [] := by simpa using h
    simp only [appendRef, h]
    step as ⟨updated, value⟩
    simp [value, nativeAppend, h]
  | some last =>
    obtain ⟨prior, shape⟩ := List.getLast?_eq_some_iff.mp h
    simp only [appendRef, h]
    by_cases joined : joins last range
    · simp only [joined, ite_true, WP.spec_ok, alloc.vec.Vec.set, alloc.vec.Vec.from_val]
      simp [nativeAppend, shape, joined, lastIndex]
    · simp only [joined, ite_false]
      step as ⟨updated, value⟩
      simp [value, nativeAppend, h, joined]

theorem source_append_coverage (ranges : alloc.vec.Vec Range) (range : Range)
    (room : ranges.val.length < Usize.max)
    (allValid : ∀ old ∈ ranges.val, valid old) (rangeValid : valid range)
    (space : U8) (owner : U32) (point : U64) :
    machine_footprint.append_range ranges range ⦃ updated =>
      covers updated.val space owner point ↔
        covers ranges.val space owner point ∨ covers [range] space owner point ⦄ := by
  apply WP.spec_mono (source_append_value ranges range room)
  intro updated value
  rw [value]
  exact native_append_coverage ranges.val range allValid rangeValid space owner point

def omitted (space : U8) : Prop :=
  space = 0#u8 ∨ space = 1#u8 ∨ space = 2#u8 ∨ space = 6#u8 ∨ space = 8#u8

instance (space : U8) : Decidable (omitted space) := inferInstanceAs (Decidable
  (space = 0#u8 ∨ space = 1#u8 ∨ space = 2#u8 ∨ space = 6#u8 ∨ space = 8#u8))

def addRef (fp : machine_footprint.Footprint) (range : Range) (write : Bool) :
    Result machine_footprint.Footprint := do
  if fp.sync_only ∧ omitted range.1 then .ok fp else
    if write then
      let next ← appendRef fp.w range
      .ok { fp with w := next }
    else
      let next ← appendRef fp.r range
      .ok { fp with r := next }

theorem source_add (fp : machine_footprint.Footprint) (space : U8) (owner : U32)
    (lo hi : U64) (write : Bool) :
    machine_footprint.Footprint.add fp space owner lo hi write =
      addRef fp (space, owner, lo, hi) write := by
  cases sync : fp.sync_only <;> cases write <;>
    simp only [machine_footprint.Footprint.add, sync, Bool.false_eq_true,
      ↓reduceIte, bind_ok, source_append, addRef, omitted]
  all_goals split <;> simp_all only [UScalar.eq_equiv]
  all_goals dsimp only [UScalar.val] at *
  all_goals simp_all

theorem source_add_update (fp : machine_footprint.Footprint) (range : Range) (write : Bool)
    (readRoom : fp.r.val.length < Usize.max) (writeRoom : fp.w.val.length < Usize.max) :
    addRef fp range write ⦃ updated =>
      updated.sync_only = fp.sync_only ∧ updated.all = fp.all ∧
      updated.rel = fp.rel ∧ updated.acq = fp.acq ∧
      updated.r.val = (if fp.sync_only ∧ omitted range.1 ∨ write then fp.r.val
        else nativeAppend fp.r.val range) ∧
      updated.w.val = (if fp.sync_only ∧ omitted range.1 ∨ !write then fp.w.val
        else nativeAppend fp.w.val range) ⦄ := by
  by_cases skip : fp.sync_only ∧ omitted range.1
  · simp [addRef, skip]
  · simp only [addRef, skip, ite_false]
    cases write with
    | false =>
      simp only [Bool.false_eq_true, ite_false]
      rw [←source_append]
      step with source_append_value fp.r range readRoom as ⟨next, value⟩
      simp [value]
    | true =>
      simp only [ite_true]
      rw [←source_append]
      step with source_append_value fp.w range writeRoom as ⟨next, value⟩
      simp [value]

end MemoryWrite.MachineFootprint
