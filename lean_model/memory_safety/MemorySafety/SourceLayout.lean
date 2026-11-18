/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.SourcePrimitives

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 3000000
set_option Elab.async false

namespace MemorySafety.Source

def RowNoOverlap (base width : U64) (row : memory_safety.Allocation) : Prop :=
  ¬Overlap base.val width.val row.base.val row.bytes.val

@[step]
theorem source_layout_inner (rows : alloc.vec.Vec memory_safety.Allocation)
    (valid : Bool) (limit : Usize) (base width : U64)
    (bound : limit.val ≤ rows.val.length) :
    memory_safety.layout_valid_loop0_loop0 rows valid limit base width 0#usize
      ⦃ result => result = true → valid = true ∧
        ∀ index (inside : index < rows.val.length), index < limit.val →
          RowNoOverlap base width rows.val[index] ⦄ := by
  unfold memory_safety.layout_valid_loop0_loop0
  apply loop.spec_decr_nat
    (fun current : Bool × Usize => limit.val - current.2.val)
    (fun current : Bool × Usize => current.2.val ≤ limit.val ∧
      (current.1 = true → valid = true ∧
        ∀ index (inside : index < rows.val.length), index < current.2.val →
          RowNoOverlap base width rows.val[index]))
  · intro current invariant
    rcases current with ⟨accepted, cursor⟩
    obtain ⟨cursorBound, checkedPrefix⟩ := invariant
    dsimp only at cursorBound checkedPrefix
    unfold memory_safety.layout_valid_loop0_loop0.body
    by_cases within : cursor.val < limit.val
    · have withinSource : cursor < limit := by scalar_tac
      simp only [withinSource, ite_true]
      cases accepted with
      | false => simp
      | true =>
        simp only [ite_true]
        step as ⟨row, rowEq⟩
        step with source_overlaps base width row.base row.bytes as ⟨overlap, overlapEq⟩
        have rowIndex : row = rows.val[cursor.val] := by simpa using rowEq
        cases isOverlap : overlap with
        | false =>
          simp only [Bool.false_eq_true, ite_false, bind_ok]
          step as ⟨next, nextEq⟩
          have nextVal : next.val = cursor.val + 1 := by scalar_tac
          refine ⟨by scalar_tac, ?_, by scalar_tac⟩
          intro _
          refine ⟨(checkedPrefix rfl).1, ?_⟩
          intro index inside before
          by_cases earlier : index < cursor.val
          · exact (checkedPrefix rfl).2 index inside earlier
          · have equalIndex : index = cursor.val := by omega
            subst index
            unfold RowNoOverlap
            rw [←rowIndex]
            intro h
            have := overlapEq.mpr h
            simp [isOverlap] at this
        | true =>
          simp only [ite_true, bind_ok]
          step as ⟨next, nextEq⟩
          refine ⟨by scalar_tac, by simp, by scalar_tac⟩
    · have outsideSource : ¬cursor < limit := by scalar_tac
      simp only [outsideSource, ite_false, WP.spec_ok]
      have equalBound : cursor.val = limit.val := by omega
      simpa [equalBound] using checkedPrefix
  · refine ⟨by scalar_tac, ?_⟩
    intro h
    refine ⟨h, ?_⟩
    intro index inside before
    scalar_tac

def RowSafe (row : memory_safety.Allocation) : Prop :=
  0 < row.bytes.val ∧ row.base.val + row.bytes.val ≤ addressLimit

def LayoutPrefix (rows : List memory_safety.Allocation) (count : Nat) : Prop :=
  (∀ index (inside : index < rows.length), index < count → RowSafe rows[index]) ∧
  (∀ index (inside : index < rows.length) before (earlierInside : before < rows.length),
    index < count → before < index →
      RowNoOverlap rows[index].base rows[index].bytes rows[before])

theorem source_layout_prefix (program : memory_safety.Program) :
    memory_safety.layout_valid program
      ⦃ result => result = true →
        LayoutPrefix program.allocations.val program.allocations.val.length ⦄ := by
  unfold memory_safety.layout_valid memory_safety.layout_valid_loop0
  apply loop.spec_decr_nat
    (fun current : Bool × Usize => program.allocations.val.length - current.2.val)
    (fun current : Bool × Usize => current.2.val ≤ program.allocations.val.length ∧
      (current.1 = true → LayoutPrefix program.allocations.val current.2.val))
  · intro current invariant
    rcases current with ⟨accepted, cursor⟩
    obtain ⟨cursorBound, checkedPrefix⟩ := invariant
    dsimp only at cursorBound checkedPrefix
    unfold memory_safety.layout_valid_loop0.body
    by_cases within : cursor.val < program.allocations.val.length
    · have withinSource : cursor < alloc.vec.Vec.len program.allocations := by scalar_tac
      simp only [withinSource, ite_true]
      cases accepted with
      | false => simp
      | true =>
        simp only [ite_true]
        step as ⟨row, rowEq⟩
        have rowIndex : row = program.allocations.val[cursor.val]'within := rowEq
        have rowCheck :
            (if row.bytes = 0#u64 then ok false else do
              let base ← lift (UScalar.cast .U128 row.base)
              let width ← lift (UScalar.cast .U128 row.bytes)
              let finish ← base + width
              let maximum ← lift (UScalar.cast .U128 core.num.U64.MAX)
              let limit ← maximum + 1#u128
              if finish > limit then ok false else ok true)
              ⦃ result => result = true → RowSafe row ⦄ := by
          split
          · simp
          · step with UScalar.cast_inBounds_spec .U128 row.base (by scalar_tac) as ⟨base, baseEq⟩
            step with UScalar.cast_inBounds_spec .U128 row.bytes (by scalar_tac) as ⟨width, widthEq⟩
            step as ⟨finish, finishEq⟩
            step with UScalar.cast_inBounds_spec .U128 core.num.U64.MAX (by simp [core.num.U64.MAX, U64.rMax]; scalar_tac) as ⟨maximum, maximumEq⟩
            step as ⟨limit, limitEq⟩
            · simp only [U64.rMax] at maximumEq
              scalar_tac
            · split
              · simp
              · simp only [WP.spec_ok, true_implies, RowSafe, addressLimit]
                simp only [U64.rMax] at maximumEq
                scalar_tac
        step with rowCheck as ⟨rowAccepted, rowSafe⟩
        step with source_layout_inner program.allocations rowAccepted cursor row.base row.bytes
          (by omega) as ⟨allAccepted, allChecked⟩
        step as ⟨next, nextEq⟩
        have nextVal : next.val = cursor.val + 1 := by scalar_tac
        refine ⟨by scalar_tac, ?_, by scalar_tac⟩
        intro finalAccepted
        obtain ⟨currentAccepted, previousRows⟩ := allChecked finalAccepted
        have currentSafe := rowSafe currentAccepted
        have oldPrefix := checkedPrefix rfl
        constructor
        · intro index inside before
          by_cases earlier : index < cursor.val
          · exact oldPrefix.1 index inside earlier
          · have equalIndex : index = cursor.val := by omega
            subst index
            simpa [rowIndex] using currentSafe
        · intro index inside before earlierInside checked earlier
          by_cases already : index < cursor.val
          · exact oldPrefix.2 index inside before earlierInside already earlier
          · have equalIndex : index = cursor.val := by omega
            subst index
            simpa [rowIndex] using previousRows before earlierInside earlier
    · have outsideSource : ¬cursor < alloc.vec.Vec.len program.allocations := by scalar_tac
      simp only [outsideSource, ite_false, WP.spec_ok]
      have equalBound : cursor.val = program.allocations.val.length := by omega
      simpa [equalBound] using checkedPrefix
  · refine ⟨by scalar_tac, ?_⟩
    intro _
    constructor
    · intro index inside before
      scalar_tac
    · intro index inside before earlierInside checked earlier
      scalar_tac

theorem overlap_symmetric {a aw b bw : Nat} (overlap : Overlap a aw b bw) :
    Overlap b bw a aw := ⟨overlap.2.1, overlap.1, overlap.2.2.2, overlap.2.2.1⟩

theorem layout_prefix_safe (program : memory_safety.Program)
    (checked : LayoutPrefix program.allocations.val program.allocations.val.length) :
    LayoutSafe (programModel program) := by
  constructor
  · intro allocation member
    obtain ⟨row, rowMember, rfl⟩ := List.mem_map.mp member
    obtain ⟨index, inside, equal⟩ := List.mem_iff_getElem.mp rowMember
    have safe := checked.1 index inside inside
    rw [equal] at safe
    exact safe
  · intro first firstMember second secondMember distinct
    rcases first with ⟨first, i⟩
    rcases second with ⟨second, j⟩
    obtain ⟨_, iBound, firstEq⟩ := List.mem_zipIdx firstMember
    obtain ⟨_, jBound, secondEq⟩ := List.mem_zipIdx secondMember
    simp only [Nat.zero_add, Nat.sub_zero] at iBound jBound firstEq secondEq
    have iInside : i < program.allocations.val.length := by simpa [programModel] using iBound
    have jInside : j < program.allocations.val.length := by simpa [programModel] using jBound
    rw [firstEq, secondEq]
    simp only [programModel, List.getElem_map, allocationModel]
    by_cases before : j < i
    · exact checked.2 i iInside j jInside iInside before
    · have after : i < j := by dsimp only at distinct; omega
      intro conflict
      exact checked.2 j jInside i iInside jInside after (overlap_symmetric conflict)

@[step]
theorem source_layout_valid (program : memory_safety.Program) :
    memory_safety.layout_valid program
      ⦃ result => result = true → LayoutSafe (programModel program) ⦄ := by
  step with source_layout_prefix program as ⟨result, checked⟩
  exact layout_prefix_safe program (checked ‹result = true›)

end MemorySafety.Source
