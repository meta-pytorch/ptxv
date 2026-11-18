/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RegisterIndexModel
import MemoryWrite.MayWriteStorage

open Aeneas Aeneas.Std Result PtxvMemoryWrite
open MemoryWrite.MayWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.RegisterIndex

@[step]
theorem rust_get (index : register_index.RegisterIndex) (key : symbolic.RegisterKey)
    (valid : Directory index.pages.val) :
    register_index.RegisterIndex.get index key ⦃ result =>
      ∀ id, result = some id ↔ Stored index key id ⦄ := by
  unfold register_index.RegisterIndex.get
  step with rust_page_key key as ⟨page, pageValue⟩
  step with rust_page_position (alloc.vec.Vec.deref index.pages) page
    (by simpa only [alloc.vec.Vec.deref, Slice.from_val] using directory_ordered index.pages.val valid)
    as ⟨position, positionSpec⟩
  simp only [alloc.vec.Vec.deref, Slice.from_val] at positionSpec
  by_cases inside : position.val < index.pages.val.length
  · have test : position < alloc.vec.Vec.len index.pages := by scalar_tac
    simp only [test, ite_true]
    step with alloc.vec.Vec.index_usize_spec index.pages position inside as ⟨row, rowAt⟩
    have rowEqual : row = index.pages.val[position.val]! := by
      rw [rowAt, List.Inhabited_getElem_eq_getElem! _ _ inside]
    by_cases matched : row.key = page
    · simp only [matched, ite_true]
      step as ⟨remainder, remainderValue⟩
      simp only [lift, bind_ok]
      have offset : (core.convert.num.FromUsizeU16.from remainder).val = keyOffset key := by
        dsimp only [keyOffset]
        scalar_tac
      have rowSize : row.slots.val.length = 64 := by rw [rowEqual]; exact valid.1 position.val inside
      have offsetBound : (core.convert.num.FromUsizeU16.from remainder).val < row.slots.val.length := by
        rw [offset, rowSize]; exact Nat.mod_lt _ (by decide)
      step with alloc.vec.Vec.index_usize_spec row.slots (core.convert.num.FromUsizeU16.from remainder) offsetBound
        as ⟨result, resultAt⟩
      rename_i id
      constructor
      · intro equal
        refine ⟨position.val, inside, ?_, ?_⟩
        · simpa only [rowEqual, pageValue] using congrArg UScalar.val matched
        · rw [←rowEqual, ←offset, List.getElem?_eq_getElem offsetBound,
            ←resultAt, equal]
      · rintro ⟨other, otherBound, otherKey, otherAt⟩
        have sameKey : (index.pages.val[position.val]!).key.val =
            (index.pages.val[other]!).key.val := by
          rw [←rowEqual, matched, pageValue, otherKey]
        have samePosition := directory_key_unique index.pages.val valid position.val other inside otherBound sameKey
        subst other
        rw [←rowEqual, ←offset, List.getElem?_eq_getElem offsetBound] at otherAt
        simpa only [←resultAt] using Option.some.inj otherAt
    · simp only [matched, ite_false, WP.spec_ok]
      intro id
      constructor
      · intro impossible; cases impossible
      · rintro ⟨other, otherBound, otherKey, otherAt⟩
        have missing : (index.pages.val[position.val]!).key.val ≠ page.val := by
          intro equal; apply matched; exact UScalar.eq_of_val_eq (by simpa only [rowEqual] using equal)
        have noneEqual := lower_bound_missing_key (directory_ordered index.pages.val valid)
          positionSpec (Or.inr missing) other otherBound
        exact False.elim (noneEqual (by rw [otherKey, pageValue]))
  · have test : ¬position < alloc.vec.Vec.len index.pages := by scalar_tac
    simp only [test, ite_false, WP.spec_ok]
    intro id
    constructor
    · intro impossible; cases impossible
    · rintro ⟨other, otherBound, otherKey, otherAt⟩
      have endPosition : position.val = index.pages.val.length := by have := positionSpec.1; omega
      have noneEqual := lower_bound_missing_key (directory_ordered index.pages.val valid)
        positionSpec (Or.inl endPosition) other otherBound
      exact False.elim (noneEqual (by rw [otherKey, pageValue]))

end MemoryWrite.RegisterIndex
