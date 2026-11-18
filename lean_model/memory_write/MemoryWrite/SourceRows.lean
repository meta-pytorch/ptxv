/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Code.Funs

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.SourceRows

structure ReaderSpec {R : Type} (reader : source_rows.RegisterRows R) (source : R)
    (model : alloc.vec.Vec (alloc.vec.Vec symbolic.RegisterKey)) : Prop where
  count : reader.row_count source = .ok (alloc.vec.Vec.len model)
  length : ∀ pc, (inside : pc.val < model.val.length) →
    reader.row_len source pc = .ok (alloc.vec.Vec.len model.val[pc.val])
  key : ∀ pc position, (inside : pc.val < model.val.length) →
    (within : position.val < (model.val[pc.val]).val.length) →
    reader.key source pc position = .ok (model.val[pc.val]).val[position.val]

@[step]
theorem rust_row_total {R : Type} (reader : source_rows.RegisterRows R) (source : R)
    (pc : Usize) (expected : alloc.vec.Vec symbolic.RegisterKey)
    (read : ∀ position, (inside : position.val < expected.val.length) →
      reader.key source pc position = .ok expected.val[position.val]) :
    source_rows.collect_loop0_loop0 reader source pc (alloc.vec.Vec.len expected)
      (alloc.vec.Vec.with_capacity symbolic.RegisterKey (alloc.vec.Vec.len expected)) 0#usize
      ⦃ row => row.val = expected.val ⦄ := by
  unfold source_rows.collect_loop0_loop0
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec symbolic.RegisterKey × Usize => expected.val.length - state.2.val)
    (fun state : alloc.vec.Vec symbolic.RegisterKey × Usize =>
      state.2.val ≤ expected.val.length ∧ state.1.val = expected.val.take state.2.val)
  · rintro ⟨row, position⟩ ⟨positionBound, rowEq⟩
    dsimp only at positionBound rowEq
    unfold source_rows.collect_loop0_loop0.body
    by_cases inside : position.val < expected.val.length
    · have test : position < alloc.vec.Vec.len expected := by scalar_tac
      simp only [test, ite_true, read position inside, bind_ok]
      step with alloc.vec.Vec.push_spec row expected.val[position.val]
        (by rw [rowEq, List.length_take, Nat.min_eq_left positionBound]; have := expected.property; omega)
        as ⟨appended, appendedEq⟩
      step with Usize.add_spec (x := position) (y := 1#usize) (by scalar_tac)
        as ⟨next, nextValue⟩
      have nextVal : next.val = position.val + 1 := by scalar_tac
      refine ⟨by scalar_tac, ?_, by scalar_tac⟩
      rw [nextVal, appendedEq, rowEq, List.take_succ_eq_append_getElem inside]
      rfl
    · have test : ¬position < alloc.vec.Vec.len expected := by scalar_tac
      have complete : position.val = expected.val.length := by omega
      simp only [test, ite_false, WP.spec_ok]
      simpa only [complete, List.take_length] using rowEq
  · exact ⟨by scalar_tac, by simp [alloc.vec.Vec.with_capacity]⟩

@[step]
theorem rust_collect_total {R : Type} (reader : source_rows.RegisterRows R) (source : R)
    (model : alloc.vec.Vec (alloc.vec.Vec symbolic.RegisterKey))
    (correct : ReaderSpec reader source model) :
    source_rows.collect reader source ⦃ rows => rows.val.map alloc.vec.Vec.val = model.val.map alloc.vec.Vec.val ⦄ := by
  unfold source_rows.collect
  simp only [correct.count, bind_ok]
  unfold source_rows.collect_loop0
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec (alloc.vec.Vec symbolic.RegisterKey) × Usize =>
      model.val.length - state.2.val)
    (fun state : alloc.vec.Vec (alloc.vec.Vec symbolic.RegisterKey) × Usize =>
      state.2.val ≤ model.val.length ∧
      state.1.val.map alloc.vec.Vec.val = (model.val.take state.2.val).map alloc.vec.Vec.val)
  · rintro ⟨rows, pc⟩ ⟨pcBound, rowsEq⟩
    dsimp only at pcBound rowsEq
    unfold source_rows.collect_loop0.body
    by_cases inside : pc.val < model.val.length
    · have test : pc < alloc.vec.Vec.len model := by scalar_tac
      simp only [test, ite_true, correct.length pc inside, bind_ok]
      step with rust_row_total reader source pc model.val[pc.val]
        (fun position bound => correct.key pc position inside bound) as ⟨row, rowEq⟩
      have rowsLength : rows.val.length = pc.val := by
        have lengths := congrArg List.length rowsEq
        simp only [List.length_map, List.length_take, Nat.min_eq_left pcBound] at lengths
        exact lengths
      step with alloc.vec.Vec.push_spec rows row (by rw [rowsLength]; have := model.property; omega)
        as ⟨appended, appendedEq⟩
      step with Usize.add_spec (x := pc) (y := 1#usize) (by scalar_tac)
        as ⟨next, nextValue⟩
      have nextVal : next.val = pc.val + 1 := by scalar_tac
      refine ⟨by scalar_tac, ?_, by scalar_tac⟩
      rw [nextVal, appendedEq, List.take_succ_eq_append_getElem inside]
      simp only [List.map_append, List.map_singleton, rowsEq, rowEq]
      rfl
    · have test : ¬pc < alloc.vec.Vec.len model := by scalar_tac
      have complete : pc.val = model.val.length := by omega
      simp only [test, ite_false, WP.spec_ok]
      simpa only [complete, List.take_length] using rowsEq
  · exact ⟨by scalar_tac, by simp [alloc.vec.Vec.with_capacity]⟩

end MemoryWrite.SourceRows

namespace MemoryWrite.SourceRows

@[step]
theorem rust_collect_exact {R : Type} (reader : source_rows.RegisterRows R) (source : R)
    (model : alloc.vec.Vec (alloc.vec.Vec symbolic.RegisterKey))
    (correct : ReaderSpec reader source model) :
    source_rows.collect reader source ⦃ rows => rows = model ⦄ := by
  apply WP.spec_mono (rust_collect_total reader source model correct)
  intro rows same
  apply alloc.vec.Vec.ext
  have injective : ∀ (left right : List (alloc.vec.Vec symbolic.RegisterKey)),
      left.map alloc.vec.Vec.val = right.map alloc.vec.Vec.val → left = right := by
    intro left
    induction left with
    | nil => intro right equal; cases right <;> simp_all
    | cons first rest ih =>
        intro right equal
        cases right with
        | nil => simp at equal
        | cons other tail =>
            simp only [List.map_cons, List.cons.injEq] at equal
            exact congrArg₂ List.cons (alloc.vec.Vec.ext first other equal.1) (ih tail equal.2)
  exact injective _ _ same

end MemoryWrite.SourceRows
