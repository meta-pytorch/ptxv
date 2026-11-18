/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWritePaths

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

def JoinInvariant (original source : Slice U64)
    (state : Slice U64 × Bool × Usize) : Prop :=
  state.1.val.length = original.val.length ∧
  state.2.2.val ≤ source.val.length ∧
  (∀ position < state.2.2.val,
    state.1.val[position]! = original.val[position]! ||| source.val[position]!) ∧
  (∀ position, state.2.2.val ≤ position → position < original.val.length →
    state.1.val[position]! = original.val[position]!) ∧
  (state.2.1 = true ↔ ∃ position < state.2.2.val,
    original.val[position]! ||| source.val[position]! ≠ original.val[position]!)

@[step]
theorem rust_join_words (target source : Slice U64)
    (lengths : target.val.length = source.val.length) :
    cache_graph.join_words target source ⦃ changed result =>
      result.val.length = target.val.length ∧
      (∀ position < target.val.length,
        result.val[position]! = target.val[position]! ||| source.val[position]!) ∧
      (changed = true ↔ result.val ≠ target.val) ⦄ := by
  classical
  unfold cache_graph.join_words cache_graph.join_words_loop
  apply loop.spec_decr_nat
    (fun state : Slice U64 × Bool × Usize => source.val.length - state.2.2.val)
    (JoinInvariant target source)
  · rintro ⟨current, changed, index⟩ ⟨currentLength, bound, processed, suffix, changedIff⟩
    dsimp only at currentLength bound processed suffix changedIff
    unfold cache_graph.join_words_loop.body
    by_cases hlt : index.val < source.val.length
    · have test : index < Slice.len source := by scalar_tac
      simp only [test, ite_true]
      step with Slice.index_usize_spec current index (by scalar_tac) as ⟨left, leftEq⟩
      step with Slice.index_usize_spec source index hlt as ⟨right, rightEq⟩
      simp only [lift, bind_ok]
      have currentBound : index.val < current.val.length := by omega
      have leftOriginal : left = target.val[index.val]! := by
        calc
          left = current.val[index.val]! := by
            simpa only [List.getElem!_eq_getElem?_getD,
              List.getElem?_eq_getElem currentBound, Option.getD_some] using leftEq
          _ = target.val[index.val]! := suffix index.val (by omega) (by omega)
      have rightOriginal : right = source.val[index.val]! := by
        simpa only [List.getElem!_eq_getElem?_getD,
          List.getElem?_eq_getElem hlt, Option.getD_some] using rightEq
      simp only [bne_iff_ne]
      split
      · rename_i differs
        step with Slice.update_spec current index (left ||| right) (by scalar_tac) as ⟨updated, updatedEq⟩
        step as ⟨next, nextEq⟩; try scalar_tac
        subst updated
        refine ⟨?_, ?_⟩
        · dsimp only [JoinInvariant]
          refine ⟨by simp [currentLength], by scalar_tac, ?_, ?_, ?_⟩
          · intro position hposition
            by_cases same : position = index.val
            · subst position
              rw [←Slice.getElem!_Nat_eq (current.set index (left ||| right)) index.val,
                Slice.getElem!_Nat_set_eq current index index.val (left ||| right)
                ⟨rfl, by scalar_tac⟩, leftOriginal, rightOriginal]
            · have before : position < index.val := by omega
              rw [←Slice.getElem!_Nat_eq (current.set index (left ||| right)) position,
                Slice.getElem!_Nat_set_ne current index position (left ||| right) (Ne.symm same)]
              simpa only [Slice.getElem!_Nat_eq] using processed position before
          · intro position hafter hposition
            have different : index.val ≠ position := by scalar_tac
            rw [←Slice.getElem!_Nat_eq (current.set index (left ||| right)) position,
              Slice.getElem!_Nat_set_ne current index position (left ||| right) different]
            simpa only [Slice.getElem!_Nat_eq] using suffix position (by scalar_tac) hposition
          · constructor
            · intro _
              exact ⟨index.val, by scalar_tac, by simpa [←leftOriginal, ←rightOriginal] using Ne.symm differs⟩
            · intro _; rfl
        · scalar_tac
      · rename_i same
        have same : left ||| right = left := (not_not.mp same).symm
        step as ⟨next, nextEq⟩; try scalar_tac
        refine ⟨?_, ?_⟩
        · dsimp only [JoinInvariant]
          refine ⟨currentLength, by scalar_tac, ?_, ?_, ?_⟩
          · intro position hposition
            by_cases atIndex : position = index.val
            · subst position
              rw [suffix index.val (by omega) (by omega)]
              simpa [leftOriginal, rightOriginal] using same.symm
            · exact processed position (by scalar_tac)
          · intro position hafter hposition
            simpa only [Slice.getElem!_Nat_eq] using suffix position (by scalar_tac) hposition
          · rw [changedIff]
            constructor
            · rintro ⟨position, hposition, differs⟩
              exact ⟨position, by scalar_tac, differs⟩
            · rintro ⟨position, hposition, differs⟩
              refine ⟨position, ?_, differs⟩
              by_contra beyond
              have atIndex : position = index.val := by scalar_tac
              subst position
              exact differs (by simpa [←leftOriginal, ←rightOriginal] using same)
        · scalar_tac
    · have test : ¬index < Slice.len source := by scalar_tac
      simp only [test, ite_false, WP.spec_ok]
      have endIndex : index.val = source.val.length := by omega
      refine ⟨currentLength, ?_, ?_⟩
      · intro position hposition
        exact processed position (by omega)
      · rw [changedIff]
        constructor
        · rintro ⟨position, hposition, differs⟩ equal
          apply differs
          have := processed position hposition
          rw [equal] at this
          exact this.symm
        · intro differs
          by_contra noPosition
          apply differs
          apply List.ext_getElem!
          · exact currentLength
          · intro position
            by_cases hposition : position < target.val.length
            · have same : target.val[position]! ||| source.val[position]! =
                  target.val[position]! := by
                by_contra different
                exact noPosition ⟨position, by omega, different⟩
              rw [processed position (by omega), same]
            · simp [List.getElem!_eq_getElem?_getD, List.getElem?_eq_none (by omega : current.val.length ≤ position),
                List.getElem?_eq_none (by omega : target.val.length ≤ position)]
  · refine ⟨rfl, by scalar_tac, ?_, ?_, ?_⟩
    · intro position hposition; scalar_tac
    · intro position _ _; rfl
    · simp


theorem word_contains_at (words : List U64) (register : Nat)
    (inside : register / 64 < words.length) :
    WordContains words register ↔ words[register / 64]!.bv[register % 64] = true := by
  simp only [WordContains, List.getElem?_eq_getElem inside,
    Option.some.injEq, List.getElem!_eq_getElem?_getD, Option.getD_some]
  constructor
  · rintro ⟨word, equal, bit⟩
    simpa only [←equal] using bit
  · intro bit
    exact ⟨_, rfl, bit⟩

theorem word_contains_absent (words : List U64) (register : Nat)
    (outside : words.length ≤ register / 64) : ¬WordContains words register := by
  simp [WordContains, List.getElem?_eq_none outside]

theorem word_contains_union (left right result : List U64)
    (leftLength : left.length = result.length) (rightLength : right.length = result.length)
    (joined : ∀ position < result.length, result[position]! = left[position]! ||| right[position]!)
    (register : Nat) :
    WordContains result register ↔ WordContains left register ∨ WordContains right register := by
  by_cases inside : register / 64 < result.length
  · rw [word_contains_at result register inside,
      word_contains_at left register (by omega), word_contains_at right register (by omega),
      joined (register / 64) inside]
    have bitBound : register % 64 < 64 := Nat.mod_lt _ (by decide)
    change ((left[register / 64]!.bv ||| right[register / 64]!.bv)[register % 64]) = true ↔ _
    rw [BitVec.getElem_or bitBound]
    simp only [Bool.or_eq_true]
  · have noneResult := word_contains_absent result register (by omega)
    have noneLeft := word_contains_absent left register (by omega)
    have noneRight := word_contains_absent right register (by omega)
    simp only [noneResult, noneLeft, noneRight, or_self]

theorem rust_join_covers_union (target source : Slice U64)
    (lengths : target.val.length = source.val.length) (changed : Bool) (result : Slice U64)
    (execution : cache_graph.join_words target source = .ok (changed, result)) :
    ∀ register, WordContains result.val register ↔
      WordContains target.val register ∨ WordContains source.val register := by
  have specification := rust_join_words target source lengths
  simp only [execution, WP.spec_ok] at specification
  obtain ⟨resultLength, joined, _⟩ := specification
  exact word_contains_union target.val source.val result.val resultLength.symm
    (lengths.symm.trans resultLength.symm)
    (fun position bound => joined position (by omega))

theorem rust_join_unchanged_is_equal (target source : Slice U64)
    (lengths : target.val.length = source.val.length) (result : Slice U64)
    (execution : cache_graph.join_words target source = .ok (false, result)) :
    result.val = target.val := by
  have specification := rust_join_words target source lengths
  simp only [execution, WP.spec_ok] at specification
  by_contra different
  have impossible := specification.2.2.mpr different
  contradiction


theorem zero_words_empty (count register : Nat) :
    ¬WordContains (List.replicate count 0#u64) register := by
  unfold WordContains
  by_cases inside : register / 64 < count
  · simp only [List.getElem?_replicate_of_lt inside, Option.some.injEq]
    rintro ⟨word, equal, bit⟩
    subst word
    simp at bit
  · simp only [List.getElem?_replicate, inside, ite_false, reduceCtorEq, false_and,
      exists_const, not_false_eq_true]

end MemoryWrite.MayWrite
