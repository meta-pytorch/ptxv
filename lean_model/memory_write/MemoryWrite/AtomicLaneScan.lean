/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AtomicLaneCoverage

set_option autoImplicit false

namespace MemoryWrite.AtomicLaneScan

variable {α : Type} [Inhabited α]

def AnySuffix (rel : α → α → Prop) (left : α) (claims : List α) (start : Nat) : Prop :=
  ∃ index, start ≤ index ∧ index < claims.length ∧ rel left claims[index]!

def AnyPair (rel : α → α → Prop) (claims : List α) (start : Nat) : Prop :=
  ∃ first second, start ≤ first ∧ first < second ∧ second < claims.length ∧
    rel claims[first]! claims[second]!

theorem suffix_advance (rel : α → α → Prop) (left : α) (claims : List α) (index : Nat)
    (absent : ¬rel left claims[index]!) :
    AnySuffix rel left claims index ↔ AnySuffix rel left claims (index + 1) := by
  constructor
  · rintro ⟨other, bound, inside, found⟩
    refine ⟨other, ?_, inside, found⟩
    have different : other ≠ index := by
      intro same
      exact absent (by simpa [same] using found)
    omega
  · rintro ⟨other, bound, inside, found⟩
    exact ⟨other, by omega, inside, found⟩

theorem pair_advance (rel : α → α → Prop) (claims : List α) (index : Nat)
    (absent : ¬AnySuffix rel claims[index]! claims (index + 1)) :
    AnyPair rel claims index ↔ AnyPair rel claims (index + 1) := by
  constructor
  · rintro ⟨first, second, bound, ordered, inside, found⟩
    refine ⟨first, second, ?_, ordered, inside, found⟩
    have different : first ≠ index := by
      intro same
      exact absent ⟨second, by omega, inside, by simpa [same] using found⟩
    omega
  · rintro ⟨first, second, bound, ordered, inside, found⟩
    exact ⟨first, second, by omega, ordered, inside, found⟩

theorem a_pair_is_found_in_its_first_suffix (rel : α → α → Prop) (claims : List α)
    (index : Nat) (found : AnySuffix rel claims[index]! claims (index + 1)) :
    AnyPair rel claims index := by
  obtain ⟨second, bound, inside, found⟩ := found
  exact ⟨index, second, by omega, by omega, inside, found⟩

theorem no_pair_at_the_end (rel : α → α → Prop) (claims : List α)
    (index : Nat) (outside : claims.length ≤ index) : ¬AnyPair rel claims index := by
  rintro ⟨first, second, bound, ordered, inside, found⟩
  omega

theorem no_suffix_at_the_end (rel : α → α → Prop) (left : α) (claims : List α)
    (index : Nat) (outside : claims.length ≤ index) : ¬AnySuffix rel left claims index := by
  rintro ⟨other, bound, inside, found⟩
  omega

end MemoryWrite.AtomicLaneScan
