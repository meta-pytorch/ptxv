/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteOutgoing

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

theorem word_contains_bound (words : List U64) (register : Nat) (bit : WordContains words register) :
    register < words.length * 64 := by
  have bound := bit.bound
  omega

theorem words_eq_of_contains (left right : List U64) (lengths : left.length = right.length)
    (same : ∀ register, WordContains left register ↔ WordContains right register) : left = right := by
  apply List.ext_getElem lengths
  intro index leftBound rightBound
  apply U64.bv_eq_imp_eq
  apply BitVec.eq_of_getElem_eq
  intro bit bitBound
  change bit < 64 at bitBound
  have quotient : (index * 64 + bit) / 64 = index := by omega
  have remainder : (index * 64 + bit) % 64 = bit := by omega
  have leftSpec := word_contains_at left (index * 64 + bit) (by omega)
  have rightSpec := word_contains_at right (index * 64 + bit) (by omega)
  simp only [quotient, remainder, ←List.Inhabited_getElem_eq_getElem! left index leftBound] at leftSpec
  simp only [quotient, remainder, ←List.Inhabited_getElem_eq_getElem! right index rightBound] at rightSpec
  exact Bool.eq_iff_iff.mpr (leftSpec.symm.trans ((same _).trans rightSpec))

theorem growing_different_words_add_bit (before after : List U64)
    (lengths : before.length = after.length)
    (growth : ∀ register, WordContains before register → WordContains after register)
    (different : after ≠ before) :
    ∃ register, register < before.length * 64 ∧
      WordContains after register ∧ ¬WordContains before register := by
  classical
  by_contra absent
  push Not at absent
  apply different
  apply words_eq_of_contains after before lengths.symm
  intro register
  constructor
  · intro afterBit
    have bound : register < before.length * 64 := by
      rw [lengths]; exact word_contains_bound after register afterBit
    exact absent register bound afterBit
  · exact growth register

end MemoryWrite.MayWrite
