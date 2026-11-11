/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureFormatsCore
import Mathlib.Tactic.FinCases
import Mathlib.Data.Fintype.Fin

set_option maxRecDepth 100000
set_option maxHeartbeats 0

namespace GB300Math.TextureHalfExact

def halfUnits (word : Nat) : Nat :=
  let exponent := word / 2^10 % 32
  if exponent = 0 then word % 2^10
  else (2^10 + word % 2^10) * 2^(exponent - 1)

def floatUnits (word : Nat) : Nat :=
  let exponent := word / 2^23 % 256
  if exponent = 0 then word % 2^23
  else (2^23 + word % 2^23) * 2^(exponent - 1)

private def Exact (word : Nat) : Prop :=
  (word % 32768 < 31744 →
    floatUnits (TextureFormatsCore.halfToFloat word) = halfUnits word * 2^125) ∧
  TextureFormatsCore.halfToFloat word / 2^31 = word / 2^15 ∧
  (word % 32768 ≥ 31744 →
    TextureFormatsCore.halfToFloat word % 2^31 = 0x7f800000 + word % 2^10 * 2^13)

private theorem exact_block (high : Fin 256) :
    ∀ low : Fin 256, Exact (high.val * 256 + low.val) := by
  fin_cases high <;> unfold Exact <;> decide +kernel

private theorem all_exact (word : Fin 65536) : Exact word.val := by
  have hw := word.isLt
  have hi : word.val / 256 < 256 := by omega
  have lo : word.val % 256 < 256 := Nat.mod_lt _ (by decide)
  have h := exact_block ⟨word.val / 256, hi⟩ ⟨word.val % 256, lo⟩
  have split : word.val / 256 * 256 + word.val % 256 = word.val := by omega
  simp only at h
  rw [split] at h
  exact h

theorem finite_value_exact (word : Fin 65536) (finite : word.val % 32768 < 31744) :
    floatUnits (TextureFormatsCore.halfToFloat word.val) = halfUnits word.val * 2^125 :=
  (all_exact word).1 finite

theorem sign_preserved (word : Fin 65536) :
    TextureFormatsCore.halfToFloat word.val / 2^31 = word.val / 2^15 :=
  (all_exact word).2.1

theorem special_payload_preserved (word : Fin 65536)
    (special : word.val % 32768 ≥ 31744) :
    TextureFormatsCore.halfToFloat word.val % 2^31 =
      0x7f800000 + word.val % 2^10 * 2^13 :=
  (all_exact word).2.2 special

end GB300Math.TextureHalfExact
#print axioms GB300Math.TextureHalfExact.finite_value_exact
#print axioms GB300Math.TextureHalfExact.sign_preserved
#print axioms GB300Math.TextureHalfExact.special_payload_preserved
