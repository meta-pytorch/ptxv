/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import TanhCentralUniform

namespace GB300Math.TanhDiscontinuity
open FloatLib.Floats.Formats.BinaryInterchange
set_option maxRecDepth 100000
set_option maxHeartbeats 2000000

noncomputable def value (word : Nat) : ℝ :=
  (Model.ofNatBits word : Model FloatFormat.binary32).toReal

noncomputable def error (word : Nat) : ℝ :=
  |value ((Mufu.tanh word).getD 0) - Real.tanh (value word)|

theorem native_bits :
    Mufu.tanh 0x3e9fffff = some 0x3e9afd0a ∧
    Mufu.tanh 0x3ea00000 = some 0x3e9afc98 := by decide +kernel

theorem decoded_values :
    value 0x3e9fffff = 10485759/33554432 ∧ value 0x3ea00000 = 5/16 ∧
    value 0x3e9afd0a = 5078661/16777216 ∧ value 0x3e9afc98 = 1269651/4194304 := by
  have h :
      (Model.ofNatBits 0x3e9fffff : Model FloatFormat.binary32).toRat? = some (10485759/33554432) ∧
      (Model.ofNatBits 0x3ea00000 : Model FloatFormat.binary32).toRat? = some (5/16) ∧
      (Model.ofNatBits 0x3e9afd0a : Model FloatFormat.binary32).toRat? = some (5078661/16777216) ∧
      (Model.ofNatBits 0x3e9afc98 : Model FloatFormat.binary32).toRat? = some (1269651/4194304) := by
    decide +kernel
  have h0 := Model.Algebraic.toReal_of_toRat?_eq_some h.1
  have h1 := Model.Algebraic.toReal_of_toRat?_eq_some h.2.1
  have h2 := Model.Algebraic.toReal_of_toRat?_eq_some h.2.2.1
  have h3 := Model.Algebraic.toReal_of_toRat?_eq_some h.2.2.2
  norm_num [value] at h0 h1 h2 h3 ⊢
  exact ⟨h0,h1,h2,h3⟩

theorem ideal_monotone (x y : ℝ) (h : x ≤ y) : Real.tanh x ≤ Real.tanh y := by
  rw [TanhCentralAnalytic.tanh_negative_exp, TanhCentralAnalytic.tanh_negative_exp]
  have hx := Real.exp_pos (-2*x)
  have hy := Real.exp_pos (-2*y)
  have he := Real.exp_le_exp.mpr (show -2*y ≤ -2*x by linarith)
  apply (div_le_div_iff₀ (by positivity) (by positivity)).mpr
  nlinarith

theorem downward_step :
    value 0x3e9fffff < value 0x3ea00000 ∧
    value ((Mufu.tanh 0x3e9fffff).getD 0) -
      value ((Mufu.tanh 0x3ea00000).getD 0) = 114/2^25 := by
  rw [native_bits.1,native_bits.2]
  simp only [Option.getD_some]
  rw [decoded_values.1,decoded_values.2.1,decoded_values.2.2.1,decoded_values.2.2.2]
  norm_num

theorem accuracy_lower_bound : 57/2^25 ≤ max (error 0x3e9fffff) (error 0x3ea00000) := by
  have hm := ideal_monotone _ _ downward_step.1.le
  have h0 := le_abs_self (value ((Mufu.tanh 0x3e9fffff).getD 0)-Real.tanh (value 0x3e9fffff))
  have h1 := neg_le_abs (value ((Mufu.tanh 0x3ea00000).getD 0)-Real.tanh (value 0x3ea00000))
  have hstep := downward_step.2
  have hl := le_max_left (error 0x3e9fffff) (error 0x3ea00000)
  have hr := le_max_right (error 0x3e9fffff) (error 0x3ea00000)
  unfold error at hl hr ⊢
  norm_num only [Nat.reducePow] at hstep ⊢
  linarith

theorem not_uniform_2neg20 :
    ¬ (∀ word : Nat, word < 0x7f800000 → error word ≤ 1/2^20) := by
  intro h
  have h0 := h 0x3e9fffff (by decide)
  have h1 := h 0x3ea00000 (by decide)
  have hm := max_le h0 h1
  have hb := accuracy_lower_bound
  norm_num only [Nat.reducePow] at hb hm
  linarith

#print axioms native_bits
#print axioms decoded_values
#print axioms ideal_monotone
#print axioms downward_step
#print axioms accuracy_lower_bound
#print axioms not_uniform_2neg20
end GB300Math.TanhDiscontinuity
