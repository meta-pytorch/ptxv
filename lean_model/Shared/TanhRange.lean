/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import TanhLinear
import Exp2Boundary
import SqrtBounds

namespace GB300Math.TanhRange
set_option maxRecDepth 100000
set_option maxHeartbeats 3000000

abbrev lim : Int := 131071
abbrev dropMax : Int := 4718593

def valid (a b c : Int) (frac : Nat) : Prop :=
  c ≤ 0 ∧ 0 ≤ b ∧ 0 ≤ b*2^17+2*c*lim ∧
    a*2^34+b*lim*2^17+c*lim^2-c*dropMax < (2 : Int)^frac
instance (a b c : Int) (f : Nat) : Decidable (valid a b c f) := inferInstanceAs (Decidable (_ ∧ _ ∧ _ ∧ _))

def coeff (e j : Nat) : Int × Int × Int := (TanhTables.binades[e]!).2[j]!
def frac (e : Nat) : Nat := 184-min (e+119) 126

theorem certificate : ∀ e : Fin 11, ∀ j : Fin 16,
    valid (coeff e j).1 (coeff e j).2.1 (coeff e j).2.2 (frac e) := by decide +kernel

theorem discard_bound : ∀ e : Fin 11, 2 ≤ (TanhTables.binades[e.val]!).1 := by decide +kernel

theorem polynomial_bound (a b c : Int) (f k : Nat)
    (hv : valid a b c f) (hk : k < 2^17) :
    a*2^34+b*(k : Int)*2^17+c*(Fp64Seed.squareTruncated k 19 : Int) < (2 : Int)^f := by
  have hsq : Fp64Seed.squareTruncated k 19 ≤ k*k := by
    unfold Fp64Seed.squareTruncated
    exact Nat.sub_le _ _
  have hd := SqrtBounds.square_drop_bound k
  have hd' : (k : Int)*(k : Int)-(Fp64Seed.squareTruncated k 19 : Int) ≤ dropMax := by
    have he := Nat.sub_add_cancel hsq
    have hc : ((k*k-Fp64Seed.squareTruncated k 19 : Nat) : Int)+(Fp64Seed.squareTruncated k 19 : Int)=(k : Int)*(k : Int) := by exact_mod_cast he
    have hh : ((k*k-Fp64Seed.squareTruncated k 19 : Nat) : Int) ≤ 4718593 := by exact_mod_cast hd
    dsimp [dropMax]
    omega
  rcases hv with ⟨hc,hb,hm,hu⟩
  have hk' : 0 ≤ (k : Int) ∧ (k : Int) ≤ lim := by dsimp [lim];omega
  have hlin : 0 ≤ b*2^17+c*(lim+(k : Int)) := by nlinarith
  have hmono := mul_nonneg (show 0 ≤ lim-(k : Int) by dsimp [lim] at *;omega) hlin
  have hdrop := mul_nonpos_of_nonpos_of_nonneg hc (show 0 ≤ dropMax-((k : Int)*(k : Int)-(Fp64Seed.squareTruncated k 19 : Int)) by omega)
  dsimp [lim,dropMax] at *
  nlinarith

theorem fixed_range (m f : Nat) (hf : 1 ≤ f ∧ f ≤ 126) (hm : m < 2^f) :
    0x00800000 ≤ Mufu.fixedToWord m f ∧ Mufu.fixedToWord m f < 0x3f800000 := by
  have hl : m.log2 < f := by
    by_cases hz : m=0
    · subst m;simp;omega
    · exact (Nat.log2_lt hz).mpr hm
  have hr (v : Nat) : v%2^23 < 2^23 := Nat.mod_lt _ (by decide)
  dsimp [Mufu.fixedToWord]
  split <;> have h := hr (if m.log2 ≥ 23 then m/2^(m.log2-23) else m*2^(23-m.log2)) <;> split_ifs at h <;> omega

theorem middle_magnitude (word : Nat) (he : 119 ≤ Mufu.exponent word ∧ Mufu.exponent word < 130) :
    let e := Mufu.exponent word
    let (discard, coefficients) := TanhTables.binades[e-119]!
    let index := Mufu.fraction word/2^discard
    let k := index%2^17
    let (a,b,c) := coefficients[index/2^17]!
    let value := a*2^34+b*(k : Int)*2^17+c*(Fp64Seed.squareTruncated k 19 : Int)
    0x00800000 ≤ Mufu.fixedToWord value.toNat (184-(min e 126 : Int)) ∧
      Mufu.fixedToWord value.toNat (184-(min e 126 : Int)) < 0x3f800000 := by
  let e : Fin 11 := ⟨Mufu.exponent word-119,by omega⟩
  let d := (TanhTables.binades[e.val]!).1
  have hd : 2 ≤ d := discard_bound e
  have hp : (4 : Nat) ≤ 2^d := by simpa using Nat.pow_le_pow_right (by decide : 0 < 2) hd
  have hf : Mufu.fraction word < 2^23 := by dsimp [Mufu.fraction];omega
  let index := Mufu.fraction word/2^d
  have hi : index < 2^21 := by
    dsimp [index]
    apply (Nat.div_lt_iff_lt_mul (by positivity)).mpr
    nlinarith
  let j : Fin 16 := ⟨index/2^17,by omega⟩
  let k := index%2^17
  have hk : k < 2^17 := Nat.mod_lt _ (by decide)
  have hb := polynomial_bound (coeff e j).1 (coeff e j).2.1 (coeff e j).2.2 (frac e) k (certificate e j) hk
  let v := (coeff e j).1*2^34+(coeff e j).2.1*(k : Int)*2^17+(coeff e j).2.2*(Fp64Seed.squareTruncated k 19 : Int)
  have hnat : v.toNat < 2^(frac e) := by
    apply (Int.toNat_lt' (by positivity)).mpr
    exact_mod_cast hb
  have hfr : 1 ≤ frac e ∧ frac e ≤ 126 := by dsimp [frac,e];omega
  have hr := fixed_range v.toNat (frac e) hfr hnat
  have hfcast : (frac e : Int) = 184-(min (Mufu.exponent word) 126 : Int) := by dsimp [frac,e];omega
  rw [hfcast] at hr
  exact hr

theorem word_parts (word : Nat) (hw : word < 2^32) :
    word = Mufu.sign word+Mufu.magnitude word ∧ (Mufu.sign word = 0 ∨ Mufu.sign word = 2^31) := by
  by_cases hs : word < 2^31
  · rw [RcpBounds.small_sign word hs]
    dsimp [Mufu.magnitude]
    omega
  · have hlt : word-2^31 < 2^31 := by omega
    have he : word=2^31+(word-2^31) := by omega
    have hp := RcpBounds.high_sign (word-2^31) hlt
    rw [←he] at hp
    rw [hp]
    dsimp [Mufu.magnitude]
    omega

theorem middle_result (word : Nat) (he : 119 ≤ Mufu.exponent word ∧ Mufu.exponent word < 130)
    (hf : Mufu.magnitude word < 0x7f800000) :
    ∃ mag, 0x00800000 ≤ mag ∧ mag < 0x3f800000 ∧ Mufu.tanh word = some (Mufu.sign word+mag) := by
  have h0 : ¬ Mufu.magnitude word > 0x7f800000 := by omega
  have h1 : ¬ Mufu.exponent word < 119 := by omega
  have h2 : ¬ Mufu.exponent word ≥ 130 := by omega
  simp only [Mufu.tanh,h0,h1,h2,ite_false]
  exact ⟨_,(middle_magnitude word he).1,(middle_magnitude word he).2,rfl⟩

theorem finite_magnitude (word : Nat) (hw : word < 2^32)
    (hf : Mufu.magnitude word < 0x7f800000) :
    ∃ mag, mag ≤ 0x3f800000 ∧ Mufu.tanh word = some (Mufu.sign word+mag) := by
  by_cases hl : Mufu.exponent word < 119
  · refine ⟨Mufu.magnitude word,?_,?_⟩
    · dsimp [Mufu.exponent,Mufu.magnitude] at *;omega
    · rw [TanhLinear.small_bits word hw (by omega)]
      exact congrArg some (word_parts word hw).1
  · by_cases hh : Mufu.exponent word ≥ 130
    · refine ⟨0x3f800000,by omega,?_⟩
      simp [Mufu.tanh,not_lt.mpr hf.le,hl,hh]
    · obtain ⟨mag,_,hm,ho⟩ := middle_result word (by omega) hf
      exact ⟨mag,le_of_lt hm,ho⟩

theorem finite_range_and_sign (word : Nat) (hw : word < 2^32)
    (hf : Mufu.magnitude word < 0x7f800000) :
    ∃ out, Mufu.tanh word = some out ∧ out < 2^32 ∧
      Mufu.magnitude out ≤ 0x3f800000 ∧ Mufu.sign out = Mufu.sign word := by
  obtain ⟨mag,hm,ho⟩ := finite_magnitude word hw hf
  refine ⟨Mufu.sign word+mag,ho,?_,?_,?_⟩
  · rcases (word_parts word hw).2 with hs|hs <;> omega
  · rcases (word_parts word hw).2 with hs|hs <;> rw [hs] <;> dsimp [Mufu.magnitude] <;> omega
  · rcases (word_parts word hw).2 with hs|hs
    · rw [hs,Nat.zero_add];exact RcpBounds.small_sign mag (by omega)
    · rw [hs];exact RcpBounds.high_sign mag (by omega)

theorem signed_zero_bits : Mufu.tanh 0 = some 0 ∧ Mufu.tanh 0x80000000 = some 0x80000000 := by decide +kernel

theorem decode_range (word : Nat) (_hw : word < 2^32) (hm : Mufu.magnitude word ≤ 0x3f800000) :
    |Exp2Bounds.finiteValue word| ≤ 1 := by
  have he : Mufu.exponent word ≤ 127 := by dsimp [Mufu.exponent,Mufu.magnitude] at *;omega
  have hf : Mufu.fraction word < 2^23 := by dsimp [Mufu.fraction];omega
  unfold Exp2Bounds.finiteValue
  rw [abs_mul]
  have hsign : |(if word < 2^31 then 1 else -1 : ℝ)| = 1 := by split <;> norm_num
  rw [hsign,one_mul]
  by_cases hz : Mufu.exponent word = 0
  · rw [ite_eq_left hz,abs_of_nonneg (by positivity)]
    have hfq : (Mufu.fraction word : ℝ) ≤ 2^23 := by exact_mod_cast (show Mufu.fraction word ≤ 2^23 by omega)
    norm_num
    linarith
  · rw [ite_eq_right hz,abs_of_nonneg (by positivity)]
    by_cases ht : Mufu.exponent word = 127
    · have hfr : Mufu.fraction word = 0 := by
        have h := Exp2Boundary.magnitude_fields word
        rw [ht] at h
        omega
      rw [ht,hfr]
      norm_num
    · have hp : (2 : ℝ)^((Mufu.exponent word : Int)-150) ≤ (2 : ℝ)^(-24 : Int) := by
        apply zpow_le_zpow_right₀ (by norm_num : (1 : ℝ) ≤ 2)
        omega
      have hfq : (Mufu.fraction word : ℝ) ≤ 2^23 := by exact_mod_cast (show Mufu.fraction word ≤ 2^23 by omega)
      have h := mul_le_mul (show (2^23 : ℝ)+Mufu.fraction word ≤ 2^24 by norm_num;linarith) hp (by positivity) (by positivity)
      norm_num at h ⊢
      exact h

theorem finite_real_range_and_sign (word : Nat) (hw : word < 2^32)
    (hf : Mufu.magnitude word < 0x7f800000) :
    ∃ out, Mufu.tanh word = some out ∧ out < 2^32 ∧
      |Exp2Bounds.finiteValue out| ≤ 1 ∧ Mufu.sign out = Mufu.sign word := by
  obtain ⟨out,ho,hw',hm,hs⟩ := finite_range_and_sign word hw hf
  exact ⟨out,ho,hw',decode_range out hw' hm,hs⟩

#print axioms certificate
#print axioms polynomial_bound
#print axioms fixed_range
#print axioms middle_magnitude
#print axioms finite_magnitude
#print axioms finite_range_and_sign
#print axioms signed_zero_bits
#print axioms decode_range
#print axioms finite_real_range_and_sign
end GB300Math.TanhRange
