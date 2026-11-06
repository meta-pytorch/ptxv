import Mathlib.Tactic

namespace GB300Math.TextureSeamlessBounds

def corner (a b c : ℚ) : ℚ := (86*a+85*b+85*c)/256

def ideal (a b c : ℚ) : ℚ := (a+b+c)/3

theorem corner_bias (a b c : ℚ) :
    corner a b c - ideal a b c = ((a-b)+(a-c))/768 := by
  unfold corner ideal
  ring

theorem corner_error (a b c : ℚ) :
    |corner a b c - ideal a b c| ≤ (|a-b|+|a-c|)/768 := by
  rw [corner_bias, abs_div, abs_of_pos (by norm_num : (0:ℚ)<768)]
  exact div_le_div_of_nonneg_right (abs_add_le (a-b) (a-c)) (by norm_num)

theorem range_error (a b c lo hi : ℚ)
    (ha : lo ≤ a ∧ a ≤ hi) (hb : lo ≤ b ∧ b ≤ hi) (hc : lo ≤ c ∧ c ≤ hi) :
    |corner a b c - ideal a b c| ≤ (hi-lo)/384 := by
  have ab : |a-b| ≤ hi-lo := abs_le.mpr ⟨by linarith, by linarith⟩
  have ac : |a-c| ≤ hi-lo := abs_le.mpr ⟨by linarith, by linarith⟩
  have h := corner_error a b c
  linarith

theorem tight_range_error (lo hi : ℚ) (h : lo ≤ hi) :
    |corner hi lo lo - ideal hi lo lo| = (hi-lo)/384 := by
  rw [corner_bias]
  rw [abs_of_nonneg (by positivity : (0:ℚ) ≤ ((hi-lo)+(hi-lo))/768)]
  ring

theorem constant_preserved (a : ℚ) : corner a a a = a := by
  unfold corner
  ring

theorem coefficient_redistribution (w : Nat) :
    w - 2*((w+1)/3) + (w+1)/3 + (w+1)/3 = w := by omega

theorem coefficient_remainder (w : Nat) :
    -1 ≤ (w:Int) - 3*((w+1)/3:Nat) ∧ (w:Int) - 3*((w+1)/3:Nat) ≤ 1 := by omega

theorem truncation_error (n d : Nat) (hd : 0 < d) :
    |((n/d : Nat) : ℚ) - (n : ℚ)/d| < 1 := by
  have hd' : (0:ℚ) < d := by exact_mod_cast hd
  have hl : ((n/d : Nat) : ℚ)*d ≤ n := by exact_mod_cast Nat.div_mul_le_self n d
  have hu : (n : ℚ) < (((n/d : Nat) : ℚ)+1)*d := by
    have h := Nat.lt_mul_div_succ n hd
    rw [Nat.mul_comm d] at h
    exact_mod_cast h
  have lower : ((n/d : Nat) : ℚ) ≤ (n:ℚ)/d := (le_div_iff₀ hd').2 hl
  rw [abs_of_nonpos (sub_nonpos.mpr lower)]
  have upper : (n:ℚ)/d < ((n/d:Nat):ℚ)+1 := (div_lt_iff₀ hd').2 hu
  linarith

theorem weighted_alignment_error {n : Nat} (w x a : Fin n → ℚ) (step : ℚ)
    (hw : ∀ i, 0 ≤ w i) (sum_weights : ∑ i, w i = 256)
    (error : ∀ i, |a i-x i| ≤ step) :
    |(∑ i, w i*a i)/256 - (∑ i, w i*x i)/256| ≤ step := by
  have bound : |∑ i, w i*(a i-x i)| ≤ 256*step := by
    calc
      |∑ i, w i*(a i-x i)| ≤ ∑ i, |w i*(a i-x i)| := Finset.abs_sum_le_sum_abs _ _
      _ = ∑ i, w i*|a i-x i| := by simp only [abs_mul, abs_of_nonneg (hw _)]
      _ ≤ ∑ i, w i*step := Finset.sum_le_sum (fun i _ => mul_le_mul_of_nonneg_left (error i) (hw i))
      _ = 256*step := by rw [← Finset.sum_mul, sum_weights]
  have identity : (∑ i, w i*a i)/256 - (∑ i, w i*x i)/256 =
      (∑ i, w i*(a i-x i))/256 := by
    simp only [mul_sub, Finset.sum_sub_distrib]
    ring
  rw [identity, abs_div]
  norm_num
  linarith

theorem final_rounding_error {n : Nat} (w x a : Fin n → ℚ) (step output halfUlp : ℚ)
    (hw : ∀ i, 0 ≤ w i) (sum_weights : ∑ i, w i = 256)
    (alignment : ∀ i, |a i-x i| ≤ step)
    (rounding : |output-(∑ i, w i*a i)/256| ≤ halfUlp) :
    |output-(∑ i, w i*x i)/256| ≤ step+halfUlp := by
  have h := weighted_alignment_error w x a step hw sum_weights alignment
  have triangle := abs_add_le (output-(∑ i, w i*a i)/256)
      ((∑ i, w i*a i)/256-(∑ i, w i*x i)/256)
  have identity : output-(∑ i, w i*x i)/256 =
      (output-(∑ i, w i*a i)/256)+((∑ i, w i*a i)/256-(∑ i, w i*x i)/256) := by ring
  rw [identity]
  linarith

theorem subnormal_flush_error (fraction : Nat) (negative : Bool)
    (hf : fraction < 2^23) :
    |(if negative then -((fraction:ℚ)/2^149) else (fraction:ℚ)/2^149)| < 1/2^126 := by
  have nonneg : (0:ℚ) ≤ (fraction:ℚ)/2^149 := by positivity
  have bound : (fraction:ℚ)/2^149 < 1/2^126 := by
    have h : (fraction:ℚ) < 2^23 := by exact_mod_cast hf
    calc
      (fraction:ℚ)/2^149 < 2^23/2^149 := div_lt_div_of_pos_right h (by positivity)
      _ = 1/2^126 := by norm_num
  cases negative <;> simp only [Bool.false_eq_true, ↓reduceIte, abs_neg, abs_of_nonneg nonneg] <;> exact bound

end GB300Math.TextureSeamlessBounds
#print axioms GB300Math.TextureSeamlessBounds.corner_bias
#print axioms GB300Math.TextureSeamlessBounds.corner_error
#print axioms GB300Math.TextureSeamlessBounds.range_error
#print axioms GB300Math.TextureSeamlessBounds.tight_range_error
#print axioms GB300Math.TextureSeamlessBounds.constant_preserved
#print axioms GB300Math.TextureSeamlessBounds.coefficient_redistribution
#print axioms GB300Math.TextureSeamlessBounds.coefficient_remainder

#print axioms GB300Math.TextureSeamlessBounds.truncation_error
#print axioms GB300Math.TextureSeamlessBounds.weighted_alignment_error
#print axioms GB300Math.TextureSeamlessBounds.final_rounding_error

#print axioms GB300Math.TextureSeamlessBounds.subnormal_flush_error
