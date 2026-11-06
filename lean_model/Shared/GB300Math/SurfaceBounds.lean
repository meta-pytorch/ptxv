import GB300Math.SurfaceNumeric

namespace GB300Math.SurfaceBounds

theorem fixed_and_round_error (x : ℚ) (t maximum halfScale : Nat)
    (hm : 0 < maximum) (hs : 0 < halfScale)
    (lower : (t : ℚ) ≤ x * (2*halfScale))
    (upper : x * (2*halfScale) ≤ t+1) :
    let q := (t*maximum + halfScale-1) / (2*halfScale)
    |(q : ℚ)*(2*halfScale) - x*maximum*(2*halfScale)| ≤ maximum+halfScale := by
  have hr := SurfaceNumeric.round_error (t*maximum) halfScale hs
  dsimp at hr ⊢
  have h₁ : (((t*maximum+halfScale-1)/(2*halfScale) * (2*halfScale) : Nat) : ℚ) -
      (t*maximum : Nat) ≤ halfScale := by exact_mod_cast hr.1
  have h₂ : ((t*maximum : Nat) : ℚ) -
      ((t*maximum+halfScale-1)/(2*halfScale) * (2*halfScale) : Nat) ≤ halfScale := by exact_mod_cast hr.2
  push_cast at h₁ h₂
  have hl := mul_le_mul_of_nonneg_right lower (show (0 : ℚ) ≤ maximum by positivity)
  have hu := mul_le_mul_of_nonneg_right upper (show (0 : ℚ) ≤ maximum by positivity)
  apply abs_le.mpr
  constructor <;> nlinarith

theorem normalized_error (x : ℚ) (t maximum halfScale : Nat)
    (hm : 0 < maximum) (hs : 0 < halfScale)
    (lower : (t : ℚ) ≤ x * (2*halfScale))
    (upper : x * (2*halfScale) ≤ t+1) :
    let q := (t*maximum + halfScale-1) / (2*halfScale)
    |(q : ℚ)/maximum - x| ≤ 1/(2*halfScale) + 1/(2*maximum) := by
  have h := fixed_and_round_error x t maximum halfScale hm hs lower upper
  dsimp at h ⊢
  have hm' : (0 : ℚ) < maximum := by exact_mod_cast hm
  have hs' : (0 : ℚ) < halfScale := by exact_mod_cast hs
  have hd : (0 : ℚ) < (maximum : ℚ)*(2*halfScale) := by positivity
  have identity :
      (((t*maximum+halfScale-1)/(2*halfScale) : Nat) : ℚ)/maximum - x =
      ((((t*maximum+halfScale-1)/(2*halfScale) : Nat) : ℚ)*(2*halfScale) - x*maximum*(2*halfScale)) /
        (maximum*(2*halfScale)) := by field_simp
  rw [identity, abs_div, abs_of_pos hd]
  apply (div_le_iff₀ hd).2
  convert h using 1
  field_simp

theorem rational_value_error (numerator divisor maximum halfScale : Nat)
    (hd : 0 < divisor) (hm : 0 < maximum) (hs : 0 < halfScale) :
    let q := ((numerator/divisor)*maximum + halfScale-1)/(2*halfScale)
    |(q : ℚ)/maximum - (numerator : ℚ)/(divisor*(2*halfScale))| ≤
      1/(2*halfScale) + 1/(2*maximum) := by
  have hd' : (0 : ℚ) < divisor := by exact_mod_cast hd
  have hs' : (0 : ℚ) < halfScale := by exact_mod_cast hs
  have hl := Nat.div_mul_le_self numerator divisor
  have hu := Nat.lt_mul_div_succ numerator hd
  rw [Nat.mul_comm divisor] at hu
  have hlow : ((numerator/divisor : Nat) : ℚ)*divisor ≤ numerator := by exact_mod_cast hl
  have hupp : (numerator : ℚ) ≤ (((numerator/divisor : Nat) : ℚ)+1)*divisor := by
    exact_mod_cast Nat.le_of_lt hu
  apply normalized_error _ _ maximum halfScale hm hs
  · have eqn : (numerator : ℚ)/(divisor*(2*halfScale))*(2*halfScale) = numerator/divisor := by field_simp
    rw [eqn]
    exact (le_div_iff₀ hd').2 hlow
  · have eqn : (numerator : ℚ)/(divisor*(2*halfScale))*(2*halfScale) = numerator/divisor := by field_simp
    rw [eqn]
    exact (div_le_iff₀ hd').2 hupp

end GB300Math.SurfaceBounds

#print axioms GB300Math.SurfaceBounds.fixed_and_round_error
#print axioms GB300Math.SurfaceBounds.normalized_error
#print axioms GB300Math.SurfaceBounds.rational_value_error
