import GB300Math.TextureGradient1D
import GB300Math.StochasticValue
import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic

namespace GB300Math.TextureGradientNormBounds

noncomputable def octagon (x y : ℝ) : ℝ := max (max x y) (11/16*(x+y))

theorem octagon_square_upper (x y : ℝ) :
    (octagon x y)^2 ≤ x^2+y^2 := by
  unfold octagon
  rcases le_total x y with hxy | hxy
  · rw [max_eq_right hxy]
    rcases le_total y (11/16*(x+y)) with h | h
    · rw [max_eq_right h]
      nlinarith [sq_nonneg (x-y)]
    · rw [max_eq_left h]
      nlinarith [sq_nonneg x]
  · rw [max_eq_left hxy]
    rcases le_total x (11/16*(x+y)) with h | h
    · rw [max_eq_right h]
      nlinarith [sq_nonneg (x-y)]
    · rw [max_eq_left h]
      nlinarith [sq_nonneg y]

theorem octagon_square_lower (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
    121*(x^2+y^2) ≤ 146*(octagon x y)^2 := by
  unfold octagon
  rcases le_total x y with hxy | hxy
  · rw [max_eq_right hxy]
    rcases le_total y (11/16*(x+y)) with h | h
    · rw [max_eq_right h]
      have h1 : 0 ≤ (11*x-5*y)*(y-x) := mul_nonneg (by linarith) (by linarith)
      nlinarith
    · rw [max_eq_left h]
      have h1 : 0 ≤ (5*y-11*x)*(5*y+11*x) := mul_nonneg (by linarith) (by positivity)
      nlinarith
  · rw [max_eq_left hxy]
    rcases le_total x (11/16*(x+y)) with h | h
    · rw [max_eq_right h]
      have h1 : 0 ≤ (11*y-5*x)*(x-y) := mul_nonneg (by linarith) (by linarith)
      nlinarith
    · rw [max_eq_left h]
      have h1 : 0 ≤ (5*x-11*y)*(5*x+11*y) := mul_nonneg (by linarith) (by positivity)
      nlinarith

theorem finite_counterexample :
    TextureGradient1D.position 0x3bb00000 0x3b200000 = 118 := by decide +kernel

theorem counterexample_output :
    TextureGradient1D.sample 0x3bb00000 0x3b200000 = 0x3fbb0000 := by decide +kernel

theorem counterexample_inputs :
    StochasticValue.rationalValue 8 23 127 0x7f7fffff 0x3bb00000 = some (11/2048) ∧
    StochasticValue.rationalValue 8 23 127 0x7f7fffff 0x3b200000 = some (5/2048) := by
  decide +kernel

theorem counterexample_decoded_output :
    StochasticValue.rationalValue 8 23 127 0x7f7fffff
      (TextureGradient1D.sample 0x3bb00000 0x3b200000) = some (1+118/256) := by
  rw [counterexample_output]
  decide +kernel

theorem scaled_squared_norm :
    ((256:ℝ)*(11/2048))^2 + (256*(5/2048))^2 = 146/64 := by norm_num

noncomputable def euclideanLod : ℝ := Real.log 146/(2*Real.log 2)-3

theorem euclidean_lod_gap : (1:ℝ)/8 < euclideanLod-118/256 := by
  have hl2 : 0 < Real.log (2:ℝ) := Real.log_pos (by norm_num)
  have hp : (2:ℝ)^459 < (146:ℝ)^64 := by
    exact_mod_cast (show (2:Nat)^459 < 146^64 by decide +kernel)
  have hl := Real.log_lt_log (by positivity : (0:ℝ) < 2^459) hp
  rw [Real.log_pow, Real.log_pow] at hl
  have hh : ((75:ℝ)/128+3)*(2*Real.log 2) < Real.log 146 := by norm_num at hl; nlinarith
  have hd : (75:ℝ)/128+3 < Real.log 146/(2*Real.log 2) :=
    (lt_div_iff₀ (by positivity : (0:ℝ) < 2*Real.log 2)).2 hh
  unfold euclideanLod
  linarith

#print axioms counterexample_inputs
#print axioms counterexample_decoded_output
#print axioms scaled_squared_norm
#print axioms octagon_square_upper
#print axioms octagon_square_lower
#print axioms finite_counterexample
#print axioms counterexample_output
#print axioms euclidean_lod_gap
end GB300Math.TextureGradientNormBounds
