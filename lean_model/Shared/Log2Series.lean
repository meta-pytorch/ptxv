import Mathlib.Analysis.SpecialFunctions.Log.Deriv
import Mathlib.Tactic
namespace GB300Math.Log2Series
set_option maxHeartbeats 3000000
set_option maxRecDepth 10000
noncomputable def polynomial (x : ℝ) : ℝ :=
  2*∑ i∈Finset.range 15, ((x-1)/(x+1))^(2*i+1)/(2*i+1)
def polynomialQ (x : Rat) : Rat :=
  2*∑ i∈Finset.range 15, ((x-1)/(x+1))^(2*i+1)/(2*i+1)

theorem approximation (x : ℝ) (hx : 1≤x ∧ x≤2) :
    |Real.log x-polynomial x|≤1/10^14 := by
  let t := (x-1)/(x+1)
  have ht0 : 0≤t := by dsimp [t];exact div_nonneg (by linarith) (by linarith)
  have ht1 : t≤1/3 := by
    dsimp [t]
    apply (div_le_iff₀ (by linarith : 0<x+1)).mpr
    linarith
  have ht : |t|<1 := by rw [abs_of_nonneg ht0];linarith
  have heq : (1+t)/(1-t)=x := by
    apply (div_eq_iff (by linarith : 1-t≠0)).mpr
    dsimp [t]
    field_simp [show x+1≠0 by linarith]
    ring
  have h := Real.sum_range_sub_log_div_le ht 15
  rw [heq,abs_of_nonneg ht0] at h
  have hsq : t^2≤(1/3:ℝ)^2 := pow_le_pow_left₀ ht0 ht1 2
  have hp : t^31≤(1/3:ℝ)^31 := pow_le_pow_left₀ ht0 ht1 31
  have hden : 0<1-t^2 := by nlinarith
  have hb : t^31/(1-t^2)≤1/(2*10^14) := by
    apply (div_le_iff₀ hden).mpr
    norm_num at hp hsq ⊢
    nlinarith
  have he : |Real.log x-polynomial x|=2*|1/2*Real.log x-∑ i∈Finset.range 15,t^(2*i+1)/(2*i+1)| := by
    rw [← abs_of_nonneg (by norm_num : (0:ℝ)≤2),← abs_mul]
    unfold polynomial
    congr 1
    dsimp [t]
    ring
  rw [he]
  norm_num only [Nat.reduceMul,Nat.reduceAdd] at h
  nlinarith

theorem polynomial_cast (x : Rat) : (polynomialQ x:ℝ)=polynomial (x:ℝ) := by
  unfold polynomialQ polynomial
  push_cast
  rfl

#print axioms approximation
#print axioms polynomial_cast
end GB300Math.Log2Series
