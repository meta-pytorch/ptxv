import Exp2Cells
namespace GB300Math.Exp2Local
open Exp2Analytic Exp2Cells
set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

theorem center_error (j : Fin 256) :
    |Real.exp ((logRat:ℝ)*j.val/256)-(center j.val:ℝ)|≤2/10^12 := by
  have hj : (0:ℝ)≤j.val ∧ (j.val:ℝ)≤255 := by exact_mod_cast (show 0≤j.val ∧ j.val≤255 by have:=j.isLt;omega)
  have ha : |(logRat:ℝ)*j.val/256|≤1 := by norm_num [logRat];rw [abs_of_nonneg (by positivity)];nlinarith
  have h:=series_error _ ha
  have hc:= (cells_valid j).1
  have hr : |series ((logRat:ℝ)*j.val/256)-(center j.val:ℝ)|≤1/10^12 := by
    have hh:=(Rat.cast_le (K:=ℝ)).mpr hc
    simpa only [Rat.cast_abs,Rat.cast_sub,Rat.cast_div,Rat.cast_one,Rat.cast_pow,Rat.cast_ofNat,series_cast,Rat.cast_mul,Rat.cast_natCast] using hh
  have ht:=abs_sub_le (Real.exp ((logRat:ℝ)*j.val/256)) (series ((logRat:ℝ)*j.val/256)) (center j.val:ℝ)
  linarith

theorem local_ideal (j : Fin 256) (t : ℝ) (ht : 0≤t ∧ t≤1) :
    |localTaylor j.val t-Real.exp (Real.log 2*((j.val:ℝ)+t)/256)|≤21/10^9 := by
  have hj : (0:ℝ)≤j.val ∧ (j.val:ℝ)≤255 := by exact_mod_cast (show 0≤j.val ∧ j.val≤255 by have:=j.isLt;omega)
  have hl : 0≤(logRat:ℝ) ∧ (logRat:ℝ)≤3/4 := by norm_num [logRat]
  have ha : (logRat:ℝ)*j.val/256≤1 := by nlinarith
  have hd : 0≤(logRat:ℝ)/256*t ∧ (logRat:ℝ)/256*t≤3/1024 := by constructor <;> nlinarith
  have h:=local_error _ _ _ ha hd (center_error j)
  change |Real.exp ((logRat:ℝ)*j.val/256+(logRat:ℝ)/256*t)-localTaylor j.val t|≤_ at h
  have hab : |Real.log 2*((j.val:ℝ)+t)/256-((logRat:ℝ)*j.val/256+(logRat:ℝ)/256*t)|≤2/10^14 := by
    have hi : Real.log 2*((j.val:ℝ)+t)/256-((logRat:ℝ)*j.val/256+(logRat:ℝ)/256*t)=(Real.log 2-(logRat:ℝ))*(((j.val:ℝ)+t)/256) := by ring
    rw [hi,abs_mul,abs_of_nonneg (show 0≤((j.val:ℝ)+t)/256 by linarith)]
    have hh:=mul_le_mul_of_nonneg_right log_rat_error (show 0≤((j.val:ℝ)+t)/256 by linarith)
    nlinarith
  have ha' : (logRat:ℝ)*j.val/256+(logRat:ℝ)/256*t≤1 := by nlinarith
  have hp:=perturb _ _ ha' (show |Real.log 2*((j.val:ℝ)+t)/256-((logRat:ℝ)*j.val/256+(logRat:ℝ)/256*t)|≤1 by linarith)
  have htri:=abs_sub_le (localTaylor j.val t) (Real.exp ((logRat:ℝ)*j.val/256+(logRat:ℝ)/256*t)) (Real.exp (Real.log 2*((j.val:ℝ)+t)/256))
  have hrev : |localTaylor j.val t-Real.exp ((logRat:ℝ)*j.val/256+(logRat:ℝ)/256*t)|≤1/50000000 := by rwa [abs_sub_comm]
  have hprev : |Real.exp ((logRat:ℝ)*j.val/256+(logRat:ℝ)/256*t)-Real.exp (Real.log 2*((j.val:ℝ)+t)/256)|≤12/10^14 := by rw [abs_sub_comm];linarith
  linarith

theorem polynomial_ideal (j : Fin 256) (t : ℝ) (ht : 0≤t ∧ t≤1) :
    (29/10^9:ℝ)≤nativePoly j.val t-Real.exp (Real.log 2*((j.val:ℝ)+t)/256) ∧
    nativePoly j.val t-Real.exp (Real.log 2*((j.val:ℝ)+t)/256)≤196/10^9 := by
  have hp:=polynomial_between j t ht
  have hlow:=polynomial_lower j t ht
  have hl:=abs_le.mp (local_ideal j t ht)
  constructor <;> linarith

#print axioms center_error
#print axioms local_ideal
#print axioms polynomial_ideal
end GB300Math.Exp2Local
