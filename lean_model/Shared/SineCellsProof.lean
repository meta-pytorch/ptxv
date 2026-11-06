import SineCellData
import SineLocal
namespace GB300Math.SineCells
set_option maxRecDepth 12000
set_option maxHeartbeats 4000000

theorem polynomial_cast (x : Rat) : (polynomialQ x : ℝ) = SineTaylor.polynomial (x:ℝ) := by
  simp [polynomialQ,SineTaylor.polynomial]

theorem center_sin (j : Fin 256) :
    |Real.sin ((SineTaylor.piRat:ℝ)*j.val/512)-(centerSin j.val:ℝ)| ≤ 1/1000000000 := by
  have hj := j.isLt
  have h := cells_valid j
  dsimp only [CellValid] at h
  have hlo : -(1/1000000000000:ℝ) ≤ (centerSin j.val:ℝ)-SineTaylor.polynomial ((SineTaylor.piRat:ℝ)*j.val/512) := by
    have hh : (polynomialQ (SineTaylor.piRat*j.val/512):ℝ) = SineTaylor.polynomial ((SineTaylor.piRat:ℝ)*j.val/512) := by simpa using polynomial_cast (SineTaylor.piRat*j.val/512)
    rw [←hh]
    simpa using (Rat.cast_le (K:=ℝ)).mpr h.1
  have hhi : (centerSin j.val:ℝ)-SineTaylor.polynomial ((SineTaylor.piRat:ℝ)*j.val/512) ≤ (1/1000000000000:ℝ) := by
    have hh : (polynomialQ (SineTaylor.piRat*j.val/512):ℝ) = SineTaylor.polynomial ((SineTaylor.piRat:ℝ)*j.val/512) := by simpa using polynomial_cast (SineTaylor.piRat*j.val/512)
    rw [←hh]
    simpa using (Rat.cast_le (K:=ℝ)).mpr h.2.1
  have hx : 0 ≤ (SineTaylor.piRat:ℝ)*j.val/512 ∧ (SineTaylor.piRat:ℝ)*j.val/512 ≤ 2 := by
    have hjr : (j.val:ℝ) ≤ 256 := by exact_mod_cast (show j.val ≤ 256 by omega)
    norm_num [SineTaylor.piRat]
    constructor <;> nlinarith [show (0:ℝ) ≤ j.val by positivity]
  have ha := SineTaylor.approximation _ hx
  have hab := abs_le.mp ha
  exact abs_le.mpr ⟨by linarith,by linarith⟩

theorem center_cos (j : Fin 256) :
    |Real.cos ((SineTaylor.piRat:ℝ)*j.val/512)-(centerCos j.val:ℝ)| ≤ 1/1000000000 := by
  have hj := j.isLt
  have h := cells_valid j
  dsimp only [CellValid] at h
  let x : Rat := SineTaylor.piRat*(256-j.val)/512
  have hc : |(centerCos j.val:ℝ)-SineTaylor.polynomial (x:ℝ)| ≤ 1/1000000000000 := by
    rw [←polynomial_cast]
    simpa using (Rat.cast_le (K:=ℝ)).mpr (abs_le.mpr ⟨h.2.2.1,h.2.2.2.1⟩ : |centerCos j.val-polynomialQ x| ≤ 1/1000000000000)
  have hjr : (j.val:ℝ) ≤ 256 := by exact_mod_cast (show j.val ≤ 256 by omega)
  have hx : 0 ≤ (x:ℝ) ∧ (x:ℝ) ≤ 2 := by
    dsimp [x]
    norm_num [SineTaylor.piRat]
    constructor <;> nlinarith [show (0:ℝ) ≤ j.val by positivity]
  have ha := SineTaylor.approximation _ hx
  have hp := Real.abs_sin_sub_sin_le (Real.pi/2-(SineTaylor.piRat:ℝ)*j.val/512) (x:ℝ)
  have hid : Real.pi/2-(SineTaylor.piRat:ℝ)*j.val/512-(x:ℝ) = (Real.pi-(SineTaylor.piRat:ℝ))/2 := by
    dsimp [x]
    push_cast
    ring
  rw [hid,abs_div] at hp
  rw [Real.sin_pi_div_two_sub] at hp
  have hpi := SineTaylor.pi_approx
  have t1 := abs_sub_le (Real.cos ((SineTaylor.piRat:ℝ)*j.val/512)) (Real.sin (x:ℝ)) (SineTaylor.polynomial (x:ℝ))
  have t2 := abs_sub_le (Real.cos ((SineTaylor.piRat:ℝ)*j.val/512)) (SineTaylor.polynomial (x:ℝ)) (centerCos j.val:ℝ)
  rw [abs_sub_comm (SineTaylor.polynomial (x:ℝ))] at t2
  norm_num at hp
  linarith

#print axioms center_sin
#print axioms center_cos
end GB300Math.SineCells
