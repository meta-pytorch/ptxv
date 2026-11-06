import SineTaylor
import GB300Math.Mufu
namespace GB300Math.SinePhase
set_option maxHeartbeats 3000000

def idealIndex (q : Nat) : Nat := if q/2^23%2=1 then 2^23-q%2^23 else q%2^23
def tableIndex (q : Nat) : Nat := if q/2^23%2=1 then 2^23-1-q%2^23 else q%2^23
noncomputable def phaseSign (q : Nat) : ℝ := if q/2^24=1 then -1 else 1

theorem index_bounds (q : Nat) (_hq : q < 2^25) :
    tableIndex q < 2^23 ∧ idealIndex q ≤ 2^23 ∧
    tableIndex q ≤ idealIndex q ∧ idealIndex q ≤ tableIndex q+1 := by
  unfold idealIndex tableIndex
  split_ifs <;> omega

theorem ideal_fold (q : Nat) (hq : q < 2^25) :
    Real.sin (Real.pi*q/2^24) = phaseSign q*Real.sin (Real.pi*idealIndex q/2^24) := by
  have hr : q%2^23 < 2^23 := Nat.mod_lt q (by decide)
  have hd : q/2^23 < 4 := by omega
  generalize hrem : q%2^23 = r at *
  generalize hquot : q/2^23 = k at *
  have hqeq : q=k*2^23+r := by omega
  have hc : (q:ℝ)=(k:ℝ)*2^23+r := by exact_mod_cast hqeq
  interval_cases k
  · have hsign : q/2^24≠1 := by omega
    simp only [phaseSign,idealIndex,hquot,hrem,hsign,ite_false]
    norm_num
    rw [hc]
    norm_num
  · have hsign : q/2^24≠1 := by omega
    have hi : ((2^23-r:Nat):ℝ) = 2^23-(r:ℝ) := by rw [Nat.cast_sub (by omega)];norm_num
    simp only [phaseSign,idealIndex,hquot,hrem,hsign,ite_false]
    norm_num only [Nat.reduceMod,ite_true,one_mul]
    norm_num only [Nat.reducePow] at hi
    rw [hi,hc]
    norm_num only [Nat.cast_ofNat]
    convert Real.sin_pi_sub (Real.pi*(8388608-(r:ℝ))/16777216) using 1; congr 1; ring
  · have hsign : q/2^24=1 := by omega
    simp only [phaseSign,idealIndex,hquot,hrem,hsign,ite_true]
    norm_num only [Nat.reduceMod,OfNat.zero_ne_ofNat,ite_false,neg_mul,one_mul]
    rw [hc]
    norm_num only [Nat.cast_ofNat]
    convert Real.sin_add_pi (Real.pi*(r:ℝ)/16777216) using 1; congr 1; ring
  · have hsign : q/2^24=1 := by omega
    have hi : ((2^23-r:Nat):ℝ) = 2^23-(r:ℝ) := by rw [Nat.cast_sub (by omega)];norm_num
    simp only [phaseSign,idealIndex,hquot,hrem,hsign,ite_true]
    norm_num only [Nat.reduceMod,neg_mul,one_mul]
    norm_num only [Nat.reducePow] at hi
    rw [hi,hc]
    norm_num only [Nat.cast_ofNat]
    have h := Real.sin_add_two_pi (-(Real.pi*(8388608-(r:ℝ))/16777216))
    rw [Real.sin_neg] at h
    convert h using 1; congr 1; ring

theorem reflection_error (q : Nat) (hq : q < 2^25) :
    |Real.sin (Real.pi*idealIndex q/2^24)-Real.sin (Real.pi*tableIndex q/2^24)| ≤ Real.pi/2^24 := by
  have hb := index_bounds q hq
  have hlo : (tableIndex q:ℝ) ≤ idealIndex q := by exact_mod_cast hb.2.2.1
  have hhi : (idealIndex q:ℝ) ≤ (tableIndex q:ℝ)+1 := by exact_mod_cast hb.2.2.2
  have h := Real.abs_sin_sub_sin_le (Real.pi*idealIndex q/2^24) (Real.pi*tableIndex q/2^24)
  have hp := Real.pi_pos
  rw [abs_of_nonneg (by nlinarith : 0 ≤ Real.pi*idealIndex q/2^24-Real.pi*tableIndex q/2^24)] at h
  nlinarith
#print axioms ideal_fold
#print axioms reflection_error
end GB300Math.SinePhase
