import Mathlib.Analysis.SpecialFunctions.Log.Basic
import Mathlib.Tactic
import GB300Math.TextureGradientCertificates

set_option maxRecDepth 1000000
set_option maxHeartbeats 0

namespace GB300Math.TextureGradientBounds

open TextureGradient

theorem log_enclosure (x : ℝ) (q : Nat) (hx : 0 < x)
    (lo : (2:ℝ)^(2*q) ≤ 8*x^512)
    (hi : x^512 ≤ 8*(2:ℝ)^(2*q)) :
    |(q:ℝ)/256-Real.log x/Real.log 2| ≤ 3/512 := by
  have hlog : 0 < Real.log 2 := Real.log_pos (by norm_num)
  have hl := Real.log_le_log (by positivity : (0:ℝ) < 2^(2*q)) lo
  have hh := Real.log_le_log (by positivity : (0:ℝ) < x^512) hi
  have h8 : Real.log 8 = 3*Real.log 2 := by
    calc Real.log 8 = Real.log ((2:ℝ)^3) := by norm_num
         _ = 3*Real.log 2 := by rw [Real.log_pow]; norm_num
  rw [Real.log_mul (by norm_num : (8:ℝ) ≠ 0) (by positivity), Real.log_pow,
      Real.log_pow, h8] at hl
  rw [Real.log_pow, Real.log_mul (by norm_num : (8:ℝ) ≠ 0) (by positivity),
      Real.log_pow, h8] at hh
  push_cast at hl hh
  have hll : (q:ℝ)/256-3/512 ≤ Real.log x/Real.log 2 := by
    apply (le_div_iff₀ hlog).mpr
    nlinarith
  have hhh : Real.log x/Real.log 2 ≤ (q:ℝ)/256+3/512 := by
    apply (div_le_iff₀ hlog).mpr
    nlinarith
  exact abs_le.mpr ⟨by linarith, by linarith⟩

theorem certificate_implies_error (i q : Nat) (x : ℝ) (h : Certificate i q)
    (hlo : (256+(i:ℝ))/256 ≤ x) (hhi : x ≤ (257+(i:ℝ))/256) :
    |(q:ℝ)/256-Real.log x/Real.log 2| ≤ 3/512 := by
  have hi : 0 ≤ (i:ℝ) := Nat.cast_nonneg _
  have hx : 0 < x := by linarith
  have hl : (2:ℝ)^(4096+2*q) ≤ 8*(256+(i:ℝ))^512 := by
    simpa only [Nat.cast_pow, Nat.cast_mul, Nat.cast_add, Nat.cast_ofNat] using
      (Nat.cast_le (α := ℝ)).mpr h.1
  have hh : (257+(i:ℝ))^512 ≤ 8*(2:ℝ)^(4096+2*q) := by
    simpa only [Nat.cast_pow, Nat.cast_mul, Nat.cast_add, Nat.cast_ofNat] using
      (Nat.cast_le (α := ℝ)).mpr h.2
  have hxl : (256+(i:ℝ))^512 ≤ (256*x)^512 :=
    pow_le_pow_left₀ (by positivity) (by linarith) _
  have hxh : (256*x)^512 ≤ (257+(i:ℝ))^512 :=
    pow_le_pow_left₀ (by positivity) (by linarith) _
  have hpow : (256:ℝ)^512 = 2^4096 := by
    rw [show (256:ℝ) = (2:ℝ)^8 by norm_num, ← pow_mul]
  rw [pow_add] at hl hh
  rw [mul_pow, hpow] at hxl hxh
  have hp : (0:ℝ) < 2^4096 := by positivity
  apply log_enclosure x q hx
  · apply (mul_le_mul_iff_right₀ hp).mp
    calc
      (2:ℝ)^4096 * 2^(2*q) ≤ 8*(256+(i:ℝ))^512 := hl
      _ ≤ 8*(2^4096*x^512) := mul_le_mul_of_nonneg_left hxl (by norm_num)
      _ = 2^4096*(8*x^512) := mul_left_comm _ _ _
  · apply (mul_le_mul_iff_right₀ hp).mp
    calc
      (2:ℝ)^4096*x^512 ≤ (257+(i:ℝ))^512 := hxh
      _ ≤ 8*(2^4096*2^(2*q)) := hh
      _ = 2^4096*(8*2^(2*q)) := mul_left_comm _ _ _

theorem all_bins_error (i : Fin 256) (x : ℝ)
    (hlo : (256+(i.val:ℝ))/256 ≤ x) (hhi : x ≤ (257+(i.val:ℝ))/256) :
    |(logTable[i.val]!:ℝ)/256-Real.log x/Real.log 2| ≤ 3/512 :=
  certificate_implies_error i.val _ x
    (all_certificates i.val (List.mem_range.mpr i.isLt)) hlo hhi

#print axioms log_enclosure
#print axioms certificate_implies_error
#print axioms all_bins_error
end GB300Math.TextureGradientBounds
