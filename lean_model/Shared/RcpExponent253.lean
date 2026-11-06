import RcpBounds
import RcpFtzBoundary

namespace GB300Math.RcpExponent253

def inputWord (negative : Bool) (fraction : Nat) : Nat :=
  (if negative then 2^31 else 0) + 253*2^23 + fraction

theorem input_fields (negative : Bool) (fraction : Nat) (hf : fraction < 2^23) :
    Mufu.exponent (inputWord negative fraction) = 253 ∧
    Mufu.fraction (inputWord negative fraction) = fraction ∧
    Mufu.magnitude (inputWord negative fraction) < 0x7f800000 := by
  cases negative <;> simp only [inputWord, Bool.false_eq_true, ↓reduceIte] <;>
    dsimp [Mufu.exponent, Mufu.fraction, Mufu.magnitude] <;> omega

theorem top_mantissa_iff_zero (fraction : Nat) (hf : fraction < 2^23) :
    (Mufu.interpolate MufuTables.rcp fraction).toNat = 2^24 ↔ fraction = 0 := by
  constructor
  · intro h
    have hr := RcpBounds.interpolate_range fraction hf
    have he := (RcpBounds.interpolate_error fraction hf).2
    have hz : Mufu.interpolate MufuTables.rcp fraction = 2^24 := by omega
    rw [hz] at he
    norm_num at he
    nlinarith
  · rintro rfl
    decide +kernel

theorem raw_boundary (negative : Bool) (fraction : Nat) (hf : fraction < 2^23) :
    Mufu.rcp (inputWord negative fraction) =
      (if negative then 2^31 else 0) + (if fraction = 0 then 2^23 else 0) := by
  have h := input_fields negative fraction hf
  have hs : Mufu.sign (inputWord negative fraction) = (if negative then 2^31 else 0) := by
    cases negative
    · simp only [inputWord, Bool.false_eq_true, ↓reduceIte, Nat.zero_add]
      exact RcpFtzBoundary.small_sign _ (by omega)
    · simp only [inputWord, ↓reduceIte]
      rw [Nat.add_assoc]
      exact RcpFtzBoundary.high_sign _ (by omega)
  have hi := top_mantissa_iff_zero fraction hf
  by_cases hz : fraction = 0
  · have hm := hi.mpr hz
    subst fraction
    simp [Mufu.rcp, h.1, h.2.1, not_lt.mpr (le_of_lt h.2.2), hm, hs]
  · have hm := mt hi.mp hz
    norm_num at hm
    simp [Mufu.rcp, h.1, h.2.1, not_lt.mpr (le_of_lt h.2.2), hm, hs, hz]

theorem positive_fraction_relative_error_one (negative : Bool) (fraction : Nat)
    (hf : fraction < 2^23) (hp : 0 < fraction) :
    |RcpFtzBoundary.finiteValue (Mufu.rcp (inputWord negative fraction)) *
      RcpFtzBoundary.finiteValue (inputWord negative fraction) - 1| = 1 := by
  have hz : fraction ≠ 0 := by omega
  rw [raw_boundary negative fraction hf]
  simp only [hz, ite_false, Nat.add_zero]
  cases negative <;> norm_num [RcpFtzBoundary.finiteValue, Mufu.exponent, Mufu.fraction]

theorem exact_at_zero_fraction (negative : Bool) :
    RcpFtzBoundary.finiteValue (Mufu.rcp (inputWord negative 0)) *
      RcpFtzBoundary.finiteValue (inputWord negative 0) = 1 := by
  rw [raw_boundary negative 0 (by decide)]
  cases negative <;> norm_num [inputWord, RcpFtzBoundary.finiteValue, Mufu.exponent, Mufu.fraction]

theorem finite_input_nonzero (negative : Bool) (fraction : Nat) (hf : fraction < 2^23) :
    RcpFtzBoundary.finiteValue (inputWord negative fraction) ≠ 0 := by
  have h := input_fields negative fraction hf
  have hp : (0 : ℚ) < 2^23+(fraction : ℚ) := by positivity
  unfold RcpFtzBoundary.finiteValue
  rw [h.1,h.2.1]
  norm_num
  split <;> nlinarith

theorem reciprocal_relative_error_one (negative : Bool) (fraction : Nat)
    (hf : fraction < 2^23) (hp : 0 < fraction) :
    |RcpFtzBoundary.finiteValue (Mufu.rcp (inputWord negative fraction)) -
      1 / RcpFtzBoundary.finiteValue (inputWord negative fraction)| /
      |1 / RcpFtzBoundary.finiteValue (inputWord negative fraction)| = 1 := by
  have hz : fraction ≠ 0 := by omega
  have hvalue : RcpFtzBoundary.finiteValue (Mufu.rcp (inputWord negative fraction)) = 0 := by
    rw [raw_boundary negative fraction hf]
    simp only [hz, ite_false, Nat.add_zero]
    cases negative <;> norm_num [RcpFtzBoundary.finiteValue, Mufu.exponent, Mufu.fraction]
  rw [hvalue, zero_sub, abs_neg]
  exact div_self (abs_ne_zero.mpr (one_div_ne_zero (finite_input_nonzero negative fraction hf)))

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem first_positive_output : Mufu.rcp 0x7e800000 = 0x00800000 := by decide +kernel

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem next_positive_flushes : Mufu.rcp 0x7e800001 = 0 := by decide +kernel

#print axioms top_mantissa_iff_zero
#print axioms raw_boundary
#print axioms positive_fraction_relative_error_one
#print axioms exact_at_zero_fraction
#print axioms reciprocal_relative_error_one
#print axioms first_positive_output
#print axioms next_positive_flushes
end GB300Math.RcpExponent253
