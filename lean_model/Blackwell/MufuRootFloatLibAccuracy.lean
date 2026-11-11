import MufuRsqTightRealBounds

namespace GB300Math.MufuRootBounds

open MufuReciprocalBounds

theorem floatLib_positive_value (bf16 : Bool) (word : Nat)
    (finite : word < infinity bf16)
    (checked : decodeMatches bf16 word = true) :
    (floatLibValue bf16 word).map (fun q : ℚ => (q : ℝ)) =
      some (magnitudeValue bf16 word) := by
  have small : word < 32768 := by cases bf16 <;> simp [infinity] at finite <;> omega
  have zero : word / 32768 = 0 := Nat.div_eq_of_lt small
  have finite' : word % 32768 < infinity bf16 := by simpa [Nat.mod_eq_of_lt small] using finite
  simpa [zero] using floatLib_real_value bf16 word finite' checked

theorem checked_floatLib_sqrt_relative (bf16 : Bool) (word : Nat)
    (inside : domain bf16 word (output bf16 false word) = true)
    (checked : valid bf16 false word = true)
    (input_decode : decodeMatches bf16 word = true)
    (output_decode : decodeMatches bf16 (output bf16 false word) = true) :
    ∃ x y : ℝ,
      (floatLibValue bf16 word).map (fun q : ℚ => (q : ℝ)) = some x ∧
      (floatLibValue bf16 (output bf16 false word)).map (fun q : ℚ => (q : ℝ)) = some y ∧
      |y-Real.sqrt x| / |Real.sqrt x| ≤
        1-Real.sqrt (((lower bf16 false).1 : ℝ)/(lower bf16 false).2) := by
  have h := inside
  simp only [domain, Bool.and_eq_true, decide_eq_true_eq] at h
  exact ⟨_, _, floatLib_positive_value bf16 word h.1.1.2 input_decode,
    floatLib_positive_value bf16 _ h.2 output_decode,
    checked_sqrt_relative bf16 word inside checked⟩

theorem checked_floatLib_rsq_relative (bf16 : Bool) (word : Nat)
    (inside : domain bf16 word (output bf16 true word) = true)
    (checked : MufuRsqTightBounds.valid bf16 word = true)
    (input_decode : decodeMatches bf16 word = true)
    (output_decode : decodeMatches bf16 (output bf16 true word) = true) :
    ∃ x y : ℝ,
      (floatLibValue bf16 word).map (fun q : ℚ => (q : ℝ)) = some x ∧
      (floatLibValue bf16 (output bf16 true word)).map (fun q : ℚ => (q : ℝ)) = some y ∧
      |y-1/Real.sqrt x| / |1/Real.sqrt x| ≤
        1-Real.sqrt (((MufuRsqTightBounds.lower bf16).1 : ℝ)/(MufuRsqTightBounds.lower bf16).2) := by
  have h := inside
  simp only [domain, Bool.and_eq_true, decide_eq_true_eq] at h
  exact ⟨_, _, floatLib_positive_value bf16 word h.1.1.2 input_decode,
    floatLib_positive_value bf16 _ h.2 output_decode,
    MufuRsqTightBounds.checked_rsq_relative bf16 word inside checked⟩

#print axioms floatLib_positive_value
#print axioms checked_floatLib_sqrt_relative
#print axioms checked_floatLib_rsq_relative

end GB300Math.MufuRootBounds
