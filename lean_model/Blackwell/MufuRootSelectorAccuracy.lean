import MufuRootFloatLibUniversal
import MufuSelectorSm103

namespace GB300Math.MufuRootSelectorAccuracy

open MufuRootBounds MufuReciprocalBounds

def selected (modifier selector input : Nat) : Nat :=
  MufuSelectorSm103.modify 16 modifier (input / 2^(16*selector))

def result (bf16 reciprocal : Bool) (modifier selector input : Nat) : Nat :=
  (MufuSelectorSm103.evaluate (if bf16 then "bf16" else "f16")
    (if reciprocal then "rsq" else "sqrt") modifier selector input).getD 65536

theorem result_eq (bf16 reciprocal : Bool) (modifier selector input : Nat) :
    result bf16 reciprocal modifier selector input =
      MufuRootBounds.output bf16 reciprocal (selected modifier selector input) := by
  cases bf16 <;> cases reciprocal <;>
    simp [result, selected, MufuSelectorSm103.evaluate, MufuRootBounds.output]

theorem source_floatLib_sqrt (bf16 : Bool) (modifier : Fin 4) (selector : Fin 2)
    (input : BitVec 32)
    (inside : domain bf16 (selected modifier.val selector.val input.toNat)
      (result bf16 false modifier.val selector.val input.toNat) = true) :
    ∃ x y : ℝ,
      (floatLibValue bf16 (selected modifier.val selector.val input.toNat)).map (fun q : ℚ => (q : ℝ)) = some x ∧
      (floatLibValue bf16 (result bf16 false modifier.val selector.val input.toNat)).map (fun q : ℚ => (q : ℝ)) = some y ∧
      |y-Real.sqrt x| / |Real.sqrt x| ≤
        1-Real.sqrt (((MufuRootBounds.lower bf16 false).1 : ℝ)/(MufuRootBounds.lower bf16 false).2) := by
  rw [result_eq] at inside ⊢
  have h := inside
  simp only [domain, Bool.and_eq_true, decide_eq_true_eq] at h
  have input_small : selected modifier.val selector.val input.toNat < 65536 := by
    cases bf16 <;> simp [infinity] at h <;> omega
  exact MufuRootFloatLibUniversal.sqrt_floatLib_all bf16
    ⟨selected modifier.val selector.val input.toNat, input_small⟩ inside

theorem source_floatLib_rsq (bf16 : Bool) (modifier : Fin 4) (selector : Fin 2)
    (input : BitVec 32)
    (inside : domain bf16 (selected modifier.val selector.val input.toNat)
      (result bf16 true modifier.val selector.val input.toNat) = true) :
    ∃ x y : ℝ,
      (floatLibValue bf16 (selected modifier.val selector.val input.toNat)).map (fun q : ℚ => (q : ℝ)) = some x ∧
      (floatLibValue bf16 (result bf16 true modifier.val selector.val input.toNat)).map (fun q : ℚ => (q : ℝ)) = some y ∧
      |y-1/Real.sqrt x| / |1/Real.sqrt x| ≤
        1-Real.sqrt (((MufuRsqTightBounds.lower bf16).1 : ℝ)/(MufuRsqTightBounds.lower bf16).2) := by
  rw [result_eq] at inside ⊢
  have h := inside
  simp only [domain, Bool.and_eq_true, decide_eq_true_eq] at h
  have input_small : selected modifier.val selector.val input.toNat < 65536 := by
    cases bf16 <;> simp [infinity] at h <;> omega
  exact MufuRootFloatLibUniversal.rsq_floatLib_all bf16
    ⟨selected modifier.val selector.val input.toNat, input_small⟩ inside

#print axioms result_eq
#print axioms source_floatLib_sqrt
#print axioms source_floatLib_rsq

end GB300Math.MufuRootSelectorAccuracy
