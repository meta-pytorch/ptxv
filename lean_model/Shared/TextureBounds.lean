import TextureBits

namespace GB300Math.TextureBits

theorem roundWeight_error (n : Nat) :
    256 * roundWeight n ≤ n + 128 ∧ n ≤ 256 * roundWeight n + 128 := by
  unfold roundWeight
  omega

theorem complement_weight_error (fraction total : Nat) (h : fraction ≤ 256) :
    256 * (total - roundWeight (fraction * total)) ≤ (256 - fraction) * total + 128 ∧
    (256 - fraction) * total ≤ 256 * (total - roundWeight (fraction * total)) + 128 := by
  have hr := roundWeight_error (fraction * total)
  have hb := roundWeight_le total fraction h
  have he : fraction * total + (256 - fraction) * total = 256 * total := by
    rw [← Nat.add_mul]
    congr 1
    omega
  have hs : 256 * roundWeight (fraction * total) +
      256 * (total - roundWeight (fraction * total)) = 256 * total := by
    rw [← Nat.mul_add]
    congr 1
    omega
  omega

def WithinCoefficientStep (actual numerator : Nat) : Prop :=
  65536 * actual ≤ numerator + 65536 ∧ numerator ≤ 65536 * actual + 65536

theorem two_stage_error (a u y v : Nat) (hy : y ≤ 256)
    (first : 256 * u ≤ a + 128 ∧ a ≤ 256 * u + 128)
    (second : 256 * v ≤ u * y + 128 ∧ u * y ≤ 256 * v + 128) :
    WithinCoefficientStep v (a * y) := by
  have firstHi := Nat.mul_le_mul_right y first.1
  have firstLo := Nat.mul_le_mul_right y first.2
  have secondHi := Nat.mul_le_mul_left 256 second.1
  have secondLo := Nat.mul_le_mul_left 256 second.2
  unfold WithinCoefficientStep
  simp only [Nat.add_mul, Nat.mul_add, Nat.mul_assoc] at firstHi firstLo secondHi secondLo ⊢
  omega

def idealPlaneNumerators (x y total : Nat) : Array Nat :=
  #[(256 - x) * total * (256 - y), x * total * (256 - y),
    (256 - x) * total * y, x * total * y]

theorem plane_error (x y total i : Nat) (hx : x ≤ 256) (hy : y ≤ 256) (hi : i < 4) :
    WithinCoefficientStep ((plane x y total)[i]!) ((idealPlaneNumerators x y total)[i]!) := by
  let highX := roundWeight (x * total)
  let lowX := total - highX
  have firstHigh := roundWeight_error (x * total)
  have firstLow := complement_weight_error x total hx
  have secondHigh := roundWeight_error (highX * y)
  have secondLow := roundWeight_error (lowX * (256 - y))
  have secondHighComplement := complement_weight_error y highX hy
  have secondLowComplement := complement_weight_error (256 - y) lowX (by omega)
  have hyc : 256 - y ≤ 256 := by omega
  have hyy : 256 - (256 - y) = y := by omega
  rw [hyy] at secondLowComplement
  rw [Nat.mul_comm y highX, Nat.mul_comm (256 - y) highX] at secondHighComplement
  rw [Nat.mul_comm (256 - y) lowX, Nat.mul_comm y lowX] at secondLowComplement
  have cases : i = 0 ∨ i = 1 ∨ i = 2 ∨ i = 3 := by omega
  rcases cases with rfl | rfl | rfl | rfl
  · exact two_stage_error _ lowX (256 - y) _ hyc firstLow secondLow
  · exact two_stage_error _ highX (256 - y) _ hyc firstHigh secondHighComplement
  · exact two_stage_error _ lowX y _ hy firstLow secondLowComplement
  · exact two_stage_error _ highX y _ hy firstHigh secondHigh

#print axioms roundWeight_error
#print axioms complement_weight_error
#print axioms plane_error

end GB300Math.TextureBits
