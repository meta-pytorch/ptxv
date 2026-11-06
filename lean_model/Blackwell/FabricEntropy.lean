import FabricFP8

namespace FabricEntropy

theorem refine_threshold (entropy threshold radix tail : Nat) (h : tail < radix) :
    threshold ≤ entropy ↔ threshold * radix ≤ entropy * radix + tail := by
  constructor
  · intro he
    have hm := Nat.mul_le_mul_right radix he
    omega
  · intro he
    by_cases ht : threshold ≤ entropy
    · exact ht
    · have hnext : entropy + 1 ≤ threshold := by omega
      have hm := Nat.mul_le_mul_right radix hnext
      simp only [Nat.add_mul, Nat.one_mul] at hm
      omega

theorem decision_threshold (remainder denominator multiple entropy : Nat)
    (hd : 0 < denominator) (hr : remainder ≤ denominator) :
    denominator * (denominator * multiple) ≤
        remainder * (denominator * multiple) + entropy * denominator ↔
      (denominator-remainder) * multiple ≤ entropy := by
  have hsum : remainder * multiple + (denominator-remainder) * multiple =
      denominator * multiple := by
    rw [← Nat.add_mul]
    congr 1
    omega
  have hfactor : remainder * (denominator * multiple) + entropy * denominator =
      denominator * (remainder * multiple + entropy) := by
    simp [Nat.mul_add, Nat.mul_comm, Nat.mul_left_comm]
  rw [hfactor, Nat.mul_le_mul_left_iff hd]
  omega

theorem entropy_refinement (r d k e radix tail : Nat)
    (hd : 0 < d) (hr : r ≤ d) (ht : tail < radix) :
    d*(d*k) ≤ r*(d*k)+e*d ↔
      d*(d*(k*radix)) ≤ r*(d*(k*radix))+(e*radix+tail)*d := by
  rw [decision_threshold r d k e hd hr,
      decision_threshold r d (k*radix) (e*radix+tail) hd hr]
  rw [← Nat.mul_assoc]
  exact refine_threshold e ((d-r)*k) radix tail ht

def validGap (e4 : Bool) (code : Nat) : Prop :=
  0 < FabricFP8.unit e4 (code+1)-FabricFP8.unit e4 code ∧
  (FabricFP8.unit e4 (code+1)-FabricFP8.unit e4 code) ∣
    2^(7*FabricFP8.width e4)

theorem e4_gap_divides (code : Fin 127) : validGap true code := by
  have allCodes : ∀ code : Fin 127, validGap true code := by
    unfold validGap
    decide +kernel
  exact allCodes code

theorem e5_gap_divides (code : Fin 124) : validGap false code := by
  have allCodes : ∀ code : Fin 124, validGap false code := by
    unfold validGap
    decide +kernel
  exact allCodes code

theorem nondividing_scale_can_change :
    ¬ (3*2 ≤ 1*2+1*3 ↔ 3*(2*2) ≤ 1*(2*2)+(1*2+1)*3) := by decide +kernel

theorem tail_must_fit : ¬ (1 ≤ 0 ↔ 1*2 ≤ 0*2+2) := by decide +kernel

#print axioms e4_gap_divides
#print axioms e5_gap_divides
#print axioms nondividing_scale_can_change
#print axioms tail_must_fit
#print axioms refine_threshold
#print axioms decision_threshold
#print axioms entropy_refinement
end FabricEntropy
