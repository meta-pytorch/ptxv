import GB300Math.Fp64Seed

set_option maxRecDepth 100000

namespace GB300Math.TanhSmall

def firstBinade (word : Nat) : Option Nat :=
  let magnitude := word % 2 ^ 31
  if 0x3b800000 ≤ magnitude ∧ magnitude < 0x3c000000 then
    let k := (magnitude - 0x3b800000) / 64
    let correction := (Fp64Seed.squareTruncated k 19 + 2 ^ 26 - 1) / 2 ^ 26
    some ((word / 2 ^ 31 % 2) * 2 ^ 31 + 0x3b800000 + 64 * k - correction)
  else none

theorem lower_endpoint : firstBinade 0x3b800000 = some 0x3b800000 := by decide +kernel
theorem next_quantized_input : firstBinade 0x3b800040 = some 0x3b800040 := by decide +kernel
theorem first_input_plateau : ∀ k : Fin 64, firstBinade (0x3b800000 + k) = some 0x3b800000 := by
  decide +kernel
theorem upper_endpoint : firstBinade 0x3bffffff = some 0x3bfffec0 := by decide +kernel

theorem quantization (k low : Nat) (hk : k < 131072) (hl : low < 64) :
    firstBinade (0x3b800000 + 64 * k + low) = firstBinade (0x3b800000 + 64 * k) := by
  have hm1 : (0x3b800000 + 64 * k + low) % 2 ^ 31 = 0x3b800000 + 64 * k + low := by omega
  have hm2 : (0x3b800000 + 64 * k) % 2 ^ 31 = 0x3b800000 + 64 * k := by omega
  have hs1 : (0x3b800000 + 64 * k + low) / 2 ^ 31 = 0 := by omega
  have hs2 : (0x3b800000 + 64 * k) / 2 ^ 31 = 0 := by omega
  have hc1 : 0x3b800000 ≤ 0x3b800000 + 64 * k + low ∧ 0x3b800000 + 64 * k + low < 0x3c000000 := by omega
  have hc2 : 0x3b800000 ≤ 0x3b800000 + 64 * k ∧ 0x3b800000 + 64 * k < 0x3c000000 := by omega
  have hq1 : (0x3b800000 + 64 * k + low - 0x3b800000) / 64 = k := by omega
  have hq2 : (0x3b800000 + 64 * k - 0x3b800000) / 64 = k := by simp
  simp only [firstBinade, hm1, hm2, hs1, hs2, hc1, hc2, hq1, hq2]

#print axioms quantization

#print axioms lower_endpoint
#print axioms first_input_plateau
#print axioms upper_endpoint

end GB300Math.TanhSmall
