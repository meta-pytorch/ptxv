import Std
import GB300Math.TensorLegacyBlackwell

namespace GB300Math.TensorPermutationCases
open GB300Math.Tensor

def ratio (x : GB300Math.Tensor.Dyadic) : Rat :=
  if x.exponent ≥ 0 then (x.significand : Rat) * 2^x.exponent.toNat
  else (x.significand : Rat) / 2^(-x.exponent).toNat

def exactDot (a b : Array Nat) (c : Nat) : Rat :=
  (List.range a.size).foldl (fun total i =>
    total + ratio (decode a[i]! 10 5) * ratio (decode b[i]! 10 5))
    (ratio (decode c 23 8))

def a : Array Nat := #[0x4b25, 0xcb25, 0x4fe6, 0xcfe6, 0x53ca, 0xd3ca, 0x4a5b, 0xca5b, 0x4da5, 0xcda5, 0x521a, 0xd21a, 0x48f6, 0xc8f6, 0x1, 0x0]

def b : Array Nat := #[0x4a7b, 0x4a7b, 0x4e9e, 0x4e9e, 0x5398, 0x5398, 0x4952, 0x4952, 0x4fe8, 0x4fe8, 0x5249, 0x5249, 0x4a91, 0x4a91, 0x1, 0x3c00]

def separatedA : Array Nat := #[0x4b25, 0x4fe6, 0x53ca, 0x4a5b, 0x4da5, 0x521a, 0x48f6, 0x1, 0xcb25, 0xcfe6, 0xd3ca, 0xca5b, 0xcda5, 0xd21a, 0xc8f6, 0x0]

def separatedB : Array Nat := #[0x4a7b, 0x4e9e, 0x5398, 0x4952, 0x4fe8, 0x5249, 0x4a91, 0x1, 0x4a7b, 0x4e9e, 0x5398, 0x4952, 0x4fe8, 0x5249, 0x4a91, 0x3c00]

def c : Nat := 0xa7000000

set_option maxRecDepth 100000
set_option maxHeartbeats 3000000

theorem all_inputs_finite :
    (a ++ b ++ separatedA ++ separatedB).all (fun x => !isNaN x 10 5 && !isInf x 10 5) = true ∧
    isNaN c 23 8 = false ∧ isInf c 23 8 = false := by decide +kernel

theorem separated_is_permutation :
    separatedA = ((List.range 8).toArray.map fun i => a[2*i]!) ++
      ((List.range 8).toArray.map fun i => a[2*i+1]!) ∧
    separatedB = ((List.range 8).toArray.map fun i => b[2*i]!) ++
      ((List.range 8).toArray.map fun i => b[2*i+1]!) := by decide +kernel

theorem original_ideal : exactDot a b c = (1 : Rat)/2^49 := by decide +kernel

theorem separated_ideal : exactDot separatedA separatedB c = (1 : Rat)/2^49 := by decide +kernel

theorem blackwell_original_prediction :
    TensorLegacyBlackwell.dot "f16" "f32" a b c = 0 := by decide +kernel

theorem blackwell_separated_prediction :
    TensorLegacyBlackwell.dot "f16" "f32" separatedA separatedB c = 0 := by decide +kernel

#print axioms all_inputs_finite
#print axioms separated_is_permutation
#print axioms original_ideal
#print axioms separated_ideal
#print axioms blackwell_original_prediction
#print axioms blackwell_separated_prediction
end GB300Math.TensorPermutationCases
