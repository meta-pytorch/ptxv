import TensorPermutationCases

namespace GB300Math.TensorPermutationNormalCases
open GB300Math.Tensor GB300Math.TensorPermutationCases

def normalA := a.set! 14 0x400
def normalB := b.set! 14 0x400
def normalC : Nat := 0xb1000000
def separated (x : Array Nat) :=
  ((List.range 8).toArray.map fun i => x[2*i]!) ++
    ((List.range 8).toArray.map fun i => x[2*i+1]!)

set_option maxRecDepth 100000
set_option maxHeartbeats 3000000

theorem all_nonzero_inputs_normal :
    (normalA ++ normalB).all (fun x => x % 32768 == 0 ||
      (x / 1024 % 32 > 0 && x / 1024 % 32 < 31)) = true ∧
    normalC / 2^23 % 256 > 0 ∧ normalC / 2^23 % 256 < 255 := by decide +kernel

theorem original_ideal :
    exactDot normalA normalB normalC = (1 : Rat)/2^29 := by decide +kernel

theorem separated_ideal :
    exactDot (separated normalA) (separated normalB) normalC =
      (1 : Rat)/2^29 := by decide +kernel

theorem blackwell_original_prediction :
    TensorLegacyBlackwell.dot "f16" "f32" normalA normalB normalC = 0 := by decide +kernel

theorem blackwell_separated_prediction :
    TensorLegacyBlackwell.dot "f16" "f32" (separated normalA) (separated normalB) normalC =
      0 := by decide +kernel

#print axioms all_nonzero_inputs_normal
#print axioms original_ideal
#print axioms separated_ideal
#print axioms blackwell_original_prediction
#print axioms blackwell_separated_prediction
end GB300Math.TensorPermutationNormalCases
