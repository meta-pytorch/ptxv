import FabricBF16

namespace FabricScale
open FloatLib.Floats.Formats.BinaryInterchange
open FabricBF16Block

def inputs : List Nat := [43618,16281,16048,16090]
def doubled : List Nat := [43746,16409,16176,16218]
def value (word : Nat) : Rat :=
  ((Model.ofNatBits word : Model FloatFormat.bfloat16).toRat?).getD 0

def ideal (words : List Nat) : Rat := words.foldl (fun sum word => sum+value word) 0

theorem normal_inputs :
    (inputs ++ doubled).all (fun w => 0 < (w%32768)/128 ∧ (w%32768)/128 < 255) = true := by
  decide +kernel

theorem inputs_doubled :
    (inputs.zip doubled).all (fun pair => value pair.2 == 2*value pair.1) = true := by
  decide +kernel

theorem exact_sums : ideal inputs = 503/256-113/2^49 ∧
    ideal doubled = 503/128-113/2^48 := by decide +kernel

theorem observed_words : candidate inputs = some 16379 ∧ candidate doubled = some 16508 := by
  decide +kernel

theorem exact_rounding : exact inputs = 16379 ∧ exact doubled = 16507 := by decide +kernel

theorem output_values : value 16379 = 251/128 ∧ value 16508 = 63/16 := by decide +kernel

theorem scale_discrepancy : value 16508 - 2*value 16379 = 1/64 := by decide +kernel

theorem doubled_error_exceeds_half_ulp : 1/128 < value 16508 - ideal doubled := by
  decide +kernel

#print axioms normal_inputs
#print axioms inputs_doubled
#print axioms exact_sums
#print axioms observed_words
#print axioms exact_rounding
#print axioms output_values
#print axioms scale_discrepancy
#print axioms doubled_error_exceeds_half_ulp
end FabricScale
