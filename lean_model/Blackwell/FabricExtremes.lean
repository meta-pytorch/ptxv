import FabricFP8

namespace FabricExtremes
open FloatLib.Floats.Formats.BinaryInterchange

def inputs (e4 high : Bool) : Array (Array Nat) :=
  #[(Array.replicate 8 (if high then 63 else 0)).set! 0 120,
    (Array.replicate 8 (if high ∧ !e4 then 51 else 0)).set! 0 1]
def result (e4 high : Bool) : Nat := ((FabricFP8.predict e4 (inputs e4 high))[0]!).getD 0

def value (e4 : Bool) (word : Nat) : Rat :=
  ((Model.ofNatBits word : Model (FabricFP8.scalarFormat e4)).toRat?).getD 0

def ideal (e4 : Bool) : Rat := value e4 120+value e4 1

theorem same_targets (e4 : Bool) :
    (inputs e4 true).map (·[0]!) = (inputs e4 false).map (·[0]!) := by
  cases e4 <;> decide +kernel

theorem predicted_outputs :
    result true false = 120 ∧ result true true = 121 ∧
    result false false = 120 ∧ result false true = 121 := by decide +kernel

theorem e4_values : ideal true = 256+1/512 ∧ value true 121 = 288 := by decide +kernel

theorem e5_values : ideal false = 32768+1/65536 ∧ value false 121 = 40960 := by decide +kernel

theorem e4_relative_error :
    (value true 121-ideal true)/ideal true = 16383/131073 ∧
    12499/100000 < (value true 121-ideal true)/ideal true := by decide +kernel

theorem e5_relative_error :
    (value false 121-ideal false)/ideal false = 536870911/2147483649 ∧
    24999/100000 < (value false 121-ideal false)/ideal false := by decide +kernel

#print axioms same_targets
#print axioms predicted_outputs
#print axioms e4_values
#print axioms e5_values
#print axioms e4_relative_error
#print axioms e5_relative_error
end FabricExtremes
