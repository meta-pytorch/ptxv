import FabricBF16StochasticBounds

set_option maxRecDepth 100000
set_option maxHeartbeats 0

namespace FabricBF16Extremes
open FloatLib.Floats.Formats.BinaryInterchange
open FabricBF16Block

def inputs (high : Bool) : Array (Array Nat) :=
  #[#[0x3f80, if high then 0x3fff else 0, if high then 0x3fff else 0, if high then 0x3fff else 0,0,0,0,0],
    #[0x2800, if high then 0x3bfe else 0, if high then 0x3bfe else 0, if high then 0x3bfe else 0,0,0,0,0]]

def value (word : Nat) : Rat :=
  ((Model.ofNatBits word : Model FloatFormat.bfloat16).toRat?).getD 0

def ideal : Rat := value 0x3f80+value 0x2800

theorem fixed_target : ((inputs false)[0]!)[0]! = ((inputs true)[0]!)[0]! ∧
    ((inputs false)[1]!)[0]! = ((inputs true)[1]!)[0]! := by decide +kernel

theorem nonzero_inputs_normal : (inputs true).all (fun words => words.all
    (fun word => word=0 ∨ (0<word/128 ∧ word/128<255))) = true := by decide +kernel

theorem exact_ideal : ideal=1+1/2^47 := by decide +kernel

theorem low_entropy : entropy ((Array.range 4).map fun lane =>
    accumulate ((inputs false).toList.map fun rank => rank[lane]!)) 0 = 0 := by decide +kernel

theorem maximal_entropy : entropy ((Array.range 4).map fun lane =>
    accumulate ((inputs true).toList.map fun rank => rank[lane]!)) 0 = 2^42-1 := by decide +kernel

theorem low_output : (predict (inputs false) true)[0]! = 0x3f80 := by decide +kernel

theorem high_output : (predict (inputs true) true)[0]! = 0x3f81 := by decide +kernel

theorem rne_output : (predict (inputs true) false)[0]! = 0x3f80 := by decide +kernel

theorem low_error : ideal-value 0x3f80 = 1/2^47 := by decide +kernel

theorem high_error : value 0x3f81-ideal = 1/2^7-1/2^47 := by decide +kernel

theorem relative_error : (value 0x3f81-ideal)/ideal=(2^40-1)/(2^47+1) := by decide +kernel

theorem nearly_full_spacing : 1/128-1/2^46 < (value 0x3f81-ideal)/ideal := by decide +kernel

theorem error_ratio : (value 0x3f81-ideal)/(ideal-value 0x3f80)=1099511627775 := by decide +kernel

theorem exceeds_half_spacing : 1/256 < value 0x3f81-ideal := by decide +kernel

#print axioms fixed_target
#print axioms nonzero_inputs_normal
#print axioms exact_ideal
#print axioms low_entropy
#print axioms maximal_entropy
#print axioms low_output
#print axioms high_output
#print axioms rne_output
#print axioms low_error
#print axioms high_error
#print axioms relative_error
#print axioms nearly_full_spacing
#print axioms error_ratio
#print axioms exceeds_half_spacing
end FabricBF16Extremes
