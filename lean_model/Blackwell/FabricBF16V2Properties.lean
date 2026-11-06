import FabricBF16V2Bounds
import FabricBF16Extremes
import FabricBF16Layout

namespace FabricBF16V2Properties
open FabricBF16Block FabricBF16V2
open FloatLib.Floats.Formats.BinaryInterchange

theorem word_array_perm (xs ys : Array (Array Nat)) (hp : xs.toList.Perm ys.toList)
    (group lane : Nat) :
    (((Array.range 4).map fun k => xs.toList.map fun rank => rank[group*4+k]!)[lane]!).Perm
      (((Array.range 4).map fun k => ys.toList.map fun rank => rank[group*4+k]!)[lane]!) := by
  by_cases h : lane<4
  · simpa [h] using hp.map (fun rank => rank[group*4+lane]!)
  · simp [h]

theorem predict_perm (xs ys : Array (Array Nat)) (hp : xs.toList.Perm ys.toList) (sr : Bool) :
    FabricBF16V2.predict xs sr = FabricBF16V2.predict ys sr := by
  have hc (group lane : Nat) := FabricBF16V2Bounds.coarseShift_perm (word_array_perm xs ys hp group lane)
  unfold FabricBF16V2.predict
  dsimp only
  simp_rw [Array.map_map,Function.comp_def,FabricBF16Permutation.state_array_perm xs ys hp,hc]

def broadcast : Array (Array Nat) := #[Array.replicate 8 0x3fff,Array.replicate 8 0x3180]

theorem broadcast_outputs : FabricBF16V2.predict broadcast true = Array.replicate 8 0x4000 ∧
    FabricBF16V2.predict broadcast false = Array.replicate 8 0x3fff := by decide +kernel

theorem broadcast_values :
    (Model.ofNatBits 0x3fff : Model FloatFormat.bfloat16).toRat?=some (255/128:Rat) ∧
    (Model.ofNatBits 0x3180 : Model FloatFormat.bfloat16).toRat?=some (1/(2^28:Rat)) ∧
    (Model.ofNatBits 0x4000 : Model FloatFormat.bfloat16).toRat?=some (2:Rat) := by decide +kernel

theorem broadcast_error_ratio :
    ((2:Rat)-(255/128+1/2^28))/(1/2^28)=2097151 := by decide +kernel

theorem cancellation_ideal :
    (Model.sum FloatFormat.bfloat16
      #[(Model.ofNatBits 0x3f80 : Model FloatFormat.bfloat16),Model.ofNatBits 0xbf80,Model.ofNatBits 0x2b80,Model.ofNatBits 0]
      .nearestEven).toRat?=some (1/(2^40:Rat)) := by decide +kernel

theorem cancellation_relative_error : ((1:Rat)/2^40-0)/(1/2^40)=1 := by decide +kernel

theorem cancellation_normal_inputs : ∀ w∈[0x3f80,0xbf80,0x2b80],
    0<w%32768/128 ∧ w%32768/128<255 := by decide +kernel

theorem old_budget_too_small :
    (4:Rat)*(2^86-1)/2^133+1/2^48 < (1:Rat)/2^40 := by decide +kernel

theorem extremes_preserved :
    (FabricBF16V2.predict (FabricBF16Extremes.inputs false) true)[0]! =0x3f80 ∧
    (FabricBF16V2.predict (FabricBF16Extremes.inputs true) true)[0]! =0x3f81 := by decide +kernel

theorem layout_preserved :
    (FabricBF16V2.predict FabricBF16Layout.original true)[0]! =0x3f81 ∧
    (FabricBF16V2.predict FabricBF16Layout.swapped true)[0]! =0x3f80 := by decide +kernel

#print axioms word_array_perm
#print axioms predict_perm
#print axioms broadcast_outputs
#print axioms broadcast_values
#print axioms broadcast_error_ratio
#print axioms cancellation_ideal
#print axioms cancellation_relative_error
#print axioms cancellation_normal_inputs
#print axioms old_budget_too_small
#print axioms extremes_preserved
#print axioms layout_preserved
end FabricBF16V2Properties
