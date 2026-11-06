import FabricF16V2Bounds

namespace FabricF16V2Witnesses
open FabricF16V2 FloatLib.Floats.Formats.BinaryInterchange

def highNeighbors : Array (Array Nat) :=
  #[#[0x7800,0x3fff,0x3fff,0x3fff,0,0,0,0],
    #[1,0x1380,0x1380,0x1380,0,0,0,0]]

def lowNeighbors : Array (Array Nat) :=
  #[#[0x7800,0x3c00,0x3c00,0x3c00,0,0,0,0],
    #[1,0,0,0,0,0,0,0]]

def threeUnits : Array (Array Nat) :=
  #[Array.replicate 8 0x4000,Array.replicate 8 0xc000,
    Array.replicate 8 3,Array.replicate 8 0]

theorem high_rounds_up : (predict highNeighbors true)[0]! =0x7801 := by decide +kernel

theorem low_rounds_down : (predict lowNeighbors true)[0]! =0x7800 := by decide +kernel

theorem nearest_rounds_down : (predict highNeighbors false)[0]! =0x7800 := by decide +kernel

theorem high_decoded : (Model.ofNatBits (predict highNeighbors true)[0]! : Model FloatFormat.binary16).toRat?=some 32800 := by
  rw [high_rounds_up]
  decide +kernel

theorem nearest_decoded : (Model.ofNatBits (predict highNeighbors false)[0]! : Model FloatFormat.binary16).toRat?=some 32768 := by
  rw [nearest_rounds_down]
  decide +kernel

theorem exact_target :
    (Model.ofNatBits 0x7800 : Model FloatFormat.binary16).toRat?=some 32768 ∧
    (Model.ofNatBits 1 : Model FloatFormat.binary16).toRat?=some (1/(2^24:Rat)) := by decide +kernel

theorem amplification :
    |(32800:Rat)-(32768+1/2^24)|=536870911*|(32768:Rat)-(32768+1/2^24)| := by norm_num

theorem flush_three : (predict threeUnits false)[0]! =0 ∧
    (predict threeUnits true)[0]! =0 := by decide +kernel

theorem lost_three_units :
    (Model.ofNatBits 3:Model FloatFormat.binary16).toRat?=some (3/(2^24:Rat)) := by decide +kernel

#print axioms high_rounds_up
#print axioms low_rounds_down
#print axioms nearest_rounds_down
#print axioms high_decoded
#print axioms nearest_decoded
#print axioms exact_target
#print axioms amplification
#print axioms flush_three
#print axioms lost_three_units
end FabricF16V2Witnesses
