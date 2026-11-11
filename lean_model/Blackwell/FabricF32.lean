import FloatLib.Floats.Formats.BinaryInterchange.Configured
import GB300Math.TensorAlignmentLoss

set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace FabricF32
open FloatLib.Floats.Formats.BinaryInterchange
open GB300Math.TensorAlignmentLoss

def units (word : Nat) : Option Int := do
  unless word < 2^32 do none
  let d ← (Model.ofNatBits word : Model (.ieee 8 23)).toDyadic?
  unless d.exponent ≥ -149 do none
  let magnitude := d.significand * 2^(d.exponent+149).toNat
  return if d.negative then -(magnitude : Int) else magnitude

def maxExponent (words : List Nat) : Nat :=
  words.foldr (fun word e => max ((word % 2^31)/2^23) e) 0

def fineShift (words : List Nat) : Nat := 16*(maxExponent words/16)-58
def coarseShift (words : List Nat) : Nat := 16*(maxExponent words/16)-49

def aligned (words : List Nat) (values : List Int) : Int :=
  truncatedSum values (2^fineShift words)

def flush (shift : Nat) (value : Int) : Int :=
  if value.natAbs < 2^shift then 0 else value

def round (value : Int) : Nat :=
  let d : FloatLib.Numerics.Dyadic :=
    {negative := value < 0, significand := value.natAbs, exponent := -149}
  (Model.roundDyadic (.ieee 8 23) d).toNatBits

def predict (words : List Nat) : Nat := Id.run do
  let nans := words.filter fun w => w % 2^31 > 0x7f800000
  if !nans.isEmpty then
    return nans.foldl (fun p w => max p (w ||| 0x400000)) 0
  let infinities := words.filter fun w => w % 2^31 = 0x7f800000
  if !infinities.isEmpty then
    let negative := infinities.any fun w => w ≥ 2^31
    let positive := infinities.any fun w => w < 2^31
    if negative ∧ positive then return 0x7fc00000
    return if negative then 0xff800000 else 0x7f800000
  let value := aligned words (words.map fun w => (units w).getD 0)
  return round (flush (coarseShift words) value)

def cancellation : List Nat := [805306368,2952790016,201326592,0]
def midpoint : List Nat := [805306368,603979776,201326592,0]

theorem cancellation_bits : predict cancellation = 0 := by decide +kernel
theorem midpoint_bits : predict midpoint = 805306369 := by decide +kernel

theorem cancellation_exact :
    (Model.sum (.ieee 8 23) (cancellation.toArray.map fun w => (Model.ofNatBits w : Model (.ieee 8 23))) .nearestEven).toNatBits = 201326592 := by decide +kernel

theorem tiny_value :
    (Model.ofNatBits 201326592 : Model (.ieee 8 23)).toRat? = some (1/(2^103 : Rat)) := by decide +kernel

theorem cancellation_relative_error :
    ((1/(2^103 : Rat))-0)/(1/(2^103 : Rat)) = 1 := by decide +kernel

theorem flush_loss (shift : Nat) (value : Int) :
    (flush shift value-value).natAbs ≤ 2^shift := by
  unfold flush
  split
  next h => simpa using Nat.le_of_lt h
  next h => simp

#print axioms cancellation_bits
#print axioms midpoint_bits
#print axioms cancellation_exact
#print axioms tiny_value
#print axioms cancellation_relative_error
#print axioms flush_loss
end FabricF32
