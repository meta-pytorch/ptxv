import FloatLib.Floats.Formats.BinaryInterchange.Configured
import GB300Math.TensorAlignmentLoss

set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace FabricBF16Block
open FloatLib.Floats.Formats.BinaryInterchange
open GB300Math.TensorAlignmentLoss

def units (word : Nat) : Option Int := do
  unless word < 65536 do none
  let d ← (Model.ofNatBits word : Model FloatFormat.bfloat16).toDyadic?
  unless d.exponent ≥ -133 do none
  let magnitude := d.significand * 2^(d.exponent+133).toNat
  return if d.negative then -(magnitude : Int) else magnitude

def maxExponent (words : List Nat) : Nat :=
  words.foldr (fun word e => max ((word % 32768)/128) e) 0

def latticeShift (words : List Nat) : Nat := 16*(maxExponent words/16)-26

def aligned (words : List Nat) (values : List Int) : Int :=
  truncatedSum values (2^(latticeShift words))

def round (value : Int) : Nat :=
  let d : FloatLib.Numerics.Dyadic :=
    {negative := value < 0, significand := value.natAbs, exponent := -133}
  (Model.roundDyadic FloatFormat.bfloat16 d).toNatBits

def exact (words : List Nat) : Nat :=
  (Model.sum FloatFormat.bfloat16
    (words.toArray.map fun w => (Model.ofNatBits w : Model FloatFormat.bfloat16)) .nearestEven).toNatBits

inductive Accum where
  | finite (value : Int)
  | nan (negative : Bool) (payload : Nat)
  | infinity (negative : Bool)
  deriving Inhabited

def accumulate (words : List Nat) : Accum := Id.run do
  let nans := words.filter fun w => w % 32768 > 32640
  if !nans.isEmpty then
    let negative := nans.any fun w => w ≥ 32768
    let selected := if negative then nans.filter (fun w => w ≥ 32768) else nans
    return .nan negative (selected.foldl (fun p w => max p (w % 64)) 0)
  let infinities := words.filter fun w => w % 32768 = 32640
  if !infinities.isEmpty then
    let negative := infinities.any fun w => w ≥ 32768
    let positive := infinities.any fun w => w < 32768
    if negative ∧ positive then return .nan false 0
    return .infinity negative
  return .finite (aligned words (words.map fun w => (units w).getD 0))

def fraction : Accum → Nat
  | .infinity _ => 0
  | .nan _ payload => payload * 256
  | .finite value =>
    let magnitude := value.natAbs
    let exponent := max 7 magnitude.log2
    (if exponent ≤ 14 then magnitude * 2^(14-exponent)
     else magnitude / 2^(exponent-14)) % 16384

def unit (code : Nat) : Nat :=
  (code % 128 + if code / 128 = 0 then 0 else 128) * 2^(code/128-1)

def entropy (states : Array Accum) (lane : Nat) : Nat := Id.run do
  let mut result := 0
  for plane in [:14] do
    for offset in [1:4] do
      result := 2*result + fraction states[(lane+offset)%4]! / 2^(13-plane) % 2
  return result

def output (states : Array Accum) (lane : Nat) (sr : Bool) : Nat := Id.run do
  match states[lane]! with
  | .nan negative payload => return (if negative then 32768 else 0)+32704+payload
  | .infinity negative => return (if negative then 32768 else 0)+32640
  | .finite value =>
    if !sr then return round value
    let magnitude := value.natAbs
    let sign := if value < 0 then 32768 else 0
    if magnitude ≥ unit 32640 then return sign+32640
    let shift := magnitude.log2-7
    let code := 128*shift + magnitude/2^shift
    if magnitude = unit code then return sign+code
    let gap := unit (code+1)-unit code
    let remainder := magnitude-unit code
    return sign+code+(if remainder*2^42+entropy states lane*gap ≥ gap*2^42 then 1 else 0)

def candidate (words : List Nat) : Option Nat :=
  some (output #[accumulate words] 0 false)

def predict (inputs : Array (Array Nat)) (sr : Bool) : Array Nat := Id.run do
  let mut result := #[]
  for group in [:2] do
    let states := (Array.range 4).map fun lane =>
      accumulate (inputs.toList.map fun rank => rank[group*4+lane]!)
    for lane in [:4] do result := result.push (output states lane sr)
  return result

theorem accumulation_bound (words : List Nat) (values : List Int) :
    -((values.length*(2^(latticeShift words)-1) : Nat) : Int) ≤
      exactSum values-aligned words values ∧
    exactSum values-aligned words values ≤
      ((values.length*(2^(latticeShift words)-1) : Nat) : Int) := by
  exact sum_truncation_loss values (2^(latticeShift words)) (Nat.two_pow_pos _)

def witness1 : List Nat := [23961,32924,56729,34641]
def witness2 : List Nat := [29938,15204,12942,62706]
def witness3 : List Nat := [52108,61188,37250,28420]

theorem first_cancellation : candidate witness1 = some 0 ∧ exact witness1 = 34641 := by decide +kernel
theorem second_cancellation : candidate witness2 = some 0 ∧ exact witness2 = 15204 := by decide +kernel
theorem third_cancellation : candidate witness3 = some 0 ∧ exact witness3 = 52108 := by decide +kernel

#print axioms accumulation_bound
#print axioms first_cancellation
#print axioms second_cancellation
#print axioms third_cancellation
end FabricBF16Block
