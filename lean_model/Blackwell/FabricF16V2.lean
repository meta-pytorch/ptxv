/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FloatLib.Floats.Formats.BinaryInterchange.Configured

set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace FabricF16V2
open FloatLib.Floats.Formats.BinaryInterchange

def units (word : Nat) : Option Int := do
  unless word < 65536 do none
  let d ← (Model.ofNatBits word : Model FloatFormat.binary16).toDyadic?
  unless d.exponent ≥ -24 do none
  let magnitude := d.significand * 2^(d.exponent+24).toNat
  return if d.negative then -(magnitude : Int) else magnitude

def maxExponent (words : List Nat) : Nat :=
  words.foldr (fun word e => max ((word % 32768)/1024) e) 0

def coarseShift (words : List Nat) : Nat :=
  if maxExponent words < 16 then 0 else 2

def flush (shift : Nat) (value : Int) : Int :=
  if value.natAbs < 2^shift then 0 else value

def round (value : Int) : Nat :=
  let d : FloatLib.Numerics.Dyadic :=
    {negative := value < 0, significand := value.natAbs, exponent := -24}
  (Model.roundDyadic FloatFormat.binary16 d).toNatBits

def exact (words : List Nat) : Nat :=
  (Model.sum FloatFormat.binary16
    (words.toArray.map fun w => (Model.ofNatBits w : Model FloatFormat.binary16)) .nearestEven).toNatBits

inductive Accum where
  | finite (value : Int)
  | nan (negative : Bool) (payload : Nat)
  | infinity (negative : Bool)
  deriving Inhabited

def accumulate (words : List Nat) : Accum := Id.run do
  let nans := words.filter fun w => w % 32768 > 31744
  if !nans.isEmpty then
    let negative := nans.any fun w => w ≥ 32768
    let selected := if negative then nans.filter (fun w => w ≥ 32768) else nans
    return .nan negative (selected.foldl (fun p w => max p (w % 512)) 0)
  let infinities := words.filter fun w => w % 32768 = 31744
  if !infinities.isEmpty then
    let negative := infinities.any fun w => w ≥ 32768
    let positive := infinities.any fun w => w < 32768
    if negative ∧ positive then return .nan false 0
    return .infinity negative
  return .finite ((words.map fun w => (units w).getD 0).sum)

def fraction : Accum → Nat
  | .infinity _ => 0
  | .nan _ payload => payload * 32
  | .finite value =>
    let magnitude := value.natAbs
    let exponent := max 10 magnitude.log2
    (if exponent ≤ 14 then magnitude * 2^(14-exponent)
     else magnitude / 2^(exponent-14)) % 16384

def unit (code : Nat) : Nat :=
  (code % 1024 + if code / 1024 = 0 then 0 else 1024) * 2^(code/1024-1)

def entropy (states : Array Accum) (lane : Nat) : Nat := Id.run do
  let mut result := 0
  for plane in [:14] do
    for offset in [1:4] do
      result := 2*result + fraction states[(lane+offset)%4]! / 2^(13-plane) % 2
  return result

def output (states : Array Accum) (lane : Nat) (sr : Bool) : Nat := Id.run do
  match states[lane]! with
  | .nan negative payload => return (if negative then 32768 else 0)+32256+payload
  | .infinity negative => return (if negative then 32768 else 0)+31744
  | .finite value =>
    if !sr then return round value
    let magnitude := value.natAbs
    let sign := if value < 0 then 32768 else 0
    if magnitude ≥ unit 31744 then return sign+31744
    let shift := magnitude.log2-10
    let code := 1024*shift + magnitude/2^shift
    if magnitude = unit code then return sign+code
    let gap := unit (code+1)-unit code
    let remainder := magnitude-unit code
    return sign+code+(if remainder*2^42+entropy states lane*gap ≥ gap*2^42 then 1 else 0)

def finalOutput (states : Array Accum) (lane shift : Nat) (sr : Bool) : Nat :=
  match states[lane]! with
  | .finite value => if value.natAbs < 2^shift then 0 else output states lane sr
  | _ => output states lane sr

def predict (inputs : Array (Array Nat)) (sr : Bool) : Array Nat := Id.run do
  let mut result := #[]
  for group in [:2] do
    let words := (Array.range 4).map fun lane =>
      inputs.toList.map fun rank => rank[group*4+lane]!
    let states := words.map accumulate
    for lane in [:4] do result := result.push (finalOutput states lane (coarseShift words[lane]!) sr)
  return result

def cancellation : Array (Array Nat) :=
  #[Array.replicate 8 0x7bff,Array.replicate 8 0xfbff,Array.replicate 8 1,Array.replicate 8 0]

theorem cancellation_flush : (predict cancellation false)[0]! =(0:Nat) ∧
    (predict cancellation true)[0]! =(0:Nat) := by decide +kernel

def neighborLow : Array (Array Nat) :=
  #[#[0x3c00,0x3c00,0x3c00,0x3c00,0,0,0,0],#[0x1000,0,0,0,0,0,0,0]]
def neighborHigh : Array (Array Nat) :=
  #[#[0x3c00,0x3fff,0x3fff,0x3fff,0,0,0,0],#[0x1000,0,0,0,0,0,0,0]]

theorem neighbor_sensitive : (predict neighborLow true)[0]! =0x3c00 ∧
    (predict neighborHigh true)[0]! =0x3c01 := by decide +kernel

theorem neighbor_rne : (predict neighborLow false)[0]! =0x3c00 ∧
    (predict neighborHigh false)[0]! =0x3c00 := by decide +kernel

#print axioms cancellation_flush
#print axioms neighbor_sensitive
#print axioms neighbor_rne
end FabricF16V2
