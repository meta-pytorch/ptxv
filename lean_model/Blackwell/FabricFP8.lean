/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FloatLib.Floats.Formats.BinaryInterchange.Configured

set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace FabricFP8
open FloatLib.Floats.Formats.BinaryInterchange

def fractionBits (e4 : Bool) : Nat := if e4 then 3 else 2
def maximum (e4 : Bool) : Nat := if e4 then 126 else 123
def width (e4 : Bool) : Nat := if e4 then 3 else 5
def unit (e4 : Bool) (code : Nat) : Nat :=
  let f := fractionBits e4
  (code % 2^f + if code / 2^f = 0 then 0 else 2^f) * 2^(code / 2^f - 1)
def decode (e4 : Bool) (word : Nat) : Option Int :=
  if word ≥ 256 ∨ word % 128 > maximum e4 then none
  else some ((if word ≥ 128 then -1 else 1) * (unit e4 (word % 128) : Int))
def scalarFormat (e4 : Bool) : FloatFormat :=
  if e4 then FloatFormat.e4m3fn else FloatFormat.ieee 5 2

def decodeAgrees (e4 : Bool) (word : Nat) : Bool :=
  let actual := (Model.ofNatBits word : Model (scalarFormat e4)).toRat?
  let expected := (decode e4 word).map fun x => (x : Rat) / (if e4 then 512 else 65536)
  actual == expected

theorem e4_decode_agreement : (List.range 256).all (decodeAgrees true) = true := by decide +kernel
theorem e5_decode_agreement : (List.range 256).all (decodeAgrees false) = true := by decide +kernel

def mantissa (e4 : Bool) (value : Int) : Nat :=
  let magnitude := value.natAbs
  let exponent := max (fractionBits e4) magnitude.log2
  let bits := width e4
  (if bits ≥ exponent then magnitude * 2^(bits-exponent)
   else magnitude / 2^(exponent-bits)) % 2^bits

inductive Accum where
  | finite (value : Int)
  | nan (negative : Bool) (payload : Nat)
  | infinity (negative : Bool)
  deriving Inhabited

def sumWords (e4 : Bool) (words : List Nat) : Accum := Id.run do
  let nans := words.filter fun w => w % 128 > (if e4 then 126 else 124)
  if !nans.isEmpty then
    let negative := nans.any fun w => w ≥ 128
    let selected := if negative then nans.filter (fun w => w ≥ 128) else nans
    let payload := selected.foldl (fun p w => p ||| (w % 2^(fractionBits e4-1))) 0
    return .nan negative payload
  let infinities := words.filter fun w => w % 128 = 124 ∧ !e4
  if !infinities.isEmpty then
    let negative := infinities.any fun w => w ≥ 128
    let positive := infinities.any fun w => w < 128
    if negative ∧ positive then return .nan false 0
    return .infinity negative
  return .finite (words.foldl (fun total w => total + (decode e4 w).getD 0) 0)

def neighbor (e4 : Bool) : Accum → Nat
  | .finite value => mantissa e4 value
  | .infinity _ => 0
  | .nan _ payload => payload * 2^(1+width e4-fractionBits e4)

def entropy (e4 : Bool) (totals : Array Accum) (lane : Nat) : Nat := Id.run do
  let mut result := 0
  for plane in [:width e4] do
    for offset in [1:8] do
      let value := neighbor e4 totals[(lane+offset)%8]!
      result := 2*result + value / 2^(width e4-1-plane) % 2
  return result

def choose (lower numerator denominator noise scale : Nat) : Nat :=
  lower + if numerator*scale+noise*denominator ≥ denominator*scale then 1 else 0

theorem choose_adjacent (a n d e s : Nat) : choose a n d e s = a ∨ choose a n d e s = a+1 := by
  unfold choose
  split <;> simp_all

def output (e4 : Bool) (totals : Array Accum) (lane : Nat) : Nat := Id.run do
  match totals[lane]! with
  | .nan negative payload => return (if negative then 128 else 0)+(if e4 then 127 else 126+payload)
  | .infinity negative => return (if negative then 128 else 0)+124
  | .finite total =>
    let magnitude := total.natAbs
    let sign := if total < 0 then 128 else 0
    if magnitude ≥ unit e4 (maximum e4+1) then return sign+maximum e4+1
    let mut lower := 0
    for code in [:maximum e4+1] do
      if unit e4 code ≤ magnitude then lower := code
    if unit e4 lower = magnitude then return sign+lower
    let result := choose lower (magnitude-unit e4 lower)
      (unit e4 (lower+1)-unit e4 lower) (entropy e4 totals lane) (2^(7*width e4))
    return sign+result

def predict (e4 : Bool) (inputs : Array (Array Nat)) : Array (Option Nat) := Id.run do
  if !(inputs.all fun r => r.size = 8 ∧ r.all (· < 256)) then return Array.replicate 8 none
  let totals := (Array.range 8).map fun lane => sumWords e4 (inputs.toList.map fun r => r[lane]!)
  return (Array.range 8).map fun lane => some (output e4 totals lane)

#print axioms e4_decode_agreement
#print axioms e5_decode_agreement
#print axioms choose_adjacent
end FabricFP8
