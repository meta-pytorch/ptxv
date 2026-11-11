/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FloatLib.Floats.Formats.BinaryInterchange.Configured

set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace FabricF64
open FloatLib.Floats.Formats.BinaryInterchange

def units (word : Nat) : Option Int := do
  unless word < 2^64 do none
  let d ← (Model.ofNatBits word : Model FloatFormat.binary64).toDyadic?
  unless d.exponent ≥ -1074 do none
  let magnitude := d.significand * 2^(d.exponent+1074).toNat
  return if d.negative then -(magnitude : Int) else magnitude

def maxExponent (words : List Nat) : Nat :=
  words.foldr (fun w e => max ((w % 2^63)/2^52) e) 0

def fineShift (words : List Nat) : Nat := 16*(maxExponent words/16)-29

def coarseShift (words : List Nat) : Nat := 16*(maxExponent words/16)-20

def truncate (shift : Nat) (value : Int) : Int :=
  let magnitude := value.natAbs / 2^shift * 2^shift
  if value < 0 then -(magnitude : Int) else magnitude

def finiteSum (words : List Nat) : Int :=
  (words.map fun w => truncate (fineShift words) ((units w).getD 0)).sum

def flush (shift : Nat) (value : Int) : Int :=
  if value.natAbs < 2^shift then 0 else value

def round (value : Int) : Nat :=
  let d : FloatLib.Numerics.Dyadic :=
    {negative := value < 0, significand := value.natAbs, exponent := -1074}
  (Model.roundDyadic FloatFormat.binary64 d).toNatBits

def predict (words : List Nat) : Nat := Id.run do
  let nans := words.filter fun w => w % 2^63 > 2047*2^52
  if !nans.isEmpty then return nans.foldl (fun p w => max p (w ||| 2^51)) 0
  let infinities := words.filter fun w => w % 2^63 = 2047*2^52
  if !infinities.isEmpty then
    let negative := infinities.any fun w => w ≥ 2^63
    let positive := infinities.any fun w => w < 2^63
    if negative ∧ positive then return 2047*2^52+2^51
    return (if negative then 2^63 else 0)+2047*2^52
  return round (flush (coarseShift words) (finiteSum words))

def cancellation : List Nat := [0x4000000000000000,0xc000000000000000,0x3b00000000000000,0]
def belowBand : List Nat := [0x3fffffffffffffff,0xbfffffffffffffff,0x3af0000000000000,0]
def aboveBand : List Nat := [0x4000000000000000,0xc000000000000000,0x3af0000000000000,0]
def midpointLost : List Nat := [0x3ff0000000000000,0x3ca0000000000000,0x39f0000000000000,0]
def midpointSurvives : List Nat := [0x3ff0000000000000,0x3ca0000000000000,0x3a00000000000000,0]

theorem cancellation_zero : predict cancellation=0 := by decide +kernel

theorem band_changes : predict belowBand=0x3af0000000000000 ∧ predict aboveBand=0 := by decide +kernel

theorem midpoint_lost : predict midpointLost=0x3ff0000000000000 := by decide +kernel

theorem midpoint_survives : predict midpointSurvives=0x3ff0000000000001 := by decide +kernel

theorem quiet_before_max : predict [0x7ff0000000000001,0x7ff8000000000000]=0x7ff8000000000001 := by decide +kernel

theorem signed_nan : predict [0x7fffffffffffffff,0xfff0000000000001]=0xfff8000000000001 := by decide +kernel

theorem negative_zero : predict [2^63,2^63]=0 := by decide +kernel

#print axioms cancellation_zero
#print axioms band_changes
#print axioms midpoint_lost
#print axioms midpoint_survives
#print axioms quiet_before_max
#print axioms signed_nan
#print axioms negative_zero
end FabricF64
