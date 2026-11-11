/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FloatLib.Floats.Formats.BinaryInterchange.Configured
import FloatLib.Floats.Formats.BinaryInterchange.Reduction.Proof

set_option maxRecDepth 100000
set_option maxHeartbeats 0
namespace FabricF16
open FloatLib.Floats.Formats.BinaryInterchange

def predict (words : List Nat) : Nat := Id.run do
  let nans := words.filter fun w => w%32768>31744
  if !nans.isEmpty then return nans.foldl (fun p w => max p (w|||512)) 0
  let infinities := words.filter fun w => w%32768=31744
  if !infinities.isEmpty then
    let negative := infinities.any fun w => w≥32768
    let positive := infinities.any fun w => w<32768
    if negative ∧ positive then return 32256
    return if negative then 64512 else 31744
  let values := words.toArray.map fun w => (Model.ofNatBits w : Model FloatFormat.binary16)
  let result := (Model.sum FloatFormat.binary16 values .nearestEven).toNatBits
  return if result%32768=0 then 0 else result

def wide (words : List Nat) : Nat :=
  let values := words.toArray.map fun w => (Model.ofNatBits w : Model FloatFormat.binary16)
  let summed := Model.sum FloatFormat.binary32 values .nearestEven
  (Model.castWithRounding FloatFormat.binary32 FloatFormat.binary16 summed .nearestEven).toNatBits

def midpoint : List Nat := [0x3c00,0x1000,0x0001,0]
def cancellation : List Nat := [0x7bff,0xfbff,1,0]

theorem midpoint_exact : predict midpoint=0x3c01 := by decide +kernel
theorem midpoint_wide : wide midpoint=0x3c00 := by decide +kernel
theorem cancellation_exact : predict cancellation=1 := by decide +kernel
theorem negative_zero : predict [0x8000,0x8000,0x8000,0x8000]=0 := by decide +kernel
theorem max_quiet_nan : predict [0x7c01,0xfd01,0x7fff,0]=0xff01 := by decide +kernel

#print axioms midpoint_exact
#print axioms midpoint_wide
#print axioms cancellation_exact
#print axioms negative_zero
#print axioms max_quiet_nan
end FabricF16
