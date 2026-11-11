/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FabricF32
namespace FabricF32Scale
open FloatLib.Floats.Formats.BinaryInterchange

def original : List Nat := [95*2^23, 2^31+95*2^23, 9*2^23, 0]
def doubled : List Nat := [96*2^23, 2^31+96*2^23, 10*2^23, 0]
def lifted16 : List Nat := [111*2^23, 2^31+111*2^23, 25*2^23, 0]

def decode (words : List Nat) : List (Option Rat) :=
  words.map fun w => (Model.ofNatBits w : Model (.ieee 8 23)).toRat?

theorem doubled_inputs : decode doubled = (decode original).map (Option.map (2*·)) := by decide +kernel
theorem lifted_inputs : decode lifted16 = (decode original).map (Option.map (65536*·)) := by decide +kernel

theorem original_output : FabricF32.predict original = 9*2^23 := by decide +kernel
theorem doubled_output : FabricF32.predict doubled = 0 := by decide +kernel
theorem lifted_output : FabricF32.predict lifted16 = 25*2^23 := by decide +kernel

theorem original_value :
    (Model.ofNatBits (9*2^23) : Model (.ieee 8 23)).toRat? = some (1/(2^118 : Rat)) := by decide +kernel

theorem nonzero_inputs_normal :
    [original, doubled, lifted16].all (fun ws => ws.all fun w => w = 0 ∨ (0 < w/2^23%256 ∧ w/2^23%256 < 255)) = true := by decide +kernel

#print axioms doubled_inputs
#print axioms lifted_inputs
#print axioms original_output
#print axioms doubled_output
#print axioms lifted_output
#print axioms original_value
#print axioms nonzero_inputs_normal
end FabricF32Scale
