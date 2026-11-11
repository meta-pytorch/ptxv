/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.YcbcrShader
import TextureFormats

set_option maxRecDepth 100000
set_option maxHeartbeats 0

namespace GB300Math.YcbcrShader

theorem normalization_matches_texture :
    ∀ byte : Fin 256, normalized byte.val = TextureFormats.ratioBits byte.val 255 := by
  decide +kernel

theorem normalization_is_zero_or_normal :
    ∀ byte : Fin 256, normalized byte.val = 0 ∨
      (119 ≤ normalized byte.val / 2^23 ∧ normalized byte.val / 2^23 ≤ 127) := by
  decide +kernel

def decodedUnits (bits : Nat) : Nat :=
  if bits = 0 then 0 else (2^23 + bits % 2^23) * 2^(bits / 2^23 - 1)

theorem normalization_tight_error :
    ∀ byte : Fin 256,
      ((255 * decodedUnits (normalized byte.val) : Int) - byte.val * 2^149).natAbs
        ≤ 127 * 2^125 := by
  decide +kernel

theorem normalization_error_attained :
    ((255 * decodedUnits (normalized 128) : Int) - 128 * 2^149).natAbs
      = 127 * 2^125 := by
  decide +kernel

open FloatLib.Floats.Formats.BinaryInterchange

def normalizedValue (byte : Nat) : Option Rat :=
  (Model.ofNatBits (normalized byte) : Model FloatFormat.binary32).toRat?

theorem normalization_floatlib_decode :
    ∀ byte : Fin 256,
      normalizedValue byte.val = some ((decodedUnits (normalized byte.val) : Rat) / 2^149) := by
  decide +kernel

theorem normalization_floatlib_tight_error :
    ∀ byte : Fin 256, ∃ value : Rat,
      normalizedValue byte.val = some value ∧
      |value - (byte.val : Rat) / 255| ≤ 127 / (255 * 2^24) := by
  intro byte
  refine ⟨(decodedUnits (normalized byte.val) : Rat) / 2^149,
    normalization_floatlib_decode byte, ?_⟩
  have h : ∀ b : Fin 256,
      |(decodedUnits (normalized b.val) : Rat) / 2^149 - (b.val : Rat) / 255|
        ≤ 127 / (255 * 2^24) := by decide +kernel
  exact h byte

theorem normalization_floatlib_error_attained :
    ∃ value : Rat, normalizedValue 128 = some value ∧
      |value - 128 / 255| = 127 / (255 * 2^24) := by
  refine ⟨(decodedUnits (normalized 128) : Rat) / 2^149, ?_, ?_⟩
  · exact normalization_floatlib_decode ⟨128, by decide⟩
  · decide +kernel

end GB300Math.YcbcrShader

#print axioms GB300Math.YcbcrShader.normalization_matches_texture
#print axioms GB300Math.YcbcrShader.normalization_is_zero_or_normal
#print axioms GB300Math.YcbcrShader.normalization_tight_error
#print axioms GB300Math.YcbcrShader.normalization_error_attained

#print axioms GB300Math.YcbcrShader.normalization_floatlib_decode
#print axioms GB300Math.YcbcrShader.normalization_floatlib_tight_error
#print axioms GB300Math.YcbcrShader.normalization_floatlib_error_attained
