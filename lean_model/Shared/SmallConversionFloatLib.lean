/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FloatLib.Floats.Formats.BinaryInterchange.Configured
import SmallConversions

set_option maxRecDepth 100000
set_option maxHeartbeats 50000000

namespace GB300Math.SmallConversionFloatLib

open FloatLib.Floats GB300Math.SmallConversions

def idealValue (name : String) (fmt : Format) (word : Nat) : Option ℚ :=
  let e := word / 2^fmt.fractionBits % 2^fmt.exponentBits
  let f := word % 2^fmt.fractionBits
  if (name = "e4m3" ∧ e = 15 ∧ f = 7) ∨ (name = "e5m2" ∧ e = 31) then none else
    let negative := word / 2^(fmt.fractionBits+fmt.exponentBits) % 2 = 1
    let significand := f + if e = 0 then 0 else 2^fmt.fractionBits
    let power : Int := (max e 1 : Int) - (fmt.bias : Int) - (fmt.fractionBits : Int)
    some ((if negative then -1 else 1) * (significand : ℚ) * (2 : ℚ)^power)

def decodedValue (name : String) (fmt : Format) (word : Nat) : Option ℚ :=
  let bits := upSmall name fmt false false false false word 127
  ExecFloat.Binary.toRat? (ExecFloat.Binary.ofNatBits bits : ExecFloat.Binary 5 10)

theorem e4m3_decode_exact : ∀ code : Fin 256,
    decodedValue "e4m3" ⟨4, 3, 7, 126⟩ code.val =
    idealValue "e4m3" ⟨4, 3, 7, 126⟩ code.val := by decide +kernel

theorem e5m2_decode_exact : ∀ code : Fin 256,
    decodedValue "e5m2" ⟨5, 2, 15, 123⟩ code.val =
    idealValue "e5m2" ⟨5, 2, 15, 123⟩ code.val := by decide +kernel

theorem e2m3_decode_exact : ∀ code : Fin 64,
    decodedValue "e2m3" ⟨2, 3, 1, 31⟩ code.val =
    idealValue "e2m3" ⟨2, 3, 1, 31⟩ code.val := by decide +kernel

theorem e3m2_decode_exact : ∀ code : Fin 64,
    decodedValue "e3m2" ⟨3, 2, 3, 31⟩ code.val =
    idealValue "e3m2" ⟨3, 2, 3, 31⟩ code.val := by decide +kernel

theorem e2m1_decode_exact : ∀ code : Fin 16,
    decodedValue "e2m1" ⟨2, 1, 1, 7⟩ code.val =
    idealValue "e2m1" ⟨2, 1, 1, 7⟩ code.val := by decide +kernel

theorem e8m0_decode_exact : ∀ code : Fin 255,
    let bits := if code.val = 0 then 64 else code.val*128
    ExecFloat.Binary.toRat? (ExecFloat.Binary.ofNatBits bits : ExecFloat.Binary 8 7) =
      some ((2 : ℚ)^((code.val : Int)-127)) := by decide +kernel

#print axioms e4m3_decode_exact
#print axioms e5m2_decode_exact
#print axioms e2m3_decode_exact
#print axioms e3m2_decode_exact
#print axioms e2m1_decode_exact
#print axioms e8m0_decode_exact

end GB300Math.SmallConversionFloatLib
