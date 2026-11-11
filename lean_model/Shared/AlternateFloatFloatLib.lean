/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import AlternateFloatSpec
import SmallConversionFloatLibSpec

set_option maxRecDepth 100000
set_option maxHeartbeats 100000000

namespace GB300Math.AlternateFloatFloatLib
open FloatLib.Floats GB300Math.AlternateFloat GB300Math.SmallConversionFloatLib

def rational : Option Value → Option ℚ
  | some (.finite negative magnitude) =>
      some ((if negative then -1 else 1) * (magnitude : ℚ) / (unit : ℚ))
  | _ => none

def halfValue (name : String) (fmt : GB300Math.SmallConversions.Format) (code : Nat) : Option ℚ :=
  decodedValue name fmt code

theorem e4m3_decode_floatlib : ∀ code : Fin 256,
    rational (decode "e4m3" code.val) = halfValue "e4m3" ⟨4, 3, 7, 126⟩ code.val := by
  decide +kernel

theorem e5m2_decode_floatlib : ∀ code : Fin 256,
    rational (decode "e5m2" code.val) = halfValue "e5m2" ⟨5, 2, 15, 123⟩ code.val := by
  decide +kernel

theorem e3m2_decode_floatlib : ∀ code : Fin 256,
    rational (decode "e3m2" code.val) = halfValue "e3m2" ⟨3, 2, 3, 31⟩ (code.val % 64) := by
  decide +kernel

theorem e2m3_decode_floatlib : ∀ code : Fin 256,
    rational (decode "e2m3" code.val) = halfValue "e2m3" ⟨2, 3, 1, 31⟩ (code.val % 64) := by
  decide +kernel

theorem e2m1_decode_floatlib : ∀ code : Fin 16,
    rational (decode "e2m1" code.val) = halfValue "e2m1" ⟨2, 1, 1, 7⟩ code.val := by
  decide +kernel

theorem e2m1p4_decode_floatlib : ∀ code : Fin 256,
    rational (decode "e2m1p4" code.val) = halfValue "e2m1" ⟨2, 1, 1, 7⟩ (code.val % 16) := by
  decide +kernel

theorem ue8m0_decode_floatlib : ∀ code : Fin 255,
    let bits := if code.val = 0 then 64 else code.val*128
    rational (decode "ue8m0" code.val) =
      ExecFloat.Binary.toRat? (ExecFloat.Binary.ofNatBits bits : ExecFloat.Binary 8 7) := by
  decide +kernel

#print axioms e4m3_decode_floatlib
#print axioms e5m2_decode_floatlib
#print axioms e3m2_decode_floatlib
#print axioms e2m3_decode_floatlib
#print axioms e2m1_decode_floatlib
#print axioms e2m1p4_decode_floatlib
#print axioms ue8m0_decode_floatlib
end GB300Math.AlternateFloatFloatLib

namespace GB300Math.AlternateFloatFloatLib
open GB300Math.AlternateFloat

def finiteOnGrid (v : Option Value) : Prop :=
  match v with
  | some (.finite _ magnitude) => magnitude % 2^385 = 0
  | _ => True
instance (v : Option Value) : Decidable (finiteOnGrid v) := by
  unfold finiteOnGrid
  split <;> infer_instance

theorem decoded_inputs_grid : ∀ format : Fin 7, ∀ code : Fin 256,
    finiteOnGrid (decode (["e4m3", "e5m2", "e3m2", "e2m3", "e2m1", "e2m1p4", "ue8m0"][format.val]!) code.val) := by
  decide +kernel

#print axioms decoded_inputs_grid
end GB300Math.AlternateFloatFloatLib
