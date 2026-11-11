/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureRatio

namespace GB300Math.TexturePackedViews

def readLE (bytes : ByteArray) (offset count : Nat) : Nat :=
  (List.range count).foldl (fun v i => v + bytes[offset+i]!.toNat * 2^(8*i)) 0

def coordinate (bits : Nat) : Nat :=
  let e := bits / 2^23 % 256
  let m := bits % 2^23 + if e = 0 then 0 else 2^23
  m / 2^(150-max e 1)

def convert (bits : Nat) (raw : Bool) (value : Nat) : Nat :=
  if raw then value else TextureRatio.ratioBits value (2^bits-1)

theorem raw_preserves (bits value : Nat) : convert bits true value = value := rfl
theorem normalized_zero (bits : Nat) : convert bits false 0 = 0 := by
  simp [convert, TextureRatio.ratioBits]
theorem normalized_u8_max : convert 8 false 255 = 0x3f800000 := by decide
theorem normalized_u16_max : convert 16 false 65535 = 0x3f800000 := by decide
#print axioms raw_preserves
#print axioms normalized_zero
#print axioms normalized_u8_max
#print axioms normalized_u16_max
def linearIndex (masked : Bool) (x y : Nat) : Nat :=
  y * 64 + if masked then x % 32 else x

theorem index_bound (masked : Bool) (x y : Nat) (hx : x < 64) (hy : y < 64) :
    linearIndex masked x y < 4096 := by
  have hm := Nat.mod_lt x (by decide : 0 < 32)
  unfold linearIndex
  split <;> omega

theorem masked_periodic (x y : Nat) :
    linearIndex true (x + 32) y = linearIndex true x y := by
  simp [linearIndex]

#print axioms index_bound
#print axioms masked_periodic
end GB300Math.TexturePackedViews
