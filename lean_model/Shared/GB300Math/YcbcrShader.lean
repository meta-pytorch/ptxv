/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FloatLib.Floats.Formats.BinaryInterchange.Configured

namespace GB300Math.YcbcrShader
open FloatLib.Floats.Formats.BinaryInterchange

def normalized (byte : Nat) : Nat :=
  let bits := if byte = 0 then 0 else
    (byte.log2+127)*2^23 + (byte*2^(23-byte.log2))%2^23
  Model.toNatBits (Model.divWithRounding .nearestEven
    (Model.ofNatBits bits : Model FloatFormat.binary32) (Model.ofNatBits 0x437f0000))

def add (a b : Nat) : Nat :=
  Model.toNatBits (Model.addWithRounding .nearestEven
    (Model.ofNatBits a : Model FloatFormat.binary32) (Model.ofNatBits b))

def fma (a b c : Nat) : Nat :=
  Model.toNatBits (Model.fmaWithRounding .nearestEven
    (Model.ofNatBits a : Model FloatFormat.binary32)
    (Model.ofNatBits b) (Model.ofNatBits c))

def convert (mode y cb cr : Nat) : Option (List Nat) := do
  if mode > 4 || y > 255 || cb > 255 || cr > 255 then none else pure ()
  let yy := normalized y
  let bb := normalized cb
  let rr := normalized cr
  if mode = 0 then return [rr, yy, bb, 0x3f800000]
  let bb := add bb 0xbf008081
  let rr := add rr 0xbf008081
  if mode = 1 then return [rr, yy, bb, 0x3f800000]
  let (kgB, kB, kR, kgR) := if mode = 2 then
      (0xbe3fd1ef,0x3fed844d,0x3fc9930c,0xbeefadfc)
    else if mode = 3 then (0xbeb032a1,0x3fe2d0e5,0x3fb374bc,0xbf36d1a2)
    else (0xbe28809e,0x3ff0d1b7,0x3fbcbfb2,0xbf124433)
  return [fma rr kR yy, fma rr kgR (fma bb kgB yy), fma bb kB yy, 0x3f800000]

theorem identity_channel_order (y cb cr : Nat) (hy : y ≤ 255)
    (hb : cb ≤ 255) (hr : cr ≤ 255) :
    convert 0 y cb cr = some [normalized cr, normalized y, normalized cb, 0x3f800000] := by
  simp [convert,hy,hb,hr]

end GB300Math.YcbcrShader
