/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.TextureAddress

def floorCoordinate (value logSize : Nat) (normalized : Bool) : Int :=
  let e : Nat := value/2^23%256
  if e = 0 ∨ e = 255 then 0 else
  let m : Nat := value%2^23+2^23
  let scale := (e : Int)-150 + if normalized then logSize else 0
  let signed : Int := if value ≥ 2^31 then -(m : Int) else m
  if scale ≥ 0 then signed*2^scale.toNat else signed/2^(-scale).toNat

def address (value logSize mode : Nat) (normalized : Bool) : Option Nat := Id.run do
  let size := 2^logSize
  let effective := if !normalized && (mode = 0 || mode = 2) then 1 else mode
  let mag := value%2^31
  if mag > 0x7f800000 then return some 0
  if mag = 0x7f800000 then
    if effective = 3 then return none
    if effective = 1 && value < 2^31 then return some (size-1)
    return some 0
  let i := floorCoordinate value logSize normalized
  if effective = 0 then return some (i%(size : Int)).toNat
  if effective = 1 then return some (min (size-1) i.toNat)
  if effective = 2 then
    let r := (i%((2*size) : Int)).toNat
    return some (if r < size then r else 2*size-1-r)
  if i < 0 ∨ i ≥ size then return none
  return some i.toNat

def integerBits (n : Nat) : Nat :=
  if n = 0 then 0 else (n.log2+127)*2^23+(n*2^(23-n.log2))%2^23

def evaluate (dimensions modes : Nat) (normalized : Bool) (vs : Array Nat) : Nat := Id.run do
  let logs := if dimensions = 2 then #[8,8,0] else #[6,5,5]
  let mut offset := 0
  let mut stride := 1
  for axis in [:dimensions] do
    match address vs[axis]! logs[axis]! (modes/4^axis%4) normalized with
    | none => return 0
    | some i =>
      offset := offset+stride*i
      stride := stride*2^logs[axis]!
  return integerBits (offset+1)

theorem negative_subnormal_flushes_before_border : address 0x80000001 8 3 false = some 0 := by decide +kernel

theorem negative_normal_wraps : address 0x80800000 8 0 true = some 255 := by decide +kernel

theorem nan_selects_zero : address 0xff800001 8 3 true = some 0 := by decide +kernel

theorem positive_infinity_clamps : address 0x7f800000 8 1 true = some 255 := by decide +kernel

theorem infinity_border : address 0x7f800000 8 3 true = none := by decide +kernel

theorem negative_mirror : address 0xbe800000 8 2 true = some 63 := by decide +kernel

theorem unnormalized_wrap_is_clamp : address 0xbf800000 8 0 false = some 0 := by decide +kernel

theorem exact_normalized_endpoint_wraps : address 0x3f800000 8 0 true = some 0 := by decide +kernel

#print axioms negative_subnormal_flushes_before_border
#print axioms negative_normal_wraps
#print axioms nan_selects_zero
#print axioms positive_infinity_clamps
#print axioms infinity_border
#print axioms negative_mirror
#print axioms unnormalized_wrap_is_clamp
#print axioms exact_normalized_endpoint_wraps

def word (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut n := 0
  for j in [:4] do n := n+data[offset+j]!.toNat*2^(8*j)
  return n

end GB300Math.TextureAddress

open GB300Math.TextureAddress

def main (args : List String) : IO UInt32 := do
  let [dimensionsText,modesText,normalizedText,input,output] := args |
    throw (IO.userError "Expected dimensions modes normalized inputs outputs")
  let some dimensions := dimensionsText.toNat? | throw (IO.userError "Invalid dimensions")
  let some modes := modesText.toNat? | throw (IO.userError "Invalid modes")
  let some normalized := normalizedText.toNat? | throw (IO.userError "Invalid normalized flag")
  unless (dimensions = 2 ∨ dimensions = 3) ∧ modes < 4^dimensions ∧ normalized < 2 do
    throw (IO.userError "Unsupported configuration")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 ∧ xs.size%16 = 0 ∧ ys.size = xs.size/16*4 do
    throw (IO.userError "Invalid coordinate capture sizes")
  let mut bad := 0
  for i in [:xs.size/16] do
    let vs := (Array.range 3).map (fun j => word xs (i*16+4*j))
    let expected := evaluate dimensions modes (normalized = 1) vs
    let actual := word ys (i*4)
    if expected ≠ actual then
      if bad < 3 then IO.println s!"Mismatch row={i} model={expected} hardware={actual}"
      bad := bad+1
  IO.println s!"result count={xs.size/16} mismatches={bad}"
  return if bad = 0 then 0 else 1
