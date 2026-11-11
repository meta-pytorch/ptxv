/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.TextureLinearCoordinates

def weightPlane (x y total : Nat) : Array Nat :=
  let highX := (x*total+128)/256
  let lowX := total-highX
  let lowLow := (lowX*(256-y)+128)/256
  let highHigh := (highX*y+128)/256
  #[lowLow,highX-highHigh,lowX-lowLow,highHigh]

def weights (x y z : Nat) : Array Nat :=
  weightPlane x y (256-z) ++ weightPlane x y z

def support (x y z i : Nat) : Bool :=
  (if i%2 = 0 then x < 256 else x > 0) &&
  (if i/2%2 = 0 then y < 256 else y > 0) &&
  (if i/4 = 0 then z < 256 else z > 0)

def coordinate (value logSize : Nat) (normalized : Bool) : Int :=
  let e : Nat := value/2^23%256
  if e = 0 ∨ e = 255 then -128 else
  let m : Nat := value%2^23+2^23
  let scale := (e : Int)-142 + if normalized then logSize else 0
  let signed : Int := if value ≥ 2^31 then -(m : Int) else m
  let rounded := if scale ≥ 0 then signed*2^scale.toNat
    else (signed+2^((-scale).toNat-1))/2^(-scale).toNat
  rounded-128

def effectiveMode (mode : Nat) (normalized : Bool) : Nat :=
  if !normalized && (mode = 0 || mode = 2) then 1 else mode

def position (value logSize mode : Nat) (normalized : Bool) : Int :=
  let effective := effectiveMode mode normalized
  let raw := if value%2^31 = 0x7f800000 ∧ (effective = 1 ∨ effective = 3) then
      (if value < 2^31 then 1 else -1)*2^(logSize+10)
    else coordinate value logSize normalized
  if effective = 1 then max 0 (min (((2^logSize : Nat)-1)*256 : Nat) raw) else raw

def index (i : Int) (logSize mode : Nat) (normalized : Bool) : Option Nat := Id.run do
  let size : Nat := 2^logSize
  let effective := effectiveMode mode normalized
  if effective = 0 then return some (i%(size : Int)).toNat
  if effective = 1 then return some (min (size-1) i.toNat)
  if effective = 2 then
    let r := (i%((2*size) : Int)).toNat
    return some (if r < size then r else 2*size-1-r)
  if i < 0 ∨ i ≥ size then return none
  return some i.toNat

def texel (dimensions modes : Nat) (normalized : Bool) (positions : Array Int) (corner : Nat) : Nat := Id.run do
  let logs := if dimensions = 2 then #[8,8,0] else #[6,5,5]
  let mut offset := 0
  let mut stride := 1
  for axis in [:dimensions] do
    let i := positions[axis]!/256 + (corner/2^axis%2 : Int)
    match index i logs[axis]! (modes/4^axis%4) normalized with
    | none => return 0
    | some j =>
      offset := offset+stride*j
      stride := stride*2^logs[axis]!
  return offset+1

def scaledBits (n : Nat) : Nat :=
  if n = 0 then 0 else (n.log2+119)*2^23+(n*2^(23-n.log2))%2^23

def evaluate (dimensions modes : Nat) (normalized : Bool) (vs : Array Nat) : Nat := Id.run do
  let logs := if dimensions = 2 then #[8,8,0] else #[6,5,5]
  let ps := (Array.range 3).map (fun axis => position vs[axis]! logs[axis]! (modes/4^axis%4) normalized)
  let x := (ps[0]!%256).toNat
  let y := (ps[1]!%256).toNat
  let z := if dimensions = 2 then 0 else (ps[2]!%256).toNat
  let ws := weights x y z
  let total := (Array.range (if dimensions = 2 then 4 else 8)).foldl
    (fun n i => n+ws[i]!*texel dimensions modes normalized ps i) 0
  return scaledBits total

theorem clamp_before_coefficients : evaluate 3 0 false #[1058264707,1063907344,1040001492] = 1104977920 := by decide +kernel

theorem coordinate_halfway_up : coordinate 0x3f008000 8 false = 1 := by decide +kernel

theorem coordinate_negative_halfway_up : coordinate 0xbf008000 8 false = -256 := by decide +kernel

theorem coordinate_subnormal_flush : coordinate 0x80000001 8 true = -128 := by decide +kernel

theorem coordinate_nan_zero : coordinate 0xff800001 8 true = -128 := by decide +kernel

theorem origin_border_quarter : evaluate 2 15 false #[0,0,0] = 0x3e800000 := by decide +kernel

theorem origin_wrap_average : evaluate 2 0 true #[0,0,0] = 0x47000080 := by decide +kernel

theorem texel_center_identity : evaluate 2 5 false #[0x3f000000,0x3f000000,0] = 0x3f800000 := by decide +kernel

theorem infinity_border_zero : evaluate 2 15 true #[0x7f800000,0x3f000000,0] = 0 := by decide +kernel

#print axioms clamp_before_coefficients
#print axioms coordinate_halfway_up
#print axioms coordinate_negative_halfway_up
#print axioms coordinate_subnormal_flush
#print axioms coordinate_nan_zero
#print axioms origin_border_quarter
#print axioms origin_wrap_average
#print axioms texel_center_identity
#print axioms infinity_border_zero

def word (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut n := 0
  for j in [:4] do n := n+data[offset+j]!.toNat*2^(8*j)
  return n

end GB300Math.TextureLinearCoordinates

open GB300Math.TextureLinearCoordinates

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
