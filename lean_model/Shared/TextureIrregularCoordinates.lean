/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.TextureIrregularCoordinates

def decoded (value : Nat) : Int × Nat :=
  let e : Nat := value/2^23%256
  if e = 0 ∨ e = 255 then (0,1) else
  let m : Int := if value ≥ 2^31 then -((value%2^23+2^23 : Nat) : Int) else ((value%2^23+2^23 : Nat) : Int)
  let scale := (e : Int)-150
  if scale ≥ 0 then (m*2^scale.toNat,1) else (m,2^(-scale).toNat)

def precision (size : Nat) : Nat := if size ≤ 8192 then 21 else 22

def scaled (value size : Nat) (normalized : Bool) (precisionBits : Nat := 21) : Int × Nat :=
  let (m,d) := decoded value
  if normalized then ((m*2^precisionBits/d)*(size : Int),2^precisionBits) else (m,d)

def effectiveMode (mode : Nat) (normalized : Bool) : Nat :=
  if !normalized && (mode = 0 || mode = 2) then 1 else mode

def index (i : Int) (size mode : Nat) : Option Nat := Id.run do
  if mode = 0 then return some (i%(size : Int)).toNat
  if mode = 1 then return some (min (size-1) i.toNat)
  if mode = 2 then
    let r := (i%((2*size) : Int)).toNat
    return some (if r < size then r else 2*size-1-r)
  if i < 0 ∨ i ≥ size then return none
  return some i.toNat

def point (value size mode : Nat) (normalized : Bool) (precisionBits : Nat := 21) : Option Nat := Id.run do
  let effective := effectiveMode mode normalized
  if value%2^31 > 0x7f800000 then return some 0
  if value%2^31 = 0x7f800000 then
    if effective = 3 then return none
    return some (if effective = 1 && value < 2^31 then size-1 else 0)
  let (m,d) := scaled value size normalized precisionBits
  return index (m/d) size effective

def position (value size mode : Nat) (normalized : Bool) (precisionBits : Nat := 21) : Int :=
  let effective := effectiveMode mode normalized
  let (m,d) := scaled value size normalized precisionBits
  let raw := if value%2^31 = 0x7f800000 ∧ (effective = 1 ∨ effective = 3) then
      (if value < 2^31 then 1 else -1)*(size : Int)*1024
    else (2*m*256+(d : Int))/(2*d)-128
  if effective = 1 then max 0 (min (((size-1)*256 : Nat) : Int) raw) else raw

def weightPlane (x y total : Nat) : Array Nat :=
  let highX := (x*total+128)/256
  let lowX := total-highX
  let lowLow := (lowX*(256-y)+128)/256
  let highHigh := (highX*y+128)/256
  #[lowLow,highX-highHigh,lowX-lowLow,highHigh]

def scaledBits (n : Nat) : Nat :=
  if n = 0 then 0 else (n.log2+119)*2^23+(n*2^(23-n.log2))%2^23

def evaluate (sizes : Array Nat) (modes : Nat) (normalized linear : Bool) (vs : Array Nat) : Nat := Id.run do
  let precisionBits := precision (sizes.foldl max 0)
  if !linear then
    let mut offset := 0
    let mut stride := 1
    for axis in [:sizes.size] do
      match point vs[axis]! sizes[axis]! (modes/4^axis%4) normalized precisionBits with
      | none => return 0
      | some i =>
        offset := offset+stride*i
        stride := stride*sizes[axis]!
    return scaledBits ((offset+1)*256)
  let ps := sizes.mapIdx (fun axis size => position vs[axis]! size (modes/4^axis%4) normalized precisionBits)
  let x := (ps[0]!%256).toNat
  let y := (ps[1]!%256).toNat
  let z := if sizes.size = 2 then 0 else (ps[2]!%256).toNat
  let ws := weightPlane x y (256-z) ++ weightPlane x y z
  let mut total := 0
  for corner in [:2^sizes.size] do
    let mut offset := 0
    let mut stride := 1
    let mut valid := true
    for axis in [:sizes.size] do
      let i := ps[axis]!/256+(corner/2^axis%2 : Nat)
      match index i sizes[axis]! (effectiveMode (modes/4^axis%4) normalized) with
      | none => valid := false
      | some j =>
        offset := offset+stride*j
        stride := stride*sizes[axis]!
    if valid then total := total+ws[corner]!*(offset+1)
  return scaledBits total

theorem third_selects_lower : point 0x3eaaaaab 3 1 true = some 0 := by decide +kernel

theorem fifth_selects_lower : point 0x3e4ccccd 5 1 true = some 0 := by decide +kernel

theorem negative_rounding_is_floor : scaled 0xb3000000 3 true = (-3,2097152) := by decide +kernel

theorem filtered_boundary : evaluate #[3,5] 5 true true #[1051350358,1056964608] = 1089462272 := by decide +kernel

theorem precision_transitions : precision 8192 = 21 ∧ precision 8193 = 22 ∧
    precision 16384 = 22 ∧ precision 16385 = 22 := by decide +kernel

theorem grid_remainder_bound (m : Int) (d p : Nat) (hd : 0 < d) :
    0 ≤ m*2^p-(m*2^p/d)*d ∧ m*2^p-(m*2^p/d)*d < d := by
  have hpos : (0 : Int) < d := by omega
  have hnon := Int.emod_nonneg (m*2^p) (by omega : (d : Int) ≠ 0)
  have hlt := Int.emod_lt_of_pos (m*2^p) hpos
  rw [Int.emod_def] at hnon hlt
  simpa [Int.mul_comm] using And.intro hnon hlt

#print axioms precision_transitions
#print axioms grid_remainder_bound

#print axioms third_selects_lower
#print axioms fifth_selects_lower
#print axioms negative_rounding_is_floor
#print axioms filtered_boundary

def word (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut n := 0
  for j in [:4] do n := n+data[offset+j]!.toNat*2^(8*j)
  return n

end GB300Math.TextureIrregularCoordinates

open GB300Math.TextureIrregularCoordinates

def main (args : List String) : IO UInt32 := do
  let [wText,hText,dText,mText,nText,fText,input,output] := args |
    throw (IO.userError "Expected width height depth modes normalized linear inputs outputs")
  let some w := wText.toNat? | throw (IO.userError "Invalid width")
  let some h := hText.toNat? | throw (IO.userError "Invalid height")
  let some d := dText.toNat? | throw (IO.userError "Invalid depth")
  let some m := mText.toNat? | throw (IO.userError "Invalid modes")
  let some n := nText.toNat? | throw (IO.userError "Invalid normalized flag")
  let some f := fText.toNat? | throw (IO.userError "Invalid filter")
  let sizes := if d = 0 then #[w,h] else #[w,h,d]
  unless w > 0 ∧ h > 0 ∧ w*h*(max d 1) ≤ 32768 ∧ m < 4^sizes.size ∧ n < 2 ∧ f < 2 do
    throw (IO.userError "Unsupported configuration")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 ∧ xs.size%16 = 0 ∧ ys.size = xs.size/16*4 do
    throw (IO.userError "Invalid capture sizes")
  let mut bad := 0
  for i in [:xs.size/16] do
    let vs := (Array.range 3).map (fun j => word xs (i*16+4*j))
    let expected := evaluate sizes m (n = 1) (f = 1) vs
    let actual := word ys (i*4)
    if expected ≠ actual then
      if bad < 3 then IO.println s!"Mismatch row={i} model={expected} hardware={actual}"
      bad := bad+1
  IO.println s!"result count={xs.size/16} mismatches={bad}"
  return if bad = 0 then 0 else 1
