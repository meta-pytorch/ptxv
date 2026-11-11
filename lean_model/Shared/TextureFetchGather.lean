/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.TextureFetchGather

def ratioBits (n d : Nat) : Nat := Id.run do
  if n = 0 then return 0
  let tentative := (n.log2 : Int)-(d.log2 : Int)
  let below := if tentative ≥ 0 then n < d*2^tentative.toNat
    else n*2^(-tentative).toNat < d
  let e := if below then tentative-1 else tentative
  let scaled := n*2^(23-e).toNat
  let q := scaled/d
  let r := scaled%d
  let m := q + if 2*r > d ∨ (2*r = d ∧ q%2 = 1) then 1 else 0
  let (m,e) := if m ≥ 2^24 then (m/2,e+1) else (m,e)
  return (e+127).toNat*2^23+m%2^23

def halfToFloat (v : Nat) : Nat := Id.run do
  let sign := v/2^15*2^31
  let e := v/2^10%32
  let m := v%2^10
  if e = 0 then
    if m = 0 then return sign
    return sign+(m.log2+103)*2^23+(m*2^(23-m.log2))%2^23
  if e = 31 then return sign+0x7f800000+m*2^13
  return sign+(e+112)*2^23+m*2^13

def signedValue (width value : Nat) : Int :=
  if value ≥ 2^(width-1) then (value : Int)-2^width else value

def signedNormalized (n : Int) (denominator : Nat) : Nat :=
  (if n < 0 then 2^31 else 0)+ratioBits (min n.natAbs denominator) denominator

def snorm8Filtered (value : Nat) : Nat :=
  let n := signedValue 8 value
  let m := n.natAbs
  let extension := 258*m + if (if n < 0 then m > 64 else m ≥ 64) then 1 else 0
  signedNormalized (if n < 0 then -(extension : Int) else extension) 32767

def evaluate (format mode value : Nat) : Nat :=
  match format with
  | 0 => if mode = 2 then value else ratioBits value 255
  | 1 => if mode = 2 then value else ratioBits value 65535
  | 2 => if mode = 2 then ((signedValue 8 value)%2^32).toNat
      else if mode = 1 then snorm8Filtered value
      else signedNormalized (signedValue 8 value) 127
  | 3 => if mode = 2 then ((signedValue 16 value)%2^32).toNat
      else signedNormalized (signedValue 16 value) 32767
  | 4 => if mode = 1 ∧ value%2^15 > 0x7c00 then 0x7fffe000 else halfToFloat value
  | 5 | 6 => value
  | 7 => if mode = 0 then value
      else if value%2^31 > 0x7f800000 then 0x7fffffff
      else if value/2^23%256 = 0 then value/2^31*2^31 else value
  | _ => 0

def width (format : Nat) : Nat := if format = 0 ∨ format = 2 then 8
  else if format = 1 ∨ format = 3 ∨ format = 4 then 16 else 32

def valid (format mode : Nat) : Bool :=
  format < 8 && (if format = 5 ∨ format = 6 then mode = 2
    else if format = 4 ∨ format = 7 then mode < 2 else mode < 3)

theorem snorm8_center_differs :
    evaluate 2 0 1 = 0x3c010204 ∧ evaluate 2 1 1 = 0x3c010102 := by decide +kernel

theorem snorm8_filtered_asymmetry :
    evaluate 2 1 64 = 0x3f010302 ∧ evaluate 2 1 192 = 0xbf010102 := by decide +kernel

theorem half_signaling_payload_preserved : evaluate 4 0 0xfc01 = 0xff802000 := by decide +kernel

theorem half_filtered_nan_canonical : evaluate 4 1 0xfc01 = 0x7fffe000 := by decide +kernel

theorem half_subnormal_widens : evaluate 4 0 1 = 0x33800000 := by decide +kernel

theorem signed_min_clamps : evaluate 3 0 0x8000 = 0xbf800000 := by decide +kernel

theorem signed_raw_extends : evaluate 2 2 0x80 = 0xffffff80 := by decide +kernel

theorem unsigned_endpoint : evaluate 1 0 0xffff = 0x3f800000 := by decide +kernel

#print axioms snorm8_center_differs
#print axioms snorm8_filtered_asymmetry
#print axioms half_signaling_payload_preserved
#print axioms half_filtered_nan_canonical
#print axioms half_subnormal_widens
#print axioms signed_min_clamps
#print axioms signed_raw_extends
#print axioms unsigned_endpoint

def word (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut n := 0
  for j in [:4] do n := n+data[offset+j]!.toNat*2^(8*j)
  return n

def sourceIndex (operation component row lane : Nat) : Nat :=
  if operation = 0 then row
  else if operation = 1 then row*4+lane
  else
    let dx := if lane = 1 ∨ lane = 2 then 1 else 0
    let dy := if lane = 0 ∨ lane = 1 then 1 else 0
    let x := min 255 (row%256+dx)
    let y := min 255 (row/256+dy)
    (x+256*y)*4+component

theorem gather_order :
    (Array.range 4).map (sourceIndex 2 0 0) = #[1024,1028,4,0] := by decide +kernel

theorem gather_corner_clamps : sourceIndex 2 3 65535 1 = 262143 := by decide +kernel

#print axioms gather_order
#print axioms gather_corner_clamps

end GB300Math.TextureFetchGather

open GB300Math.TextureFetchGather

def main (args : List String) : IO UInt32 := do
  let [operationText,componentText,formatText,modeText,input,output] := args |
    throw (IO.userError "Expected operation component format mode inputs outputs")
  let some operation := operationText.toNat? | throw (IO.userError "Invalid operation")
  let some component := componentText.toNat? | throw (IO.userError "Invalid component")
  let some format := formatText.toNat? | throw (IO.userError "Invalid format")
  let some mode := modeText.toNat? | throw (IO.userError "Invalid mode")
  unless operation < 3 ∧ component < 4 ∧ (operation = 2 ∨ component = 0) ∧ valid format mode ∧ mode ≠ 1 do
    throw (IO.userError "Unsupported configuration")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  let lanes := if operation = 0 then 1 else 4
  unless xs.size = 65536*lanes*4 ∧ ys.size > 0 ∧ ys.size ≤ 65536*lanes*4 ∧ ys.size%(lanes*4) = 0 do
    throw (IO.userError "Invalid capture sizes")
  for i in [:xs.size/4] do
    unless word xs (i*4) < 2^(width format) do
      throw (IO.userError "Input exceeds format width")
  let mut bad := 0
  for i in [:ys.size/4] do
    let value := word xs (sourceIndex operation component (i/lanes) (i%lanes)*4)
    let conversion := if operation = 2 ∧ mode = 0 then 1 else mode
    let expected := evaluate format conversion value
    let actual := word ys (i*4)
    if expected ≠ actual then
      if bad < 3 then IO.println s!"Mismatch row={i} model={expected} hardware={actual}"
      bad := bad+1
  IO.println s!"result count={ys.size/4} mismatches={bad}"
  return if bad = 0 then 0 else 1
