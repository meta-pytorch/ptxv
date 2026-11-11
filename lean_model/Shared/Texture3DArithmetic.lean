/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.Texture3DArithmetic

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

def magnitude (v : Nat) : Nat := v%2^31

def exponent (v : Nat) : Nat := v/2^23%256

def negative (v : Nat) : Bool := v ≥ 2^31

def encode (value scale : Int) : Nat := Id.run do
  if value = 0 then return 0
  let sign := if value < 0 then 2^31 else 0
  let n := value.natAbs
  let shift := max ((n.log2 : Int)-23) (-149-scale)
  let rounded := if shift > 0 then
      let divisor := 2^shift.toNat
      n/divisor + if n%divisor ≥ divisor/2 then 1 else 0
    else n*2^(-shift).toNat
  let e := scale+shift
  if rounded = 0 then return sign
  let (m,e) := if rounded ≥ 2^24 then (rounded/2,e+1) else (rounded,e)
  let biased := (m.log2 : Int)+e+127
  if biased ≥ 255 then return sign+0x7f800000
  if biased ≤ 0 then return sign
  return sign+biased.toNat*2^23+m%2^23


def groupedExponent (e : Int) : Int := (e+3)/4*4

def planeSum (indices : Array Nat) (values ws : Array Nat) : Int × Int × Int := Id.run do
  if indices.isEmpty then return (0,0,-10000)
  let top := indices.foldl (fun e i => max e (exponent values[i]!)) 0
  let scale := (top : Int)-154
  let mut total : Int := 0
  for i in indices do
    let value := values[i]!
    let mantissa := value%2^23+2^23
    let shift := (exponent value : Int)-150-scale
    let aligned : Nat := if shift ≥ 0 then mantissa*2^shift.toNat else mantissa/2^(-shift).toNat
    let signed : Int := if negative value then -(aligned : Int) else aligned
    total := total+signed*(ws[i]! : Int)
  return (total,scale-8,(top : Int)-127)

def interpolate (x y z : Nat) (values : Array Nat) : Nat := Id.run do
  let ws := weights x y z
  let geometric := (Array.range 8).filter (support x y z)
  if geometric.any (fun i => magnitude values[i]! > 0x7f800000) then return 0x7fffffff
  let infs := geometric.filter (fun i => magnitude values[i]! = 0x7f800000)
  if !infs.isEmpty then
    let sign := negative values[infs[0]!]!
    if infs.any (fun i => negative values[i]! != sign) then return 0x7fffffff
    return (if sign then 2^31 else 0)+0x7f800000
  let finite := geometric.filter (fun i => ws[i]! > 0 && exponent values[i]! > 0)
  if finite.isEmpty then
    return if geometric.all (fun i => negative values[i]! && exponent values[i]! = 0)
      then 2^31 else 0
  let a := planeSum (finite.filter (· < 4)) values ws
  let b := planeSum (finite.filter (· ≥ 4)) values ws
  let scale := groupedExponent (max a.2.2 b.2.2)-38
  let align := fun (v : Int × Int × Int) =>
    if v.2.1 ≥ scale then v.1*2^(v.2.1-scale).toNat
    else v.1/2^(scale-v.2.1).toNat
  return encode (align a+align b) scale

theorem grouped_exponent_bounds (e : Int) :
    e ≤ groupedExponent e ∧ groupedExponent e ≤ e+3 := by
  unfold groupedExponent
  omega

theorem grouped_exponent_period (e : Int) :
    groupedExponent (e+4) = groupedExponent e+4 := by
  unfold groupedExponent
  omega

theorem cancellation_floor :
    interpolate 212 81 248 #[0x6e18c2fd,0,0,0,0,0xea8d1ca5,0,0] = 0xe6ca2bc0 := by
  decide +kernel

theorem scale_two_changes_rounding :
    interpolate 212 81 248 #[0x6e98c2fd,0,0,0,0,0xeb0d1ca5,0,0] = 0xe74a2bbc := by
  decide +kernel

theorem scale_sixteen_restores_rounding :
    interpolate 212 81 248 #[0x7018c2fd,0,0,0,0,0xec8d1ca5,0,0] = 0xe8ca2bc0 := by
  decide +kernel

theorem tiny_positive_resolves_negative_tie :
    interpolate 170 76 64 #[0,0x5f9accac,0,0,0,0,0xf1e04656,0] = 0xef2834c0 := by
  decide +kernel

theorem geometric_zero_nan_excluded :
    interpolate 0 0 0 #[0x3f800000,0x7f800001,0x7f800001,0x7f800001,
      0x7f800001,0x7f800001,0x7f800001,0x7f800001] = 0x3f800000 := by
  decide +kernel

theorem zero_weight_nan_participates :
    interpolate 1 1 1 #[0x3f800000,0x3f800000,0x3f800000,0x3f800000,
      0x3f800000,0x3f800000,0x3f800000,0x7f800001] = 0x7fffffff := by
  decide +kernel

theorem negative_zero_preserved :
    interpolate 1 1 1 (Array.replicate 8 0x80000000) = 0x80000000 := by decide +kernel

#print axioms grouped_exponent_bounds
#print axioms grouped_exponent_period
#print axioms cancellation_floor
#print axioms scale_two_changes_rounding
#print axioms scale_sixteen_restores_rounding
#print axioms tiny_positive_resolves_negative_tie
#print axioms geometric_zero_nan_excluded
#print axioms zero_weight_nan_participates
#print axioms negative_zero_preserved

def word (bytes : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut result := 0
  for j in [:4] do result := result+bytes[offset+j]!.toNat*2^(8*j)
  return result

end GB300Math.Texture3DArithmetic

open GB300Math.Texture3DArithmetic

def main (args : List String) : IO UInt32 := do
  let [input,output] := args | throw (IO.userError "Expected stencils.inputs stencils.outputs")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 ∧ xs.size%44 = 0 ∧ ys.size = xs.size/44*4 do
    throw (IO.userError "Invalid 3D stencil capture sizes")
  let mut bad := 0
  for i in [:xs.size/44] do
    let x := word xs (i*44)
    let y := word xs (i*44+4)
    let z := word xs (i*44+8)
    unless x < 256 ∧ y < 256 ∧ z < 256 do
      throw (IO.userError s!"Noncanonical fraction at {i}")
    let vs := (Array.range 8).map (fun j => word xs (i*44+12+j*4))
    let expected := interpolate x y z vs
    let actual := word ys (i*4)
    if expected ≠ actual then
      if bad < 3 then IO.println s!"Mismatch row={i} model={expected} hardware={actual}"
      bad := bad+1
  IO.println s!"result count={xs.size/44} mismatches={bad}"
  return if bad = 0 then 0 else 1
