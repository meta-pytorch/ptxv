/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.TextureGradient

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


def blendPlanes (vs ws : Array Nat) : Nat := Id.run do
  let active := (Array.range vs.size).filter (fun i => ws[i]! > 0)
  if active.any (fun i => magnitude vs[i]! > 0x7f800000) then return 0x7fffffff
  let infs := active.filter (fun i => magnitude vs[i]! = 0x7f800000)
  if !infs.isEmpty then
    let sign := negative vs[infs[0]!]!
    if infs.any (fun i => negative vs[i]! != sign) then return 0x7fffffff
    return (if sign then 2^31 else 0)+0x7f800000
  let finite := active.filter (fun i => exponent vs[i]! > 0)
  if finite.isEmpty then return if active.all (fun i => negative vs[i]!) then 2^31 else 0
  let top := finite.foldl (fun e i => max e (exponent vs[i]!)) 0
  let scale := groupedExponent ((top : Int)-127)-38
  let mut total : Int := 0
  for i in finite do
    let v := vs[i]!
    let mantissa : Int := (v%2^23+2^23)*ws[i]!
    let signed := if negative v then -mantissa else mantissa
    let shift := (exponent v : Int)-158-scale
    total := total + if shift ≥ 0 then signed*2^shift.toNat else signed/2^(-shift).toNat
  return encode total scale

-- Measured TXD fractional LOD table; provenance is pinned in the capture package.
def logTable : Array Nat := #[0,2,4,5,6,8,9,11,12,13,15,16,18,19,20,22,23,24,26,27,28,30,31,32,34,35,36,38,39,40,42,43,44,45,47,48,49,50,52,53,54,55,57,58,59,60,62,63,64,65,66,68,69,70,71,72,74,75,76,77,78,79,81,82,83,84,85,86,88,89,90,91,92,93,94,95,97,98,99,100,101,102,103,104,105,106,107,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133,135,136,137,137,138,139,140,141,142,143,144,145,146,147,148,149,150,151,152,153,154,155,156,157,158,159,160,161,162,162,163,164,165,166,167,168,169,170,171,172,173,173,174,175,176,177,178,179,180,181,181,182,183,184,185,186,187,188,188,189,190,191,192,193,194,195,195,196,197,198,199,200,200,201,202,203,204,205,205,206,207,208,209,210,210,211,212,213,214,214,215,216,217,218,218,219,220,221,222,222,223,224,225,226,226,227,228,229,229,230,231,232,233,233,234,235,236,236,237,238,239,239,240,241,242,242,243,244,245,245,246,247,248,248,249,250,251,251,252,253,253,254,255,255]

def gradientLog (v dimension : Nat) : Int :=
  if exponent v = 0 then -100000 else
  ((exponent v : Int)-127+(dimension.log2 : Int))*256 + (logTable[(v%2^23)/2^15]! : Int)

def gradientPosition (dx dy width height levels : Nat) : Nat :=
  min ((levels-1)*256) (max (gradientLog dx width) (gradientLog dy height)).toNat

def gradientSample (values : Array Nat) (dx dy width height spatial mip : Nat) : Nat := Id.run do
  let q := gradientPosition dx dy width height values.size
  if mip = 0 then
    let v := values[min (values.size-1) ((q+128)/256)]!
    if spatial = 0 then return v
    return interpolate 0 0 0 #[v,0,0,0,0,0,0,0]
  let a := values[min (values.size-1) (q/256)]!
  let b := values[min (values.size-1) (q/256+1)]!
  return blendPlanes #[a,b] #[256-q%256,q%256]

theorem log_table_size : logTable.size = 256 := by decide +kernel
theorem log_table_range : ∀ i ∈ List.range 256, logTable[i]! ≤ 255 := by decide +kernel
theorem log_table_monotone : ∀ i ∈ List.range 255, logTable[i]! ≤ logTable[i+1]! := by decide +kernel
theorem smallest_gradient_clamps : gradientPosition 1 0 256 256 9 = 0 := by decide +kernel
theorem negative_gradient_uses_magnitude : gradientLog 0xbf808000 256 = gradientLog 0x3f808000 256 := by decide +kernel
theorem table_low_bits_discarded : gradientLog 0x3b800001 256 = gradientLog 0x3b807fff 256 := by decide +kernel
theorem table_first_step : gradientPosition 0x3b808000 0 256 256 9 = 2 := by decide +kernel
theorem power_two_gradient_scale : gradientLog 0x3f800000 256 = gradientLog 0x40000000 128 := by decide +kernel
#print axioms log_table_size
#print axioms log_table_range
#print axioms log_table_monotone
#print axioms smallest_gradient_clamps
#print axioms negative_gradient_uses_magnitude
#print axioms table_low_bits_discarded
#print axioms table_first_step
#print axioms power_two_gradient_scale

def word (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut n := 0
  for j in [:4] do n := n+data[offset+j]!.toNat*2^(8*j)
  return n

end GB300Math.TextureGradient
open GB300Math.TextureGradient

def main (args : List String) : IO UInt32 := do
  let [ws,hs,ss,ms,valuesFile,input,output] := args |
    throw (IO.userError "Expected width height spatial mip values inputs outputs")
  let some width := ws.toNat? | throw (IO.userError "Invalid width")
  let some height := hs.toNat? | throw (IO.userError "Invalid height")
  let some spatial := ss.toNat? | throw (IO.userError "Invalid spatial mode")
  let some mip := ms.toNat? | throw (IO.userError "Invalid mip mode")
  let raw ← IO.FS.readBinFile valuesFile
  let levels := raw.size/4
  unless 1 ≤ width ∧ width ≤ 1024 ∧ width = 2^width.log2 ∧
      1 ≤ height ∧ height ≤ 1024 ∧ height = 2^height.log2 ∧
      spatial < 2 ∧ mip < 2 ∧ levels = (max width height).log2+1 ∧ raw.size%4 = 0 do
    throw (IO.userError "Unsupported configuration")
  let values := (Array.range levels).map (fun j => word raw (4*j))
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 ∧ xs.size%16 = 0 ∧ ys.size = xs.size/4 do
    throw (IO.userError "Invalid capture sizes")
  let mut bad := 0
  for i in [:xs.size/16] do
    let dx := word xs (i*16+8)
    let dy := word xs (i*16+12)
    unless word xs (i*16) = 0x3f000000 ∧ word xs (i*16+4) = 0x3f000000 ∧
        exponent dx < 255 ∧ exponent dy < 255 do
      throw (IO.userError "Unsupported coordinate or gradient")
    let expected := gradientSample values dx dy width height spatial mip
    let actual := word ys (i*4)
    if expected ≠ actual then
      if bad < 3 then IO.println s!"Mismatch row={i} model={expected} hardware={actual}"
      bad := bad+1
  IO.println s!"result count={xs.size/16} mismatches={bad}"
  return if bad = 0 then 0 else 1
