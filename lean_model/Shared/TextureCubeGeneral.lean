/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.Mufu
import FloatLib.Floats.Formats.BinaryInterchange.Rounding.Directed.Runtime

namespace GB300Math.TextureCubeGeneral

def weights (x y : Nat) : Array Nat :=
  let ll := ((256-x)*(256-y)+128)/256
  let hh := (x*y+128)/256
  #[ll,x-hh,256-x-ll,hh]

def support (x y i : Nat) : Bool :=
  (if i%2 = 0 then x < 256 else x > 0) &&
  (if i/2 = 0 then y < 256 else y > 0)

def magnitude (v : Nat) : Nat := v % 2^31

def exponent (v : Nat) : Nat := v / 2^23 % 256

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

def interpolate (x y : Nat) (values : Array Nat) : Nat := Id.run do
  let ws := weights x y
  let geometric := (Array.range 4).filter (support x y)
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
  let top := finite.foldl (fun e i => max e (exponent values[i]!)) 0
  let scale := (top : Int)-154
  let mut total : Int := 0
  for i in finite do
    let value := values[i]!
    let mantissa := value%2^23+2^23
    let shift := (exponent value : Int)-150-scale
    let aligned : Nat := if shift ≥ 0 then mantissa*2^shift.toNat else mantissa/2^(-shift).toNat
    let signed : Int := if negative value then -(aligned : Int) else aligned
    total := total + signed*(ws[i]! : Int)
  return encode total (scale-8)


def canonical (v : Array Nat) : Bool :=
  let top := v.foldl (fun a x => max a (magnitude x)) 0
  v.all (fun x => magnitude x < 0x7f800000) && 0x800000 ≤ top && top ≤ 0x7e800000

def normalized (v : Array Nat) : Array Nat :=
  let top := v.foldl (fun a x => max a (magnitude x)) 0
  let factor := GB300Math.Mufu.rcp top
  v.map (fun x =>
    let a : FloatLib.Floats.Formats.BinaryInterchange.Model .binary32 := .ofNatBits x
    let b : FloatLib.Floats.Formats.BinaryInterchange.Model .binary32 := .ofNatBits factor
    FloatLib.Floats.Formats.BinaryInterchange.Model.toNatBits (a.mulWithRounding .nearestEven b))

def face (v : Array Nat) : Nat :=
  let top := v.foldl (fun a x => max a (magnitude x)) 0
  let axis := if magnitude v[2]! = top then 2 else if magnitude v[1]! = top then 1 else 0
  2*axis + if negative v[axis]! then 1 else 0

def projections (v : Array Nat) : Nat × Nat :=
  match face v with
  | 0 => (v[2]! ^^^ 2^31,v[1]! ^^^ 2^31)
  | 1 => (v[2]!,v[1]! ^^^ 2^31)
  | 2 => (v[0]!,v[2]!)
  | 3 => (v[0]!,v[2]! ^^^ 2^31)
  | 4 => (v[0]!,v[1]! ^^^ 2^31)
  | _ => (v[0]! ^^^ 2^31,v[1]! ^^^ 2^31)

def decoded (v : Nat) : Int × Nat :=
  let e : Nat := exponent v
  if e = 0 then (0,1) else
  let m : Int := if negative v then -((v%2^23+2^23 : Nat) : Int) else ((v%2^23+2^23 : Nat) : Int)
  let shift := (e : Int)-150
  if shift ≥ 0 then (m*2^shift.toNat,1) else (m,2^(-shift).toNat)

def pointCoordinate (v size : Nat) : Nat :=
  let (m,d) := decoded v
  min (size-1) ((((m+d)*2^21/(2*d))*(size : Int))/2^21).toNat

def position (v size : Nat) : Int :=
  let (m,d) := decoded v
  let grid := (m+d)*2^21/(2*d)
  (2*grid*(size : Int)*256+2^21)/(2*2^21)-128

def sample (values : Array Nat) (size : Nat) (v : Array Nat) (linear : Bool) : Nat := Id.run do
  let f := face v
  let (a,b) := projections v
  if !linear then return values[pointCoordinate a size + size*(pointCoordinate b size+size*f)]!
  let x := max 0 (min ((size-1)*256 : Int) (position a size))
  let y := max 0 (min ((size-1)*256 : Int) (position b size))
  let vs := (Array.range 4).map (fun i =>
    let u := min (size-1) (x/256+(i%2 : Nat)).toNat
    let w := min (size-1) (y/256+(i/2 : Nat)).toNat
    values[u+size*(w+size*f)]!)
  return interpolate (x%256).toNat (y%256).toNat vs

theorem normalization_can_round_below_one :
    normalized #[1065373686,0,0] = #[1065353215,0,0] := by decide +kernel
#print axioms normalization_can_round_below_one
theorem positive_x : face #[0x3f800000,0,0] = 0 := by decide +kernel
theorem negative_x : face #[0xbf800000,0,0] = 1 := by decide +kernel
theorem positive_y : face #[0,0x3f800000,0] = 2 := by decide +kernel
theorem negative_y : face #[0,0xbf800000,0] = 3 := by decide +kernel
theorem positive_z : face #[0,0,0x3f800000] = 4 := by decide +kernel
theorem negative_z : face #[0,0,0xbf800000] = 5 := by decide +kernel
theorem z_wins_tie : face #[0x3f800000,0x3f800000,0x3f800000] = 4 := by decide +kernel
theorem y_wins_x_tie : face #[0xbf800000,0x3f800000,0] = 2 := by decide +kernel
theorem coordinate_subnormal_flush : pointCoordinate 0x80000001 16 = 8 := by decide +kernel
#print axioms positive_x
#print axioms negative_x
#print axioms positive_y
#print axioms negative_y
#print axioms positive_z
#print axioms negative_z
#print axioms z_wins_tie
#print axioms y_wins_x_tie
#print axioms coordinate_subnormal_flush

def word (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut n := 0
  for j in [:4] do n := n+data[offset+j]!.toNat*2^(8*j)
  return n
end GB300Math.TextureCubeGeneral
open GB300Math.TextureCubeGeneral

def main (args : List String) : IO UInt32 := do
  let [ss,fs,texels,input,output] := args | throw (IO.userError "Expected size linear texels inputs outputs")
  let some size := ss.toNat? | throw (IO.userError "Invalid size")
  let some f := fs.toNat? | throw (IO.userError "Invalid filter")
  unless 1 ≤ size ∧ size ≤ 128 ∧ f < 2 do throw (IO.userError "Unsupported configuration")
  let ts ← IO.FS.readBinFile texels
  unless ts.size = size*size*6*4 do throw (IO.userError "Invalid texture size")
  let values := (Array.range (ts.size/4)).map (fun j => word ts (4*j))
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 ∧ xs.size%16 = 0 ∧ ys.size = xs.size/4 do throw (IO.userError "Invalid capture sizes")
  let mut bad := 0
  for i in [:xs.size/16] do
    let v := (Array.range 3).map (fun j => word xs (i*16+j*4))
    unless canonical v do throw (IO.userError "Expected finite cube direction with nonzero normal reciprocal")
    let expected := sample values size (normalized v) (f = 1)
    let actual := word ys (i*4)
    if expected ≠ actual then
      if bad < 3 then IO.println s!"Mismatch row={i} model={expected} hardware={actual}"
      bad := bad+1
  IO.println s!"result count={xs.size/16} mismatches={bad}"
  return if bad = 0 then 0 else 1
