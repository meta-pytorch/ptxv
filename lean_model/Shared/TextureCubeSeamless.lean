/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.TextureCubeSeamless

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

def interpolate (ws geometric values : Array Nat) : Nat := Id.run do
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
  v.all (fun x => magnitude x ≤ 0x3f800000) && v.any (fun x => magnitude x = 0x3f800000)

def face (v : Array Nat) : Nat :=
  let axis := if magnitude v[2]! = 0x3f800000 then 2 else if magnitude v[1]! = 0x3f800000 then 1 else 0
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

def remap (face size : Nat) (u v : Int) : Nat :=
  let n : Int := size
  let s := 2*u+1-n
  let t := 2*v+1-n
  let xyz : Array Int := match face with
    | 0 => #[n,-t,-s]
    | 1 => #[-n,-t,s]
    | 2 => #[s,n,t]
    | 3 => #[s,-n,-t]
    | 4 => #[s,-t,n]
    | _ => #[-s,-t,-n]
  let top : Nat := xyz.foldl (fun a v => max a v.natAbs) 0
  let axis : Nat := if xyz[2]!.natAbs = top then 2 else if xyz[1]!.natAbs = top then 1 else 0
  let f : Nat := 2*axis + if xyz[axis]! < (0 : Int) then 1 else 0
  let x := xyz[0]!
  let y := xyz[1]!
  let z := xyz[2]!
  let (a,b) := match f with
    | 0 => (-z,-y)
    | 1 => (z,-y)
    | 2 => (x,z)
    | 3 => (x,-z)
    | 4 => (x,-y)
    | _ => (-x,-y)
  let coordinate := fun q => min (size-1) (((q+top)*n)/(2*(top : Int))).toNat
  coordinate a+size*(coordinate b+size*f)

def seamWeights (x y : Nat) (missing center : Nat) : Array Nat := Id.run do
  let mut ws := weights x y
  if missing ≥ 4 then return ws
  let share := (ws[missing]!+1)/3
  let rest := ws[missing]!-2*share
  for j in [:4] do
    ws := ws.set! j (if j = missing then 0 else ws[j]! + if j = center then rest else share)
  return ws

def sample (values : Array Nat) (size : Nat) (v : Array Nat) (linear : Bool) : Nat := Id.run do
  let f := face v
  let (a,b) := projections v
  if !linear then return values[pointCoordinate a size + size*(pointCoordinate b size+size*f)]!
  let x := position a size
  let y := position b size
  let us := (Array.range 4).map (fun i => x/256+(i%2 : Nat))
  let vs := (Array.range 4).map (fun i => y/256+(i/2 : Nat))
  let inside := fun q => 0 ≤ q && q < (size : Int)
  let missing := ((Array.range 4).filter (fun i => !inside us[i]! && !inside vs[i]!))[0]?.getD 4
  let center := ((Array.range 4).filter (fun i => inside us[i]! && inside vs[i]!))[0]?.getD 4
  let ws := seamWeights (x%256).toNat (y%256).toNat missing center
  let geometric := (Array.range 4).filter (fun i => i ≠ missing &&
    (support (x%256).toNat (y%256).toNat i ||
      missing < 4))
  let texels := (Array.range 4).map (fun i => values[remap f size us[i]! vs[i]!]!)
  return interpolate ws geometric texels

theorem unweighted_corner_nan :
    sample ((Array.replicate 24 0x3f800000).set! 18 0x7f800001) 2 #[0xbf800000,0xbf7f8000,0x3f000000] true = 0x7fffffff := by decide +kernel
#print axioms unweighted_corner_nan

theorem exact_corner_weights : seamWeights 128 128 1 2 = #[85,0,86,85] := by decide +kernel
theorem corner_share_rounds_up : seamWeights 244 220 0 3 = #[0,35,11,210] := by decide +kernel
#print axioms exact_corner_weights
#print axioms corner_share_rounds_up

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
end GB300Math.TextureCubeSeamless
open GB300Math.TextureCubeSeamless

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
    unless canonical v do throw (IO.userError "Expected finite canonical cube direction")
    let expected := sample values size v (f = 1)
    let actual := word ys (i*4)
    if expected ≠ actual then
      if bad < 3 then IO.println s!"Mismatch row={i} model={expected} hardware={actual}"
      bad := bad+1
  IO.println s!"result count={xs.size/16} mismatches={bad}"
  return if bad = 0 then 0 else 1
