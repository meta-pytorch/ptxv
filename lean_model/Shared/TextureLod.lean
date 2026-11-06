import Std

namespace GB300Math.TextureLod

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


def q8 (v : Nat) : Int :=
  let e : Nat := v/2^23%256
  if e = 0 ∨ (e = 255 ∧ v%2^23 ≠ 0) then 0 else
  let sign : Int := if v ≥ 2^31 then -1 else 1
  if e = 255 then sign*2^40 else
  let m : Nat := v%2^23+2^23
  let shift := (e : Int)-142
  sign*(if shift ≥ 0 then m*2^shift.toNat else m/2^(-shift).toNat : Nat)

def levelPosition (lod bias low high : Nat) : Nat :=
  (min (q8 high) (max (q8 low) (q8 lod+q8 bias))).toNat

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

def sample (values : Array Nat) (depth lod bias low high spatial mip : Nat) : Nat := Id.run do
  let q := levelPosition lod bias low high
  if mip = 0 then
    let v := values[min (values.size-1) ((q+128)/256)]!
    if spatial = 0 then return v
    return interpolate 0 0 0 #[v,0,0,0,0,0,0,0]
  let a := values[min (values.size-1) (q/256)]!
  let b := values[min (values.size-1) (q/256+1)]!
  let split := fun level weight =>
    if spatial = 1 ∧ depth > 0 ∧ max 1 (depth/2^level)%2 = 0 then
      #[weight/2,weight-weight/2]
    else #[weight,0]
  return blendPlanes #[a,a,b,b] (split (q/256) (256-q%256) ++ split (q/256+1) (q%256))

theorem fraction_truncates : q8 0x3b000000 = 0 := by decide +kernel
theorem negative_fraction_truncates : q8 0xbb000000 = 0 := by decide +kernel
theorem bias_after_quantization : levelPosition 0xbb000000 0x3e800000 0 0x40800000 = 64 := by decide +kernel
theorem nan_lod_zero : q8 0xffffffff = 0 := by decide +kernel
theorem point_half_rounds_up : sample #[0x3f800000,0x40000000] 0 0x3f000000 0 0 0x3f800000 0 0 = 0x40000000 := by decide +kernel
theorem mip_half_interpolates : sample #[0x3f800000,0x40000000] 0 0x3f000000 0 0 0x3f800000 0 1 = 0x3fc00000 := by decide +kernel
theorem mip_flushes_subnormal : sample #[1,1] 0 0 0 0 0x3f800000 0 1 = 0 := by decide +kernel
theorem point_preserves_payload : sample #[0x7f800123] 0 0 0 0 0 0 0 = 0x7f800123 := by decide +kernel
theorem mip_canonicalizes_payload : sample #[0x7f800123] 0 0 0 0 0 0 1 = 0x7fffffff := by decide +kernel

theorem spatial_depth_changes_rounding :
    sample #[0x89905229,0x82d19bf5,0x58305716,0xaba8fa30] 4 0x3f4e0000 0 0 0x40400000 1 1 = 2288096630 ∧
    sample #[0x89905229,0x82d19bf5,0x58305716,0xaba8fa30] 0 0x3f4e0000 0 0 0x40400000 1 1 = 2288096629 := by decide +kernel

#print axioms spatial_depth_changes_rounding

theorem quantization_remainder_bound (m d : Nat) (hd : 0 < d) :
    m*256-(m*256/d)*d < d := by
  have h := Nat.mod_lt (m*256) hd
  simpa [Nat.mod_def, Nat.mul_comm] using h

#print axioms fraction_truncates
#print axioms negative_fraction_truncates
#print axioms bias_after_quantization
#print axioms nan_lod_zero
#print axioms point_half_rounds_up
#print axioms mip_half_interpolates
#print axioms mip_flushes_subnormal
#print axioms point_preserves_payload
#print axioms mip_canonicalizes_payload
#print axioms quantization_remainder_bound

def word (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut n := 0
  for j in [:4] do n := n+data[offset+j]!.toNat*2^(8*j)
  return n

end GB300Math.TextureLod
open GB300Math.TextureLod

def main (args : List String) : IO UInt32 := do
  let [ds,ss,ms,bs,ls,hs,valuesFile,input,output] := args |
    throw (IO.userError "Expected depth spatial mip bias low high values inputs outputs")
  let some depth := ds.toNat? | throw (IO.userError "Invalid depth")
  let some spatial := ss.toNat? | throw (IO.userError "Invalid spatial mode")
  let some mip := ms.toNat? | throw (IO.userError "Invalid mip mode")
  let some bias := bs.toNat? | throw (IO.userError "Invalid bias")
  let some low := ls.toNat? | throw (IO.userError "Invalid low clamp")
  let some high := hs.toNat? | throw (IO.userError "Invalid high clamp")
  let raw ← IO.FS.readBinFile valuesFile
  let levels := raw.size/4
  unless depth ≤ 64 ∧ spatial < 2 ∧ mip < 2 ∧ 1 ≤ levels ∧ levels ≤ 8 ∧ raw.size%4 = 0 ∧
      bias < 2^32 ∧ low < 2^32 ∧ high < 2^32 ∧
      exponent bias < 255 ∧ exponent low < 255 ∧ exponent high < 255 ∧
      -1024 ≤ q8 bias ∧ q8 bias ≤ 1024 ∧ 0 ≤ q8 low ∧ q8 low ≤ q8 high ∧
      q8 high ≤ ((levels-1)*256 : Nat) do
    throw (IO.userError "Unsupported configuration")
  let values := (Array.range levels).map (fun j => word raw (4*j))
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 ∧ xs.size%16 = 0 ∧ ys.size = xs.size/4 do
    throw (IO.userError "Invalid capture sizes")
  let mut bad := 0
  for i in [:xs.size/16] do
    unless word xs (i*16) = 0x3f000000 ∧ word xs (i*16+4) = 0x3f000000 do
      throw (IO.userError "Expected central coordinates for constant mip planes")
    let expected := sample values depth (word xs (i*16+8)) bias low high spatial mip
    let actual := word ys (i*4)
    if expected ≠ actual then
      if bad < 3 then IO.println s!"Mismatch row={i} model={expected} hardware={actual}"
      bad := bad+1
  IO.println s!"result count={xs.size/16} mismatches={bad}"
  return if bad = 0 then 0 else 1
