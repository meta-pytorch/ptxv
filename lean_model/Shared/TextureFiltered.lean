import Std

namespace GB300Math.TextureFiltered

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

def encodeHalf (value scale : Int) : Nat := Id.run do
  if value = 0 then return 0
  let sign := if value < 0 then 2^15 else 0
  let n := value.natAbs
  let shift := max ((n.log2 : Int)-10) (-24-scale)
  let rounded := if shift > 0 then
      let divisor := 2^shift.toNat
      n/divisor + if n%divisor ≥ divisor/2 then 1 else 0
    else n*2^(-shift).toNat
  let e := scale+shift
  if rounded = 0 then return sign
  let (m,e) := if rounded ≥ 2^11 then (rounded/2,e+1) else (rounded,e)
  let biased := (m.log2 : Int)+e+15
  if biased ≥ 31 then return sign+0x7c00
  if biased ≤ 0 then return sign+m
  return sign+biased.toNat*2^10+m%2^10

def halfExponent (v : Nat) : Nat := max 1 (v/2^10%32)

def halfPlane (indices values ws : Array Nat) : Int × Int × Int := Id.run do
  if indices.isEmpty then return (0,0,-10000)
  let top := indices.foldl (fun e i => max e (halfExponent values[i]!)) 0
  let scale := (top : Int)-29
  let mut total : Int := 0
  for i in indices do
    let v := values[i]!
    let mantissa := v%2^10 + if v/2^10%32 = 0 then 0 else 2^10
    let shift := (halfExponent v : Int)-25-scale
    let aligned : Nat := if shift ≥ 0 then mantissa*2^shift.toNat else mantissa/2^(-shift).toNat
    let signed : Int := if v ≥ 2^15 then -(aligned : Int) else aligned
    total := total+signed*(ws[i]! : Int)
  return (total,scale-8,(top : Int)-15)

def halfFilter (x y z : Nat) (vs : Array Nat) : Nat := Id.run do
  let ws := weights x y z
  let geometric := (Array.range 8).filter (support x y z)
  if geometric.any (fun i => vs[i]!%2^15 > 0x7c00) then return 0x7fffe000
  let infs := geometric.filter (fun i => vs[i]!%2^15 = 0x7c00)
  if !infs.isEmpty then
    let value := vs[infs[0]!]!
    if infs.any (fun i => vs[i]! ≠ value) then return 0x7fffe000
    return halfToFloat value
  let finite := geometric.filter (fun i => ws[i]! > 0 && vs[i]!%2^15 > 0)
  if finite.isEmpty then
    return if geometric.all (fun i => vs[i]! = 0x8000) then 2^31 else 0
  let a := halfPlane (finite.filter (· < 4)) vs ws
  let b := halfPlane (finite.filter (· ≥ 4)) vs ws
  let scale := (max a.2.2 b.2.2+3)/4*4-25
  let align := fun (v : Int × Int × Int) =>
    if v.2.1 ≥ scale then v.1*2^(v.2.1-scale).toNat
    else v.1/2^(scale-v.2.1).toNat
  return halfToFloat (encodeHalf (align a+align b) scale)

def snorm8Shift (s : Int) : Int := s+(s+64+16*(s/4096))/128

def evaluate (format x y z : Nat) (vs : Array Nat) : Nat := Id.run do
  if format = 4 then return halfFilter x y z vs
  let ws := weights x y z
  let mut total : Int := 0
  for i in [:8] do
    let v := if format = 0 then (vs[i]!*257 : Int)
      else if format = 1 then vs[i]! else signedValue (if format = 2 then 8 else 16) vs[i]!
    total := total+v*(ws[i]! : Int)
  if format = 2 then return signedNormalized (snorm8Shift total) 32767
  let rounded := (total+128)/256
  return if format = 3 then signedNormalized rounded 32767
    else ratioBits rounded.toNat 65535

theorem snorm8_shift_rounding : evaluate 2 113 0 0 #[0,127,0,0,0,0,0,0] = 0x3ee1fdc4 := by decide +kernel

theorem snorm8_center_shift : evaluate 2 0 0 0 #[64,0,0,0,0,0,0,0] = 0x3f010302 := by decide +kernel

theorem snorm8_integer_error (s : Int) (low : -32768 ≤ s) (high : s ≤ 32512) :
    -20480 ≤ 32512*snorm8Shift s-32767*s ∧ 32512*snorm8Shift s-32767*s ≤ 16640 := by
  unfold snorm8Shift
  omega

theorem fixed_rounding_error (n : Int) :
    -128 ≤ 256*((n+128)/256)-n ∧ 256*((n+128)/256)-n ≤ 128 := by
  omega

theorem unorm16_halfway_up : evaluate 1 128 0 0 #[0,1,0,0,0,0,0,0] = 0x37800080 := by decide +kernel

theorem snorm16_negative_halfway_up : evaluate 3 128 0 0 #[0,0xffff,0,0,0,0,0,0] = 0 := by decide +kernel

theorem snorm16_clamps_after_filter : evaluate 3 128 0 0 #[0x8000,0x7fff,0,0,0,0,0,0] = 0 := by decide +kernel

theorem half_subnormal_rounds_away : evaluate 4 128 0 0 #[0,1,0,0,0,0,0,0] = 0x33800000 := by decide +kernel

theorem half_negative_subnormal_rounds_away : evaluate 4 128 0 0 #[0,0x8001,0,0,0,0,0,0] = 0xb3800000 := by decide +kernel

theorem half_exponent_group_matters : evaluate 4 200 27 126 #[43277,54870,53295,19313,26683,16356,62421,45411] = 0xc13ba000 := by decide +kernel

theorem half_geometric_nan_support : evaluate 4 1 1 1 #[0,0,0,0,0,0,0,0x7c01] = 0x7fffe000 := by decide +kernel

theorem half_zero_geometric_nan_excluded : evaluate 4 0 0 0 #[0x3c00,0x7c01,0x7c01,0x7c01,0x7c01,0x7c01,0x7c01,0x7c01] = 0x3f800000 := by decide +kernel

#print axioms snorm8_shift_rounding
#print axioms snorm8_center_shift
#print axioms snorm8_integer_error
#print axioms fixed_rounding_error
#print axioms unorm16_halfway_up
#print axioms snorm16_negative_halfway_up
#print axioms snorm16_clamps_after_filter
#print axioms half_subnormal_rounds_away
#print axioms half_negative_subnormal_rounds_away
#print axioms half_exponent_group_matters
#print axioms half_geometric_nan_support
#print axioms half_zero_geometric_nan_excluded

def word (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut n := 0
  for j in [:4] do n := n+data[offset+j]!.toNat*2^(8*j)
  return n

end GB300Math.TextureFiltered

open GB300Math.TextureFiltered

def main (args : List String) : IO UInt32 := do
  let [formatText,input,output] := args | throw (IO.userError "Expected format inputs outputs")
  let some format := formatText.toNat? | throw (IO.userError "Invalid format")
  unless format < 5 do
    throw (IO.userError "Unsupported format")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 ∧ xs.size%44 = 0 ∧ ys.size = xs.size/44*4 do
    throw (IO.userError "Invalid stencil capture sizes")
  let mut bad := 0
  for i in [:xs.size/44] do
    let x := word xs (i*44)
    let y := word xs (i*44+4)
    let z := word xs (i*44+8)
    let vs := (Array.range 8).map (fun j => word xs (i*44+12+4*j))
    unless x < 256 ∧ y < 256 ∧ z < 256 ∧ vs.all (· < if format = 0 ∨ format = 2 then 256 else 65536) do
      throw (IO.userError "Noncanonical fraction or input width")
    let expected := evaluate format x y z vs
    let actual := word ys (i*4)
    if expected ≠ actual then
      if bad < 3 then IO.println s!"Mismatch row={i} model={expected} hardware={actual}"
      bad := bad+1
  IO.println s!"result count={xs.size/44} mismatches={bad}"
  return if bad = 0 then 0 else 1
