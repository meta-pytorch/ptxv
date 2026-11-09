import Std

namespace GB300Math.TextureArithmetic

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

theorem ties_away : encode 16777217 (-24) = 0x3f800001 := by decide +kernel
theorem negative_ties_away : encode (-16777217) (-24) = 0xbf800001 := by decide +kernel
theorem input_subnormal_flush : interpolate 0 0 #[1,0,0,0] = 0 := by decide +kernel
theorem negative_zero_preserved :
    interpolate 1 1 #[0x80000000,0x80000000,0x80000000,0x80000000] = 0x80000000 := by decide +kernel
theorem zero_weight_nan_participates :
    interpolate 1 1 #[0x3f800000,0x3f800000,0x3f800000,0x7f800001] = 0x7fffffff := by decide +kernel
theorem geometric_zero_nan_excluded :
    interpolate 0 0 #[0x3f800000,0x7f800001,0x7f800001,0x7f800001] = 0x3f800000 := by decide +kernel
theorem zero_weight_exponent_excluded :
    interpolate 1 1 #[0x3f800000,0x3f800000,0x3f800000,0x7f7fffff] = 0x3f800000 := by decide +kernel
theorem cancellation_zero :
    interpolate 128 0 #[0x3f800000,0xbf800000,0,0] = 0 := by decide +kernel
theorem underflow_rounds_before_flush :
    interpolate 128 0 #[0x00ffffff,0,0,0] = 0x00800000 := by decide +kernel

#print axioms ties_away
#print axioms negative_ties_away
#print axioms input_subnormal_flush
#print axioms negative_zero_preserved
#print axioms zero_weight_nan_participates
#print axioms geometric_zero_nan_excluded
#print axioms zero_weight_exponent_excluded
#print axioms cancellation_zero
#print axioms underflow_rounds_before_flush

def word (bytes : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut result := 0
  for j in [:4] do result := result + bytes[offset+j]!.toNat*2^(8*j)
  return result

end GB300Math.TextureArithmetic

open GB300Math.TextureArithmetic

def main (args : List String) : IO UInt32 := do
  let [input,output] := args | throw (IO.userError "Expected stencils.u32x6 outputs.u32")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 ∧ xs.size%24 = 0 ∧ ys.size = xs.size/24*4 do
    throw (IO.userError "Invalid 2D stencil capture sizes")
  let mut bad := 0
  for i in [:xs.size/24] do
    let x := word xs (i*24)
    let y := word xs (i*24+4)
    unless x < 256 ∧ y < 256 do throw (IO.userError s!"Noncanonical fraction at {i}")
    let vs := (Array.range 4).map (fun j => word xs (i*24+8+j*4))
    let expected := interpolate x y vs
    let actual := word ys (i*4)
    if expected ≠ actual then
      if bad < 3 then IO.println s!"Mismatch row={i} model={expected} hardware={actual}"
      bad := bad+1
  IO.println s!"result count={xs.size/24} mismatches={bad}"
  return if bad = 0 then 0 else 1
