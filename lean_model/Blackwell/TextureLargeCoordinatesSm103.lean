/- Empirical SM103 height-one texture model candidate. -/
import Std

namespace GB300Math.TextureLargeCoordinatesSm103

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


def decoded (v : Nat) : Int × Nat :=
  let e : Nat := v/2^23%256
  if e = 0 ∨ e = 255 then (0,1) else
  let m : Int := if v ≥ 2^31 then -((v%2^23+2^23 : Nat) : Int) else ((v%2^23+2^23 : Nat) : Int)
  let shift := (e : Int)-150
  if shift ≥ 0 then (m*2^shift.toNat,1) else (m,2^(-shift).toNat)

def precision (size : Nat) : Nat :=
  if size ≤ 8192 then 21 else if size ≤ 65536 then 22 else 23

def effective (mode : Nat) (normalized : Bool) : Nat :=
  if !normalized && (mode = 0 || mode = 2) then 1 else mode

def coordinates (v size mode : Nat) (normalized : Bool) : Int × Nat := Id.run do
  let mode := effective mode normalized
  if v%2^31 = 0x7f800000 then
    return if mode = 1 ∨ mode = 3 then
      ((if v < 2^31 then 1 else -1)*2^40,1) else (0,1)
  let (m,d) := decoded v
  let p := precision size
  let (m,d) := if normalized then ((m*2^p/d)*(size : Int),2^p) else (m,d)
  if normalized && (mode = 0 || mode = 2) then
    return (m%((size*d*(if mode = 2 then 2 else 1) : Nat) : Int),d)
  return (m,d)

def index (i : Int) (size mode : Nat) : Option Nat := Id.run do
  if mode = 0 then return some (i%(size : Int)).toNat
  if mode = 1 then return some (min (size-1) i.toNat)
  if mode = 2 then
    let r := (i%((2*size) : Int)).toNat
    return some (if r < size then r else 2*size-1-r)
  if i < 0 ∨ i ≥ size then return none
  return some i.toNat

def point (v size mode : Nat) (normalized : Bool) : Option Nat :=
  let (m,d) := coordinates v size mode normalized
  index (m/d) size (effective mode normalized)

def position (v size mode : Nat) (normalized : Bool) : Int :=
  let (m,d) := coordinates v size mode normalized
  let raw := (2*m*256+(d : Int))/(2*d)-128
  let mode := effective mode normalized
  if mode = 1 then max 0 (min (((size-1)*256 : Nat) : Int) raw)
  else if normalized && mode = 2 then max 0 (min (((2*size-1)*256 : Nat) : Int) raw)
  else raw

def texel (values : Array Nat) (i : Option Nat) : Nat :=
  match i with
  | some j => values[j]!
  | none => 0

def evaluate (values : Array Nat) (v mode : Nat) (normalized linear : Bool) : Nat := Id.run do
  let size := values.size
  if !linear then return texel values (point v size mode normalized)
  let p := position v size mode normalized
  let mode := effective mode normalized
  let a := texel values (index (p/256) size mode)
  let b := texel values (index (p/256+1) size mode)
  return interpolate (p%256).toNat 0 #[a,b,0,0]

theorem last_correct_mirror_width : point 0x80800000 65536 2 true = some 0 := by decide +kernel
theorem mirror_width_65537 : point 0x80800000 65537 2 true = some 0 := by decide +kernel
theorem reflected_midpoint : point 0x3fc00000 131071 2 true = some 65535 := by decide +kernel
theorem last_correct_border_width : point 0x3f800000 131071 3 true = none := by decide +kernel
theorem border_at_max_width : point 0x3f800000 131072 3 true = none := by decide +kernel
theorem wrap_high_corner : index 131072 131072 0 = some 0 := by decide +kernel
theorem subnormal_coordinate_flush : point 0x807fffff 65537 2 true = some 0 := by decide +kernel
theorem power_two_grid_transitions : precision 8192 = 21 ∧ precision 8193 = 22 ∧
    precision 65536 = 22 ∧ precision 65537 = 23 := by decide +kernel
#print axioms last_correct_mirror_width
#print axioms mirror_width_65537
#print axioms reflected_midpoint
#print axioms last_correct_border_width
#print axioms border_at_max_width
#print axioms wrap_high_corner
#print axioms subnormal_coordinate_flush
#print axioms power_two_grid_transitions

def word (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut n := 0
  for j in [:4] do n := n+data[offset+j]!.toNat*2^(8*j)
  return n
end GB300Math.TextureLargeCoordinatesSm103
open GB300Math.TextureLargeCoordinatesSm103

def main (args : List String) : IO UInt32 := do
  let [mText,nText,fText,texels,input,output] := args |
    throw (IO.userError "Expected mode normalized linear texels inputs outputs")
  let some m := mText.toNat? | throw (IO.userError "Invalid mode")
  let some n := nText.toNat? | throw (IO.userError "Invalid normalized flag")
  let some f := fText.toNat? | throw (IO.userError "Invalid filter")
  let ts ← IO.FS.readBinFile texels
  unless ts.size > 0 ∧ ts.size%4 = 0 ∧ ts.size/4 ≤ 131072 ∧ m < 4 ∧ n < 2 ∧ f < 2 do
    throw (IO.userError "Unsupported configuration")
  let values := (Array.range (ts.size/4)).map (fun j => word ts (4*j))
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 ∧ xs.size%16 = 0 ∧ ys.size = xs.size/4 do
    throw (IO.userError "Invalid capture sizes")
  let mut bad := 0
  for i in [:xs.size/16] do
    unless word xs (i*16+4) = 0x3f000000 do
      throw (IO.userError "Expected y=0.5 for height-one texture")
    let expected := evaluate values (word xs (i*16)) m (n = 1) (f = 1)
    let actual := word ys (i*4)
    if expected ≠ actual then
      if bad < 3 then IO.println s!"Mismatch row={i} model={expected} hardware={actual}"
      bad := bad+1
  IO.println s!"result count={xs.size/16} mismatches={bad}"
  return if bad = 0 then 0 else 1
