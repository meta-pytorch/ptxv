import Std

namespace GB300Math.TextureAnisotropyArithmetic

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

structure PlaneSum where
  value : Int
  scale : Int
  exponent : Int
  deriving Inhabited, Repr, DecidableEq

def planeSum (plane : Array (Nat × Nat)) : PlaneSum := Id.run do
  let active := plane.filter (fun p => p.2>0 && exponent p.1>0)
  if active.isEmpty then return ⟨0,0,-10000⟩
  let top := active.foldl (fun e p => max e (exponent p.1)) 0
  let scale := (top:Int)-154
  let mut total : Int := 0
  for (value,weight) in active do
    let mantissa := value%2^23+2^23
    let shift := (exponent value:Int)-150-scale
    let aligned : Nat := if shift≥0 then mantissa*2^shift.toNat else mantissa/2^(-shift).toNat
    total := total+(if negative value then -(aligned:Int) else aligned)*(weight:Int)
  return ⟨total,scale-8,(top:Int)-127⟩

def align (value fromScale toScale : Int) : Int :=
  if fromScale≥toScale then value*2^(fromScale-toScale).toNat
  else value/2^(toScale-fromScale).toNat

def accumulate (sums : Array PlaneSum) (logarithm : Nat) : Nat := Id.run do
  let mut accumulator : Int := 0
  let mut oldScale : Int := 0
  let mut top : Int := -10000
  for i in [:(sums.size+1)/2] do
    let a := sums[i*2]?.getD ⟨0,0,-10000⟩
    let b := sums[i*2+1]?.getD ⟨0,0,-10000⟩
    top := max top (max a.exponent b.exponent)
    if top != -10000 then
      let scale := groupedExponent top-38+(logarithm:Int)
      accumulator := align accumulator oldScale scale+align a.value a.scale scale+align b.value b.scale scale
      oldScale := scale
  return encode accumulator (oldScale-(logarithm:Int))

def interpolate (planes : Array (Array (Nat × Nat))) (groups : Nat) (reverse : Bool) : Nat := Id.run do
  let planes := if reverse then planes.reverse else planes
  let geometric := planes.flatMap id
  if geometric.any (fun p => magnitude p.1>0x7f800000) then return 0x7fffffff
  let infs := geometric.filter (fun p => magnitude p.1=0x7f800000)
  if !infs.isEmpty then
    let sign := negative infs[0]!.1
    if infs.any (fun p => negative p.1 != sign) then return 0x7fffffff
    return (if sign then 2^31 else 0)+0x7f800000
  let sums := planes.map planeSum
  if sums.all (fun p => p.exponent = -10000) then
    return if geometric.all (fun p => negative p.1 && exponent p.1=0) then 2^31 else 0
  return accumulate sums groups.log2

theorem grouped_exponent_bounds (e : Int) :
    e ≤ groupedExponent e ∧ groupedExponent e ≤ e+3 := by
  unfold groupedExponent
  omega

theorem grouped_exponent_period (e : Int) :
    groupedExponent (e+4) = groupedExponent e+4 := by
  unfold groupedExponent
  omega

theorem zero_encode (scale : Int) : encode 0 scale=0 := by simp [encode]

theorem input_subnormal_flush :
    interpolate #[#[(0x007fffff,256)]] 1 false=0 := by decide +kernel

theorem negative_zero_preserved :
    interpolate #[#[(0x80000000,128),(0x807fffff,128)]] 1 false=0x80000000 := by decide +kernel

theorem zero_weight_nan_participates :
    interpolate #[#[(0x3f800000,256),(0x7f800001,0)]] 1 false=0x7fffffff := by decide +kernel

theorem opposite_infinities :
    interpolate #[#[(0x7f800000,256)],#[(0xff800000,256)]] 2 false=0x7fffffff := by decide +kernel

#print axioms grouped_exponent_bounds
#print axioms grouped_exponent_period
#print axioms zero_encode
#print axioms input_subnormal_flush
#print axioms negative_zero_preserved
#print axioms zero_weight_nan_participates
#print axioms opposite_infinities

end GB300Math.TextureAnisotropyArithmetic
