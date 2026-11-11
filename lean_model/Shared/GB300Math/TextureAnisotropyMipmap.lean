import GB300Math.TextureAnisotropyMipmapStep
import GB300Math.TextureAnisotropyBlend
import GB300Math.TextureAnisotropyOriented

namespace GB300Math.TextureAnisotropyMipmap

open TextureAnisotropyMipmapWeights

structure Level where
  index : Nat
  groups : Nat
  taps : List TextureAnisotropyTaps.Tap
  deriving Inhabited, Repr

def cap (anisotropy : Nat) : Nat :=
  if anisotropy<2 then 0 else if anisotropy<4 then 256 else
  if anisotropy<6 then 512 else if anisotropy<8 then 656 else
  if anisotropy<10 then 768 else if anisotropy<12 then 848 else
  if anisotropy<16 then 912 else 1024

def levels (gradient anisotropy : Nat) (negative optimized : Bool) : List Level :=
  let q := TextureAnisotropyTaps.logarithm gradient
  let limit := cap anisotropy
  if q<limit then
    let (groups,taps) := TextureAnisotropyTaps.taps q negative
    let taps := taps.map fun tap => if negative then {tap with offset := -tap.offset} else tap
    [⟨0,groups,taps⟩]
  else
  let lod := q-limit
  let level := lod/256
  let fraction := if optimized then
    (max 0 (min 256 (TextureAnisotropyBlend.fraction (lod%256) (2^(limit/256))))).toNat
    else lod%256
  let effective := level*256+limit+fraction
  let step := TextureAnisotropyMipmapStep.step fraction
  let (groups,base) := TextureAnisotropyTaps.taps limit false
  let lowerWeights := scaleWeights (256-fraction) (base.map TextureAnisotropyTaps.Tap.weight)
  let lower := base.zipWith (fun tap weight =>
      {tap with offset := tap.offset/128*(step:Int)*2, weight := weight}) lowerWeights
  let (upperGroups,upper) := TextureAnisotropyTaps.taps (effective-(level+1)*256) negative
  let upperWeights := scaleWeights fraction (upper.map TextureAnisotropyTaps.Tap.weight)
  let upper := upper.zipWith (fun tap weight => {tap with weight:=weight}) upperWeights
  let orient := fun (ts : List TextureAnisotropyTaps.Tap) =>
    ts.map fun tap => if negative then {tap with offset := -tap.offset} else tap
  [⟨level,groups,orient lower⟩,⟨level+1,upperGroups,orient upper⟩]

def coordinateAtLevel (q : Int) (level : Nat) : Int :=
  if level=0 then q else (q+128+(2^(level-1):Nat))/2^level-128

def evaluate (gradient anisotropy : Nat) (vertical negative optimized : Bool) (x y : Int)
    (images : Array (Array Int)) : Int := Id.run do
  let mut total := 0
  for level in levels gradient anisotropy negative optimized do
    let size := 256/2^level.index
    let qx := coordinateAtLevel x level.index
    let qy := coordinateAtLevel y level.index
    let mut subtotal := 0
    for tap in level.taps do
      let px := qx+if vertical then 0 else tap.offset
      let py := qy+if vertical then tap.offset else 0
      let ix := px/256
      let iy := py/256
      let weights := TextureAnisotropyOriented.plane tap.weight (px%256).toNat (py%256).toNat
      for (weight,i) in [weights.1,weights.2.1,weights.2.2.1,weights.2.2.2].zipIdx do
        let tx := min (size-1) (ix+(i%2:Nat)).toNat
        let ty := min (size-1) (iy+(i/2:Nat)).toNat
        subtotal := subtotal+(weight:Int)*images[level.index]![ty*size+tx]!
    total := total+subtotal*(4/level.groups:Nat)
  return total

end GB300Math.TextureAnisotropyMipmap
