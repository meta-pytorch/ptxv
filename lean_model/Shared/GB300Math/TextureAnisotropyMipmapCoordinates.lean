import GB300Math.TextureAnisotropyOrientedMipmap

namespace GB300Math.TextureAnisotropyMipmapCoordinates

def coordinateBits (value logSize : Nat) (normalized : Bool) : Int :=
  let e : Nat := value/2^23%256
  if e = 0 ∨ e = 255 then -128 else
  let m : Nat := value%2^23+2^23
  let scale := (e : Int)-142 + if normalized then logSize else 0
  let signed : Int := if value ≥ 2^31 then -(m : Int) else m
  let rounded := if scale ≥ 0 then signed*2^scale.toNat
    else (signed+2^((-scale).toNat-1))/2^(-scale).toNat
  rounded-128


def evaluate (a b anisotropy : Nat) (optimized : Bool)
    (x y : Nat) (images : Array (Array Nat)) : Nat := Id.run do
  let mut planes : Array (Array (Nat × Nat)) := #[]
  for level in TextureAnisotropyOrientedMipmap.levels a b anisotropy optimized do
    let size := 256/2^level.index
    let qx := coordinateBits x (8-level.index) true
    let qy := coordinateBits y (8-level.index) true
    for tap in level.taps do
      let px := qx+tap.x
      let py := qy+tap.y
      let ix := px/256
      let iy := py/256
      let p := TextureAnisotropyOriented.plane tap.weight (px%256).toNat (py%256).toNat
      let weights := #[p.1,p.2.1,p.2.2.1,p.2.2.2]
      let mut plane := #[]
      if tap.weight>0 then
        for i in [:4] do
          if (i%2=0 ∨ px%256≠0) ∧ (i/2=0 ∨ py%256≠0) then
            let tx := min (size-1) (ix+(i%2:Nat)).toNat
            let ty := min (size-1) (iy+(i/2:Nat)).toNat
            plane := plane.push (images[level.index]![ty*size+tx]!,weights[i]!*(4/level.groups))
      planes := planes.push plane
  return TextureAnisotropyArithmetic.interpolate planes 4 false

end GB300Math.TextureAnisotropyMipmapCoordinates
