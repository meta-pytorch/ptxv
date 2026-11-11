import GB300Math.TextureAnisotropyGeneralMipmap
import GB300Math.TextureAnisotropyMipmapCoordinates

namespace GB300Math.TextureAnisotropyJointMipmap

def evaluate (ax ay bx byy anisotropy : Nat) (optimized : Bool)
    (x y : Nat) (images : Array (Array Nat)) : Nat := Id.run do
  let mut planes : Array (Array (Nat × Nat)) := #[]
  for level in TextureAnisotropyGeneralMipmap.levels ax ay bx byy anisotropy optimized do
    let size := 256/2^level.index
    let qx := TextureAnisotropyMipmapCoordinates.coordinateBits x (8-level.index) true
    let qy := TextureAnisotropyMipmapCoordinates.coordinateBits y (8-level.index) true
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

end GB300Math.TextureAnisotropyJointMipmap
