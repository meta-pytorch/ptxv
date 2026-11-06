import GB300Math.TextureAnisotropyMipmap
import GB300Math.TextureAnisotropyArithmetic

namespace GB300Math.TextureAnisotropyMipmapArithmetic

def evaluate (gradient anisotropy : Nat) (vertical negative optimized : Bool)
    (x y : Int) (images : Array (Array Nat)) : Nat := Id.run do
  let mut planes : Array (Array (Nat × Nat)) := #[]
  for level in TextureAnisotropyMipmap.levels gradient anisotropy negative optimized do
    let size := 256/2^level.index
    let qx := TextureAnisotropyMipmap.coordinateAtLevel x level.index
    let qy := TextureAnisotropyMipmap.coordinateAtLevel y level.index
    for tap in level.taps do
      let px := qx+if vertical then 0 else tap.offset
      let py := qy+if vertical then tap.offset else 0
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

end GB300Math.TextureAnisotropyMipmapArithmetic
