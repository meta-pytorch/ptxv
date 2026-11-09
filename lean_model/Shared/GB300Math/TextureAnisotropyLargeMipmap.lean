import GB300Math.TextureAnisotropyGeneralMipmap
import GB300Math.TextureAnisotropyMipmapCoordinates

namespace GB300Math.TextureAnisotropyLargeMipmap

def evaluate (ax ay bx byy anisotropy : Nat) (optimized : Bool)
    (x y : Nat) (images : Array (Array Nat)) : Nat := Id.run do
  let mut planes : Array (Array (Nat × Nat)) := #[]
  for level in TextureAnisotropyGeneralMipmap.levels ax ay bx byy anisotropy optimized do
    let size := 256/2^level.index
    let qx := TextureAnisotropyMipmapCoordinates.coordinateBits x (8-level.index) true
    let qy := TextureAnisotropyMipmapCoordinates.coordinateBits y (8-level.index) true
    for tap in level.taps do
      let px := max 0 (min (((size-1)*256:Nat):Int) (qx+tap.x))
      let py := max 0 (min (((size-1)*256:Nat):Int) (qy+tap.y))
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

theorem clamped_coordinate_bounds (last : Nat) (coordinate : Int) :
    0 ≤ max 0 (min (last:Int) coordinate) ∧
    max 0 (min (last:Int) coordinate) ≤ last := by omega

theorem interior_coordinate_unchanged (last : Nat) (coordinate : Int)
    (hlo : 0 ≤ coordinate) (hhi : coordinate ≤ last) :
    max 0 (min (last:Int) coordinate) = coordinate := by omega

theorem clamp_idempotent (last : Nat) (coordinate : Int) :
    max 0 (min (last:Int) (max 0 (min (last:Int) coordinate))) =
    max 0 (min (last:Int) coordinate) := by omega

end GB300Math.TextureAnisotropyLargeMipmap
