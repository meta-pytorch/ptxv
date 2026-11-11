/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureAnisotropyOrientedMipmap
import GB300Math.TextureAnisotropyRankTwo

namespace GB300Math.TextureAnisotropyGeneralMipmap
open TextureGradient2D

def position (ax ay bx byy anisotropy : Nat) : Nat :=
  let ax := fromBits ax
  let ay := fromBits ay
  let bx := fromBits bx
  let byy := fromBits byy
  let major := norm ax ay bx byy
  if major.significand=0 then 0 else
    let lm := TextureAnisotropyArea.logarithm major.significand.natAbs major.scale
    let area := guardedAdd (TextureAnisotropyGeneral.multiply ax byy)
      (negate (TextureAnisotropyGeneral.multiply ay bx))
    let lower := lm-(TextureAnisotropyMipmap.cap anisotropy:Int)
    let minor := if area.significand=0 then lower else
      TextureAnisotropyArea.logarithm area.significand.natAbs area.scale-lm
    min 2048 (2048+max (min lm minor) lower).toNat

def levels (ax ay bx byy anisotropy : Nat) (optimized : Bool) :
    List TextureAnisotropyOrientedMipmap.Level :=
  let selected := TextureAnisotropyRankTwo.selection ax ay bx byy
  let q := selected.logarithm
  let x := selected.x
  let y := selected.y
  let lod := position ax ay bx byy anisotropy
  if lod=0 then
    let (groups,taps) := TextureAnisotropyOriented.taps q x y
    [⟨0,groups,taps⟩]
  else
    let ratio := q-lod
    let level := lod/256
    let fraction := if optimized then
      (max 0 (min 256 (TextureAnisotropyBlend.fraction (lod%256)
        (2^(TextureAnisotropyMipmap.cap anisotropy/256))))).toNat
      else lod%256
    let step := TextureAnisotropyMipmapStep.step fraction
    let (groups,base) := if ratio<256 then
      let (g,taps) := TextureAnisotropyOriented.taps ratio x y
      let positive := taps[taps.length-1]!
      let sx := TextureAnisotropyOrientedMipmap.signedStep (positive.x/2) step
      let sy := TextureAnisotropyOrientedMipmap.signedStep (positive.y/2) step
      (g,if ratio=0 then [⟨0,0,256⟩] else [⟨-sx,-sy,256⟩,⟨sx,sy,256⟩])
      else TextureAnisotropyOriented.taps ratio (x*(step:Int)/64) (y*(step:Int)/64)
    let weights := TextureAnisotropyMipmapWeights.scaleWeights (256-fraction)
      (base.map TextureAnisotropyOriented.Tap.weight)
    let lower := base.zipWith (fun tap weight => {tap with weight:=weight}) weights
    let (upperGroups,upper) := TextureAnisotropyOriented.taps (ratio+fraction-256) x y
    let weights := TextureAnisotropyMipmapWeights.scaleWeights fraction
      (upper.map TextureAnisotropyOriented.Tap.weight)
    let upper := upper.zipWith (fun tap weight => {tap with weight:=weight}) weights
    [⟨level,groups,lower⟩,⟨level+1,upperGroups,upper⟩]

def evaluate (ax ay bx byy anisotropy : Nat) (optimized : Bool)
    (x y : Int) (images : Array (Array Nat)) : Nat := Id.run do
  let mut planes : Array (Array (Nat × Nat)) := #[]
  for level in levels ax ay bx byy anisotropy optimized do
    let size := 256/2^level.index
    let qx := TextureAnisotropyMipmap.coordinateAtLevel x level.index
    let qy := TextureAnisotropyMipmap.coordinateAtLevel y level.index
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

end GB300Math.TextureAnisotropyGeneralMipmap
