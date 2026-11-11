/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureAnisotropyMipmap
import GB300Math.TextureAnisotropyArithmetic

namespace GB300Math.TextureAnisotropyOrientedMipmap

structure Level where
  index : Nat
  groups : Nat
  taps : List TextureAnisotropyOriented.Tap
  deriving Inhabited, Repr

def signedStep (direction : Int) (step : Nat) : Int := 2*(direction*(step:Int)/64)

def levels (a b anisotropy : Nat) (optimized : Bool) : List Level :=
  let q := TextureAnisotropyOriented.majorLog a b
  let (x,y) := TextureAnisotropyOriented.direction a b
  let limit := TextureAnisotropyMipmap.cap anisotropy
  if q<limit then
    let (groups,taps) := TextureAnisotropyOriented.taps q x y
    [⟨0,groups,taps⟩]
  else
    let lod := q-limit
    let level := lod/256
    let fraction := if optimized then
      (max 0 (min 256 (TextureAnisotropyBlend.fraction (lod%256) (2^(limit/256))))).toNat
      else lod%256
    let step := TextureAnisotropyMipmapStep.step fraction
    let sx := signedStep x step
    let sy := signedStep y step
    let (groups,base) := TextureAnisotropyTaps.taps limit false
    let lowerWeights := TextureAnisotropyMipmapWeights.scaleWeights (256-fraction)
      (base.map TextureAnisotropyTaps.Tap.weight)
    let lower := base.zipWith (fun tap weight =>
      (⟨tap.offset/128*sx,tap.offset/128*sy,weight⟩ : TextureAnisotropyOriented.Tap)) lowerWeights
    let (upperGroups,upper) := TextureAnisotropyOriented.taps (limit+fraction-256) x y
    let upperWeights := TextureAnisotropyMipmapWeights.scaleWeights fraction
      (upper.map TextureAnisotropyOriented.Tap.weight)
    let upper := upper.zipWith (fun tap weight => {tap with weight:=weight}) upperWeights
    [⟨level,groups,lower⟩,⟨level+1,upperGroups,upper⟩]

def evaluate (a b anisotropy : Nat) (optimized : Bool)
    (x y : Int) (images : Array (Array Nat)) : Nat := Id.run do
  let mut planes : Array (Array (Nat × Nat)) := #[]
  for level in levels a b anisotropy optimized do
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

theorem signed_step_error (direction : Int) (step : Nat) :
    -126 ≤ 64*signedStep direction step-2*direction*(step:Int) ∧
    64*signedStep direction step-2*direction*(step:Int) ≤ 0 := by
  unfold signedStep
  rw [Int.mul_assoc 2 direction (step:Int)]
  omega

theorem sign_reversal (direction : Int) (step : Nat) :
    signedStep direction step+signedStep (-direction) step=0 ∨
    signedStep direction step+signedStep (-direction) step= -2 := by
  unfold signedStep
  have h : (-direction)*(step:Int)= -(direction*(step:Int)) := Int.neg_mul direction (step:Int)
  rw [h]
  omega

theorem sign_reversal_attained : signedStep 35 72=78 ∧ signedStep (-35) 72= -80 := by decide

theorem axis_reduction (step : Nat) : signedStep 64 step=2*(step:Int) ∧
    signedStep (-64) step= -2*(step:Int) := by
  unfold signedStep
  omega

#print axioms axis_reduction
#print axioms signed_step_error
#print axioms sign_reversal
#print axioms sign_reversal_attained
end GB300Math.TextureAnisotropyOrientedMipmap
