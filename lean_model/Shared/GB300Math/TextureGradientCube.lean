/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.Mufu
import GB300Math.TextureGradient3D

namespace GB300Math.TextureGradientCube
open TextureGradient2D TextureGradient3D

def encode32 (v : TextureGradient2D.Dyadic) : Nat :=
  let v := rounded32 v
  let n := v.significand.natAbs
  let sign := if v.significand < 0 then 2^31 else 0
  if n = 0 then sign else
    let e := (n.log2:Int)+v.scale
    if e > 127 then sign+0x7f800000 else
      if e < -126 then sign+(n*2^(v.scale+149).toNat) else
        let shift := (n.log2:Int)-23
        let m := if shift ≥ 0 then n/2^shift.toNat else n*2^(-shift).toNat
        sign+(e+127).toNat*2^23+m%2^23

def sumBits (a b : Nat) : Nat := encode32 (add (exactFromBits a) (exactFromBits b))

def normDirection (v : Array Nat) : Array TextureGradient2D.Dyadic :=
  let top := v.foldl (fun m x => max m (x%2^31)) 0
  if top = 0 then #[⟨0,0⟩,⟨0,0⟩,⟨0,0⟩] else
    let factor := exactFromBits (Mufu.rcp top)
    v.map (fun x =>
      let a := exactFromBits x
      rounded32 ⟨a.significand*factor.significand,a.scale+factor.scale⟩)

def axis (v : Array Nat) : Nat :=
  if max (v[0]!%2^31) (v[1]!%2^31) ≤ v[2]!%2^31 then 2
  else if v[0]!%2^31 ≤ v[1]!%2^31 then 1 else 0

def projected (v delta : Array Nat) : Array TextureGradient2D.Dyadic × Bool × Bool :=
  let a := axis v
  let other := (Array.range 3).map (fun i => sumBits v[i]! delta[i]!)
  let b := axis other
  let origin := normDirection v
  let destination := normDirection other
  let isNegative := v[a]! ≥ 2^31
  let major := if isNegative then negate destination[a]! else destination[a]!
  let indices := (Array.range 3).filter (fun i => i != a)
  let gradient := indices.map (fun i =>
    let component := destination[i]!
    let difference := if i = b then
        let negative := component.significand < 0
        let start := if negative then negate origin[i]! else origin[i]!
        let first := normalized (rounded32 (add ⟨1,0⟩ (negate start)))
        let second := normalized (rounded32 (add ⟨1,0⟩ (negate major)))
        let value := rounded32 (add first second)
        if negative then negate value else value
      else
        let value := add component (negate origin[i]!)
        if (component.significand < 0) != (origin[i]!.significand < 0)
          then value else rounded32 value
    normalized ⟨difference.significand,difference.scale-1⟩)
  let flip := (b = a && (major.significand ≤ 0)) || other.all (fun x => x%2^31 = 0)
  (gradient,a != b,flip)

def position (v dx dy : Array Nat) : Nat :=
  let (x,crossX,flipX) := projected v dx
  let (y,crossY,flipY) := projected v dy
  if flipX || flipY then 2048 else
    let r := if crossX || crossY then
        maximum (TextureGradient2D.length x[0]! x[1]!) (TextureGradient2D.length y[0]! y[1]!)
      else TextureGradient2D.norm x[0]! x[1]! y[0]! y[1]!
    let n := r.significand.natAbs
    if n = 0 then 0 else
      let top := n.log2
      let index := n*256/2^top%256
      min 2048 ((r.scale+(top:Int)+8)*256+(TextureGradientAxis.logTable[index]!:Int)).toNat

def sample (v dx dy : Array Nat) : Nat := TextureMipmap.encodeQ8 (256+position v dx dy)

def normalizable (v delta : Array Nat) : Bool :=
  let other := (Array.range 3).map (fun i => sumBits v[i]! delta[i]!)
  let top := other.foldl (fun m x => max m (x%2^31)) 0
  top = 0 || Mufu.rcp top % 2^31 < 0x7f800000

def domain (v dx dy : Array Nat) : Bool :=
  v.size = 3 && dx.size = 3 && dy.size = 3 &&
  (v++dx++dy).all (fun x => x%2^31 ≤ 0x5d800000) &&
  0x21800000 ≤ v.foldl (fun m x => max m (x%2^31)) 0 &&
  normalizable v dx && normalizable v dy

theorem position_bounded (v dx dy : Array Nat) : position v dx dy ≤ 2048 := by
  unfold position
  dsimp only
  split
  · exact Nat.le_refl _
  · split <;> split
    · exact Nat.zero_le _
    · exact Nat.min_le_left _ _
    · exact Nat.zero_le _
    · exact Nat.min_le_left _ _

#print axioms position_bounded
end GB300Math.TextureGradientCube
