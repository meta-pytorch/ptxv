/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureGradientAxis

namespace GB300Math.TextureGradient2D

structure Dyadic where
  significand : Int
  scale : Int
  deriving Repr, Inhabited

def signed (negative : Bool) (n : Nat) : Int := if negative then -(n:Int) else n

def normalized (v : Dyadic) : Dyadic :=
  let n := v.significand.natAbs
  if n = 0 then ⟨0,0⟩ else
    let shift := (n.log2:Int)-9
    let m := if shift ≥ 0 then n/2^shift.toNat else n*2^(-shift).toNat
    ⟨signed (v.significand < 0) m, v.scale+shift⟩

def fromBits (v : Nat) : Dyadic :=
  let e : Nat := v/2^23%256
  let m : Nat := v%2^23
  if e = 0 ∨ (e = 255 ∧ m ≠ 0) then ⟨0,0⟩ else
    ⟨signed (v ≥ 2^31) ((m+2^23)/2^14), (e:Int)-136⟩

def atScale (v : Dyadic) (scale : Int) : Int :=
  let delta := v.scale-scale
  let n := if delta ≥ 0 then v.significand.natAbs*2^delta.toNat
    else v.significand.natAbs/2^(-delta).toNat
  signed (v.significand < 0) n

def add (a b : Dyadic) : Dyadic :=
  let scale := min a.scale b.scale
  ⟨atScale a scale+atScale b scale,scale⟩

def guardedAdd (a b : Dyadic) : Dyadic :=
  let scale := max (if a.significand = 0 then b.scale else a.scale)
    (if b.significand = 0 then a.scale else b.scale)-1
  normalized ⟨atScale a scale+atScale b scale,scale⟩

def magnitude (a : Dyadic) : Dyadic := ⟨a.significand.natAbs,a.scale⟩
def negate (a : Dyadic) : Dyadic := ⟨-a.significand,a.scale⟩

def le (a b : Dyadic) : Bool :=
  let scale := min a.scale b.scale
  atScale a scale ≤ atScale b scale

def maximum (a b : Dyadic) : Dyadic := if le a b then b else a

def scaleCoefficient (a : Dyadic) (coefficient denominatorShift roundingShift : Nat) : Dyadic :=
  let a := normalized (magnitude a)
  let divisor := 2^roundingShift
  let product := (a.significand.natAbs*coefficient+divisor/2)/divisor*divisor
  ⟨product,a.scale-(denominatorShift:Int)⟩

def length (a b : Dyadic) : Dyadic :=
  let a := magnitude a
  let b := magnitude b
  let (hi,lo) := if le a b then (b,a) else (a,b)
  add (normalized hi) (scaleCoefficient lo 11 5 3)

def diagonal (a b : Dyadic) : Dyadic :=
  let a := magnitude a
  let b := magnitude b
  let (hi,lo) := if le a b then (b,a) else (a,b)
  add (normalized (scaleCoefficient hi 11 4 3)) (scaleCoefficient lo 121 9 6)

def norm (ax ay bx byy : Dyadic) : Dyadic :=
  let p := diagonal (guardedAdd ax bx) (guardedAdd ay byy)
  let m := diagonal (guardedAdd ax (negate bx)) (guardedAdd ay (negate byy))
  maximum (maximum (length ax ay) (length bx byy)) (maximum p m)

def position (ax ay bx byy : Nat) : Nat :=
  if [ax,ay,bx,byy].any (fun v => v%2^31 = 0x7f800000) then 2048 else
    let r := norm (fromBits ax) (fromBits ay) (fromBits bx) (fromBits byy)
    let n := r.significand.natAbs
    if n = 0 then 0 else
      let top := n.log2
      let index := n*256/2^top%256
      min 2048 ((r.scale+(top:Int)+8)*256+
        (TextureGradientAxis.logTable[index]!:Int)).toNat

def sample (ax ay bx byy : Nat) : Nat := TextureMipmap.encodeQ8 (256+position ax ay bx byy)

theorem position_bounded (ax ay bx byy : Nat) : position ax ay bx byy ≤ 2048 := by
  unfold position
  dsimp only
  split
  · exact Nat.le_refl _
  · split
    · exact Nat.zero_le _
    · exact Nat.min_le_left _ _

#print axioms position_bounded
end GB300Math.TextureGradient2D
