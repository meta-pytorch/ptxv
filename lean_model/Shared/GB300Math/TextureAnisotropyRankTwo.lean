/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureAnisotropyOriented
namespace GB300Math.TextureAnisotropyRankTwo
open TextureGradient2D
set_option maxRecDepth 100000
set_option maxHeartbeats 0

def topExponent (v : TextureGradient2D.Dyadic) : Int := (v.significand.natAbs.log2:Int)+v.scale

def quantized (v : TextureGradient2D.Dyadic) (exponent : Int) : Nat :=
  if v.significand=0 ∨ topExponent v+5<exponent then 0 else
    2*(atScale (magnitude v) (exponent-3)).natAbs+1

def direction (a b : TextureGradient2D.Dyadic) : Int × Int :=
  let exponent := max (if a.significand=0 then topExponent b else topExponent a)
    (if b.significand=0 then topExponent a else topExponent b)
  let x := quantized a exponent
  let y := quantized b exponent
  (signed (a.significand<0) TextureAnisotropyOriented.directionTable[x]![y]!,
   signed (b.significand<0) TextureAnisotropyOriented.directionTable[y]![x]!)

def diagonalComponent (a : TextureGradient2D.Dyadic) : TextureGradient2D.Dyadic :=
  let scaled := scaleCoefficient a 11 4 3
  ⟨signed (a.significand<0) scaled.significand.natAbs,scaled.scale⟩

structure Selection where
  logarithm : Nat
  x : Int
  y : Int
  branch : Nat
  deriving Repr, DecidableEq

def selection (ax ay bx byy : Nat) : Selection :=
  let a := (fromBits ax,fromBits ay)
  let b := (fromBits bx,fromBits byy)
  let p := (guardedAdd a.1 b.1,guardedAdd a.2 b.2)
  let n := (guardedAdd a.1 (negate b.1),guardedAdd a.2 (negate b.2))
  let lengths := #[length a.1 a.2,length b.1 b.2,diagonal p.1 p.2,diagonal n.1 n.2]
  let k := ([0,2,1,3] : List Nat).foldl (fun best next =>
    if le (normalized lengths[next]!) (normalized lengths[best]!) then best else next) 0
  let chosen := (#[a,b,p,n])[k]!
  let chosen := if k≥2 then (diagonalComponent chosen.1,diagonalComponent chosen.2) else chosen
  let chosen := if k=3 ∧ le (normalized lengths[0]!) (normalized lengths[1]!) then
      (negate chosen.1,negate chosen.2) else chosen
  let unit := direction chosen.1 chosen.2
  let major := lengths[k]!
  let q := if major.significand=0 then 0 else
    (2048+TextureAnisotropyArea.logarithm major.significand.natAbs major.scale).toNat
  ⟨q,unit.1,unit.2,k⟩

theorem quantized_zero (exponent : Int) : quantized ⟨0,0⟩ exponent=0 := by
  simp [quantized]

theorem zero_direction : direction ⟨0,0⟩ ⟨0,0⟩=(0,0) := by decide

theorem zero_selection : selection 0 0 0 0=⟨0,0,0,0⟩ := by decide

#print axioms quantized_zero
#print axioms zero_direction
#print axioms zero_selection
end GB300Math.TextureAnisotropyRankTwo
