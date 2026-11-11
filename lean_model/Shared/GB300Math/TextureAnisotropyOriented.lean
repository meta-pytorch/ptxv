/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureAnisotropyTaps
import GB300Math.TextureAnisotropyGeneral
namespace GB300Math.TextureAnisotropyOriented
set_option maxRecDepth 100000
set_option maxHeartbeats 0

def directionTable : Array (Array Nat) := #[
#[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
#[64,45,29,20,16,13,11,9,8,7,6,6,5,5,5,4,4,4,4,3,3,3,3,3,3,3,2,2,2,2,2,2],
#[64,57,45,36,29,24,20,18,16,14,13,11,11,10,9,8,8,7,7,7,6,6,6,6,5,5,5,5,5,4,4,4],
#[64,61,53,45,38,33,29,25,22,20,18,17,16,14,13,13,12,11,11,10,9,9,9,8,8,8,7,7,7,7,6,6],
#[64,62,57,51,45,40,36,32,29,26,24,22,20,19,18,16,16,15,14,13,13,12,11,11,11,10,10,9,9,9,8,8],
#[64,63,59,55,50,45,41,37,34,31,29,26,25,23,22,20,19,18,17,16,16,15,14,14,13,13,12,12,11,11,11,10],
#[64,63,61,57,53,49,45,42,38,36,33,31,29,27,25,24,22,21,20,19,18,18,17,16,16,15,14,14,13,13,13,12],
#[64,63,62,59,56,52,49,45,42,39,37,34,32,30,29,27,26,24,23,22,21,20,19,19,18,17,17,16,16,15,15,14],
#[64,64,62,60,57,54,51,48,45,43,40,38,36,34,32,30,29,27,26,25,24,23,22,21,20,20,19,18,18,17,16,16],
#[64,64,62,61,58,56,53,51,48,45,43,41,38,36,35,33,31,30,29,27,26,25,24,23,22,22,21,20,20,19,18,18],
#[64,64,63,61,59,57,55,52,50,48,45,43,41,39,37,36,34,32,31,30,29,28,26,26,25,24,23,22,22,21,20,20],
#[64,64,63,62,60,58,56,54,52,50,47,45,43,41,40,38,36,35,33,32,31,30,29,28,27,26,25,24,23,23,22,21],
#[64,64,63,62,61,59,57,55,53,51,49,47,45,43,42,40,38,37,36,34,33,32,31,30,29,28,27,26,25,24,24,23],
#[64,64,63,62,61,60,58,56,55,53,51,49,47,45,44,42,40,39,37,36,35,34,33,31,30,30,29,28,27,26,25,25],
#[64,64,63,63,62,60,59,57,56,54,52,50,49,47,45,44,42,41,39,38,37,36,34,33,32,31,30,29,29,28,27,26],
#[64,64,63,63,62,61,59,58,56,55,53,52,50,48,47,45,44,42,41,40,38,37,36,35,34,33,32,31,30,29,29,28],
#[64,64,64,63,62,61,60,59,57,56,54,53,51,50,48,47,45,44,43,41,40,39,38,37,36,34,34,33,32,31,30,29],
#[64,64,64,63,62,61,60,59,58,57,55,54,52,51,49,48,47,45,44,43,41,40,39,38,37,36,35,34,33,32,32,31],
#[64,64,64,63,62,62,61,60,58,57,56,55,53,52,51,49,48,47,45,44,43,42,41,39,38,37,36,36,35,34,33,32],
#[64,64,64,63,63,62,61,60,59,58,57,55,54,53,52,50,49,48,46,45,44,43,42,41,40,39,38,37,36,35,34,33],
#[64,64,64,63,63,62,61,60,59,58,57,56,55,54,52,51,50,49,48,46,45,44,43,42,41,40,39,38,37,36,36,35],
#[64,64,64,63,63,62,62,61,60,59,58,57,56,54,53,52,51,50,49,47,46,45,44,43,42,41,40,39,38,38,37,36],
#[64,64,64,63,63,62,62,61,60,59,58,57,56,55,54,53,52,51,50,48,47,46,45,44,43,42,41,40,40,39,38,37],
#[64,64,64,63,63,63,62,61,60,60,59,58,57,56,55,54,53,51,50,49,48,47,46,45,44,43,42,42,41,40,39,38],
#[64,64,64,64,63,63,62,61,61,60,59,58,57,56,55,54,53,52,51,50,49,48,47,46,45,44,43,43,42,41,40,39],
#[64,64,64,64,63,63,62,62,61,60,59,59,58,57,56,55,54,53,52,51,50,49,48,47,46,45,44,43,43,42,41,40],
#[64,64,64,64,63,63,62,62,61,60,60,59,58,57,56,55,55,54,53,52,51,50,49,48,47,46,45,44,44,43,42,41],
#[64,64,64,64,63,63,62,62,61,61,60,59,58,58,57,56,55,54,53,52,51,51,50,49,48,47,46,45,44,44,43,42],
#[64,64,64,64,63,63,63,62,62,61,60,60,59,58,57,56,56,55,54,53,52,51,50,49,49,48,47,46,45,44,44,43],
#[64,64,64,64,63,63,63,62,62,61,61,60,59,58,58,57,56,55,54,54,53,52,51,50,49,48,48,47,46,45,44,44],
#[64,64,64,64,63,63,63,62,62,61,61,60,59,59,58,57,56,56,55,54,53,52,52,51,50,49,48,48,47,46,45,45],
#[64,64,64,64,63,63,63,62,62,61,61,60,60,59,58,58,57,56,55,55,54,53,52,51,51,50,49,48,47,47,46,45]
]

def quantized (bits exponent : Nat) : Nat :=
  let e := bits/2^23%256
  if e=0 ∨ e+5<exponent then 0 else
    2*((bits%2^23+2^23)/2^(20+exponent-e))+1

def direction (a b : Nat) : Int × Int :=
  let exponent := max (a/2^23%256) (b/2^23%256)
  let x := quantized a exponent
  let y := quantized b exponent
  (TextureGradient2D.signed (a≥2^31) directionTable[x]![y]!,
   TextureGradient2D.signed (b≥2^31) directionTable[y]![x]!)

def majorLog (a b : Nat) : Nat :=
  let v := TextureGradient2D.norm (TextureGradient2D.fromBits a)
    (TextureGradient2D.fromBits b) ⟨0,0⟩ ⟨0,0⟩
  if v.significand=0 then 0 else
    (2048+TextureAnisotropyArea.logarithm v.significand.natAbs v.scale).toNat

structure Tap where
  x : Int
  y : Int
  weight : Nat
  deriving Inhabited, Repr

def taps (q : Nat) (x y : Int) : Nat × List Tap :=
  if q=0 then (1,[⟨0,0,256⟩]) else
  if q<256 then
    let r := 256-TextureAnisotropyTaps.reciprocalTable[q]!
    let dx := 2*(((r:Int)*x+64)/128)
    let dy := 2*(((r:Int)*y+64)/128)
    (2,[⟨-dx,-dy,256⟩,⟨dx,dy,256⟩])
  else
    let (groups,weight) := TextureAnisotropyTaps.configuration q
    let count := 2*((groups*256+2*weight-1)/(2*weight))
    let edge := (groups*256-(count-2)*weight)/2
    (groups,(List.range count).map fun (i : Nat) =>
      let t := 2*(2*(i:Int)-(count:Int)+1)
      ⟨t*x,t*y,if i=0 ∨ i+1=count then edge else weight⟩)

def plane (weight x y : Nat) : Nat × Nat × Nat × Nat :=
  let highX := (weight*x+128)/256
  let lowX := weight-highX
  let lowLow := (lowX*(256-y)+128)/256
  let highHigh := (highX*y+128)/256
  (lowLow,highX-highHigh,lowX-lowLow,highHigh)

theorem direction_rn_brackets : ∀ a b : Fin 32, a.val+b.val>0 →
    let v := directionTable[a.val]![b.val]!
    (2*v-1)^2*(a.val^2+b.val^2) ≤ 128^2*a.val^2 ∧
    128^2*a.val^2 < (2*v+1)^2*(a.val^2+b.val^2) := by decide

theorem direction_bounded : ∀ a b : Fin 32,
    directionTable[a.val]![b.val]! ≤ 64 := by decide

theorem plane_sum (w x y : Nat) (hx : x≤256) (hy : y≤256) :
    let p := plane w x y
    p.1+p.2.1+p.2.2.1+p.2.2.2=w := by
  have h1 := TextureAnisotropyTaps.upperWeight_bounded w x false hx
  have h2 := TextureAnisotropyTaps.upperWeight_bounded ((w*x+128)/256) y false hy
  have h3 := TextureAnisotropyTaps.upperWeight_bounded (w-(w*x+128)/256) (256-y) false (by omega)
  simp only [TextureAnisotropyTaps.upperWeight, Bool.false_eq_true, ↓reduceIte] at h1 h2 h3
  unfold plane
  dsimp only
  omega

#print axioms direction_rn_brackets
#print axioms direction_bounded
#print axioms plane_sum
end GB300Math.TextureAnisotropyOriented
