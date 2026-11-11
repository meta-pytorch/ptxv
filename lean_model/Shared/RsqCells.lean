/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import SqrtBounds

set_option maxRecDepth 1000000
set_option maxHeartbeats 0

namespace GB300Math.RsqBounds
open RcpBounds (boxLo boxHi term_box)

structure Parameters where
  (a b c m step : Int)

def parameters (segment : Nat) : Parameters :=
  let (a,b,c) := MufuTables.rsq.coefficients[segment]!
  ⟨a*2^32+4294705152,b*2^18,c,
   (2^23+(segment%64 : Nat)*131072)*2^(segment/64),2^(segment/64)⟩

def numerator (p : Parameters) (k : Int) := p.a+p.b*k+p.c*k^2
def loss (p : Parameters) := p.c*4718593+2^34-1
def minimum (p : Parameters) (left : Int) := (numerator p (left+2047)-loss p)/2^34

def taylor (p : Parameters) (left : Int) : Int × Int × Int × Int × Int × Int :=
  let p0 := p.m*p.a^2-2^139
  let p1 := 2*p.m*p.a*p.b+p.step*p.a^2
  let p2 := p.m*(p.b^2+2*p.a*p.c)+2*p.step*p.a*p.b
  let p3 := 2*p.m*p.b*p.c+p.step*(p.b^2+2*p.a*p.c)
  let p4 := p.m*p.c^2+2*p.step*p.b*p.c
  let p5 := p.step*p.c^2
  (p0+p1*left+p2*left^2+p3*left^3+p4*left^4+p5*left^5,
   p1+2*p2*left+3*p3*left^2+4*p4*left^3+5*p5*left^4,
   p2+3*p3*left+6*p4*left^2+10*p5*left^3,
   p3+4*p4*left+10*p5*left^2,p4+5*p5*left,p5)

def lower (q0 q1 q2 q3 q4 q5 d : Int) :=
  q0+boxLo q1 d+boxLo q2 (d^2)+boxLo q3 (d^3)+boxLo q4 (d^4)+boxLo q5 (d^5)
def upper (q0 q1 q2 q3 q4 q5 d : Int) :=
  q0+boxHi q1 d+boxHi q2 (d^2)+boxHi q3 (d^3)+boxHi q4 (d^4)+boxHi q5 (d^5)

theorem polynomial_box (q0 q1 q2 q3 q4 q5 t d : Int) (ht : 0 ≤ t) (hd : t ≤ d) :
    lower q0 q1 q2 q3 q4 q5 d ≤ q0+q1*t+q2*t^2+q3*t^3+q4*t^4+q5*t^5 ∧
    q0+q1*t+q2*t^2+q3*t^3+q4*t^4+q5*t^5 ≤ upper q0 q1 q2 q3 q4 q5 d := by
  have h1 := term_box q1 t d ht hd
  have h2 := term_box q2 (t^2) (d^2) (pow_nonneg ht 2) (pow_le_pow_left₀ ht hd 2)
  have h3 := term_box q3 (t^3) (d^3) (pow_nonneg ht 3) (pow_le_pow_left₀ ht hd 3)
  have h4 := term_box q4 (t^4) (d^4) (pow_nonneg ht 4) (pow_le_pow_left₀ ht hd 4)
  have h5 := term_box q5 (t^5) (d^5) (pow_nonneg ht 5) (pow_le_pow_left₀ ht hd 5)
  simp only [lower,upper]
  constructor <;> omega

theorem numerator_decreasing (p : Parameters) (i j : Int)
    (hc : 0 ≤ p.c) (hb : p.b+262142*p.c ≤ 0)
    (hi : 0 ≤ i) (hij : i ≤ j) (hj : j ≤ 131071) :
    numerator p j ≤ numerator p i := by
  have hcij := mul_le_mul_of_nonneg_left (show i+j ≤ 262142 by omega) hc
  have h := mul_nonpos_of_nonpos_of_nonneg (show p.b+p.c*(i+j) ≤ 0 by nlinarith)
    (show 0 ≤ j-i by omega)
  dsimp [numerator]
  nlinarith

abbrev CellBound (p : Parameters) (left : Int) : Prop :=
  let (q0,q1,q2,q3,q4,q5) := taylor p left
  let bound := (p.m+p.step*left)*(4*minimum p left-4)
  0 ≤ p.c ∧ p.b+262142*p.c ≤ 0 ∧ 0 < p.m ∧ 0 ≤ p.step ∧
  2^23*2^34 ≤ numerator p 131071-p.c*4718593 ∧ p.a < (2^24+1)*2^34 ∧
  2 ≤ minimum p left ∧
  -bound*2^68+2*(p.m+p.step*(left+2047))*numerator p left*loss p ≤
    lower q0 q1 q2 q3 q4 q5 2047 ∧
  upper q0 q1 q2 q3 q4 q5 2047 ≤ bound*2^68

theorem cells_valid : ∀ segment : Fin 128, ∀ cell : Fin 64,
    CellBound (parameters segment.val) ((cell.val : Int)*2048) := by decide +kernel


#print axioms cells_valid
end GB300Math.RsqBounds
