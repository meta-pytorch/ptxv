/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Mathlib
import GB300Math.Mufu

namespace GB300Math.MufuRcp32Bounds

structure Cubic where
  a : Int
  b : Int
  c : Int
  d : Int

def Cubic.eval (p : Cubic) (x : Int) : Int :=
  p.a + p.b*x + p.c*x^2 + p.d*x^3

def Cubic.shift (p : Cubic) (x : Int) : Cubic :=
  ⟨p.eval x, p.b + 2*p.c*x + 3*p.d*x^2, p.c + 3*p.d*x, p.d⟩

def bernsteinValid (p : Cubic) (start width : Int) : Bool :=
  let q := p.shift start
  0 ≤ 3*q.a && 0 ≤ 3*q.a + width*q.b &&
    0 ≤ 3*q.a + 2*width*q.b + width^2*q.c &&
    0 ≤ 3*(q.a + width*q.b + width^2*q.c + width^3*q.d)

theorem bernstein_nonnegative (p : Cubic) (start width x : Int)
    (hw : 0 < width) (hx : start ≤ x) (hy : x ≤ start+width)
    (checked : bernsteinValid p start width = true) : 0 ≤ p.eval x := by
  let q := p.shift start
  have h := checked
  simp only [bernsteinValid, Bool.and_eq_true, decide_eq_true_eq] at h
  have h0 : 0 ≤ 3*q.a := h.1.1.1
  have h1 : 0 ≤ 3*q.a + width*q.b := h.1.1.2
  have h2 : 0 ≤ 3*q.a + 2*width*q.b + width^2*q.c := h.1.2
  have h3 : 0 ≤ 3*(q.a + width*q.b + width^2*q.c + width^3*q.d) := h.2
  have hleft : 0 ≤ x-start := by omega
  have hright : 0 ≤ width-(x-start) := by omega
  have nonnegative : 0 ≤
      (3*q.a)*(width-(x-start))^3 +
      3*(3*q.a+width*q.b)*(x-start)*(width-(x-start))^2 +
      3*(3*q.a+2*width*q.b+width^2*q.c)*(x-start)^2*(width-(x-start)) +
      (3*(q.a+width*q.b+width^2*q.c+width^3*q.d))*(x-start)^3 := by positivity
  have identity :
      (3*q.a)*(width-(x-start))^3 +
      3*(3*q.a+width*q.b)*(x-start)*(width-(x-start))^2 +
      3*(3*q.a+2*width*q.b+width^2*q.c)*(x-start)^2*(width-(x-start)) +
      (3*(q.a+width*q.b+width^2*q.c+width^3*q.d))*(x-start)^3 =
        (3*width^3)*p.eval x := by dsimp [q, Cubic.shift, Cubic.eval]; ring
  rw [identity] at nonnegative
  by_contra hn
  have hn : p.eval x < 0 := by omega
  have hm := mul_neg_of_pos_of_neg (by positivity : 0 < 3*width^3) hn
  omega

def squareValid (k : Nat) : Bool :=
  k*k ≤ Fp64Seed.squareTruncated k 17 + 1048577 &&
    Fp64Seed.squareTruncated k 17 ≤ k*k

def lowerPoly (j : Nat) : Cubic :=
  let (c0,c1,c2) := MufuTables.rcp.coefficients[j]!
  let m : Int := 2^23 + (j : Int)*2^16
  let a := c0*2^32 + 1060634624 - c2*4*1048577
  let b := c1*2^19
  let c := c2*4
  ⟨m*a - 2^81 + (2^24-m)*2^34, a+m*b-2^34, b+m*c, c⟩

def upperPoly (j : Nat) : Cubic :=
  let (c0,c1,c2) := MufuTables.rcp.coefficients[j]!
  let m : Int := 2^23 + (j : Int)*2^16
  let a := c0*2^32 + 1060634624
  let b := c1*2^19
  let c := c2*4
  ⟨2^81+2^58-m*a, -a-m*b, -b-m*c, -c⟩

theorem coefficient_certificates : ∀ j : Fin 128,
    0 ≤ (MufuTables.rcp.coefficients[j.val]!).2.2 ∧
    ∀ half : Fin 2,
      bernsteinValid (lowerPoly j.val) (half.val*32768) 32767 = true ∧
      bernsteinValid (upperPoly j.val) (half.val*32768) 32767 = true := by
  decide +kernel

theorem lower_nonnegative (j : Fin 128) (k : Fin 65536) :
    0 ≤ (lowerPoly j.val).eval k.val := by
  have h := (coefficient_certificates j).2 ⟨k.val/32768, by omega⟩
  exact bernstein_nonnegative _ ((k.val/32768 : Nat)*32768) 32767 k.val
    (by norm_num) (by omega) (by omega) h.1

theorem upper_nonnegative (j : Fin 128) (k : Fin 65536) :
    0 ≤ (upperPoly j.val).eval k.val := by
  have h := (coefficient_certificates j).2 ⟨k.val/32768, by omega⟩
  exact bernstein_nonnegative _ ((k.val/32768 : Nat)*32768) 32767 k.val
    (by norm_num) (by omega) (by omega) h.2

#print axioms bernstein_nonnegative
#print axioms coefficient_certificates
#print axioms lower_nonnegative
#print axioms upper_nonnegative

end GB300Math.MufuRcp32Bounds
