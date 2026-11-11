/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FabricFP8

namespace FabricDirectedBounds
open FabricFP8

def scale (e4 : Bool) : Nat := if e4 then 8 else 4

def gapCertificate (e4 : Bool) (code : Nat) : Bool :=
  let lo := unit e4 code
  let hi := unit e4 (code+1)
  hi > lo ∧ (hi = lo+1 ∨ scale e4 * (hi-lo) ≤ lo)

theorem e4_gaps : (List.range 126).all (gapCertificate true) = true := by decide +kernel

theorem e5_gaps : (List.range 123).all (gapCertificate false) = true := by decide +kernel

theorem gaps (e4 : Bool) (code : Nat) (hc : code < maximum e4) :
    unit e4 code < unit e4 (code+1) ∧
    (unit e4 (code+1) = unit e4 code+1 ∨
      scale e4 * (unit e4 (code+1)-unit e4 code) ≤ unit e4 code) := by
  cases e4 <;> simp only [maximum] at hc
  · have hcert := List.all_eq_true.mp e5_gaps code (List.mem_range.mpr hc)
    simpa [gapCertificate] using hcert
  · have hcert := List.all_eq_true.mp e4_gaps code (List.mem_range.mpr hc)
    simpa [gapCertificate] using hcert

theorem directed_relative (e4 : Bool) (code x y : Nat)
    (hc : code < maximum e4)
    (hx : unit e4 code < x) (hx' : x < unit e4 (code+1))
    (hy : y = unit e4 code ∨ y = unit e4 (code+1)) :
    scale e4 * (Int.natAbs ((y : Int)-x)) ≤ x := by
  obtain ⟨hg, h⟩ := gaps e4 code hc
  rcases h with hunit | hgap
  · omega
  · have hdist : Int.natAbs ((y : Int)-x) ≤ unit e4 (code+1)-unit e4 code := by
      rcases hy with rfl | rfl <;> omega
    have hm := Nat.mul_le_mul_left (scale e4) hdist
    omega

#print axioms e4_gaps
#print axioms e5_gaps
#print axioms directed_relative
end FabricDirectedBounds

namespace FabricDirectedBounds
open FabricFP8

def numerator (e4 : Bool) : Nat := if e4 then 16383 else 536870911
def denominator (e4 : Bool) : Nat := if e4 then 131073 else 2147483649

def sharpCertificate (e4 : Bool) (code : Nat) : Bool :=
  denominator e4 * (unit e4 (code+1)-unit e4 code-1) ≤
    numerator e4 * (unit e4 code+1)

theorem e4_sharp_gaps : (List.range 127).all (sharpCertificate true) = true := by decide +kernel

theorem e5_sharp_gaps : (List.range 124).all (sharpCertificate false) = true := by decide +kernel

theorem sharp_gaps (e4 : Bool) (code : Nat) (hc : code ≤ maximum e4) :
    denominator e4 * (unit e4 (code+1)-unit e4 code-1) ≤
      numerator e4 * (unit e4 code+1) := by
  cases e4 <;> simp only [maximum] at hc
  · change code ≤ 123 at hc
    have h := List.all_eq_true.mp e5_sharp_gaps code (List.mem_range.mpr (by omega))
    simpa [sharpCertificate] using h
  · change code ≤ 126 at hc
    have h := List.all_eq_true.mp e4_sharp_gaps code (List.mem_range.mpr (by omega))
    simpa [sharpCertificate] using h

theorem sharp_directed_relative (e4 : Bool) (code x y : Nat)
    (hc : code ≤ maximum e4)
    (hx : unit e4 code < x) (hx' : x < unit e4 (code+1))
    (hy : y = unit e4 code ∨ y = unit e4 (code+1)) :
    denominator e4 * (Int.natAbs ((y : Int)-x)) ≤ numerator e4 * x := by
  have hgap := sharp_gaps e4 code hc
  have hdist : Int.natAbs ((y : Int)-x) ≤ unit e4 (code+1)-unit e4 code-1 := by
    rcases hy with rfl | rfl <;> omega
  have hm := Nat.mul_le_mul_left (denominator e4) hdist
  have hxlo : unit e4 code+1 ≤ x := by omega
  have hmx := Nat.mul_le_mul_left (numerator e4) hxlo
  omega

theorem e4_attains :
    denominator true * (unit true 121-(unit true 120+1)) =
      numerator true * (unit true 120+1) := by decide +kernel

theorem e5_attains :
    denominator false * (unit false 121-(unit false 120+1)) =
      numerator false * (unit false 120+1) := by decide +kernel

#print axioms e4_sharp_gaps
#print axioms e5_sharp_gaps
#print axioms sharp_directed_relative
#print axioms e4_attains
#print axioms e5_attains
end FabricDirectedBounds
