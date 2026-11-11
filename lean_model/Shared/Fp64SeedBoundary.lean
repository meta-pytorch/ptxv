/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Fp64SeedBounds

set_option maxRecDepth 100000
set_option maxHeartbeats 5000000
set_option exponentiation.threshold 2000

namespace GB300Math.Fp64SeedBoundary

theorem top_mantissa (fraction : Nat) (hf : fraction < 2^20) :
    Fp64Seed.interpolate false (fraction*8) = 2^24 ↔ fraction = 0 := by
  rw [Fp64SeedBounds.reciprocal_interpolate]
  constructor
  · intro h
    have hr := RcpBounds.interpolate_range (fraction*8) (by omega)
    have he := (RcpBounds.interpolate_error (fraction*8) (by omega)).2
    have hz : Mufu.interpolate MufuTables.rcp (fraction*8) = 2^24 := by omega
    rw [hz] at he
    norm_num at he
    nlinarith
  · rintro rfl
    decide +kernel

def highInput (negative : Bool) (exponent fraction : Nat) : Nat :=
  (if negative then 2^31 else 0)+exponent*2^20+fraction

theorem fields (negative : Bool) (exponent fraction : Nat)
    (he : exponent < 2048) (hf : fraction < 2^20) :
    let w := highInput negative exponent fraction
    w%2^32 = w ∧ w/2^20%2048 = exponent ∧ w%2^20 = fraction ∧
      w &&& 2^31 = (if negative then 2^31 else 0) := by
  have h0 : exponent*2^20+fraction < 2^31 := by omega
  cases negative
  · simp only [highInput,Bool.false_eq_true,↓reduceIte,Nat.zero_add]
    refine ⟨by omega,by omega,by omega,?_⟩
    exact RcpBounds.small_sign _ h0
  · simp only [highInput,↓reduceIte]
    refine ⟨by omega,by omega,by omega,?_⟩
    rw [Nat.add_assoc]
    exact RcpBounds.high_sign _ h0

theorem high_exponent2045 (negative : Bool) (fraction : Nat) (hf : fraction < 2^20) :
    Fp64Seed.highWord false (highInput negative 2045 fraction) =
      (if negative then 2^31 else 0)+(if fraction = 0 then 2^20 else 0) := by
  have h := fields negative 2045 fraction (by decide) hf
  have hs := h.2.2.2
  change highInput negative 2045 fraction &&& 2147483648 = (if negative then 2^31 else 0) at hs
  have hi := top_mantissa fraction hf
  unfold Fp64Seed.highWord
  simp only [h.1,h.2.1,h.2.2.1,hs]
  by_cases hz : fraction = 0
  · have hm := hi.mpr hz
    simp only [show (2045 : Nat) ≠ 0 by decide,Bool.false_eq_true,false_and,
      show (2045 : Nat) ≠ 2047 by decide,ite_false,Nat.zero_mul,Nat.zero_add,hm]
    cases negative <;> norm_num [hz]
  · have hm := mt hi.mp hz
    change ¬Fp64Seed.interpolate false (fraction*8) = 16777216 at hm
    simp [hm,hz]

theorem high_exponent2046 (negative : Bool) (fraction : Nat) (hf : fraction < 2^20) :
    Fp64Seed.highWord false (highInput negative 2046 fraction) =
      (if negative then 2^31 else 0) := by
  have h := fields negative 2046 fraction (by decide) hf
  have hs := h.2.2.2
  change highInput negative 2046 fraction &&& 2147483648 = (if negative then 2^31 else 0) at hs
  unfold Fp64Seed.highWord
  simp only [h.1,h.2.1,h.2.2.1,hs]
  simp only [show (2046 : Nat) ≠ 0 by decide,Bool.false_eq_true,false_and,
    show (2046 : Nat) ≠ 2047 by decide,ite_false,Nat.zero_mul,Nat.zero_add]
  generalize Fp64Seed.interpolate false (fraction*8) = z
  have hn : (2046-(2046 : Nat) : Int)-(if z = 2^24 then 0 else 1) ≤ 0 := by
    split <;> decide
  exact ite_eq_left hn

def inputWord (negative : Bool) (exponent fraction low : Nat) : Nat :=
  (if negative then 2^63 else 0)+exponent*2^52+fraction*2^32+low

theorem input_high (negative : Bool) (exponent fraction low : Nat) (hl : low < 2^32) :
    inputWord negative exponent fraction low /2^32 = highInput negative exponent fraction := by
  cases negative <;> dsimp [inputWord,highInput] <;> omega

theorem boundary_bits (negative : Bool) (exponent fraction low : Nat)
    (he : exponent = 2045 ∨ exponent = 2046) (hf : fraction < 2^20) (hl : low < 2^32) :
    Fp64Seed.evaluate false (inputWord negative exponent fraction low) =
      (if negative then 2^63 else 0)+(if exponent = 2045 ∧ fraction = 0 then 2^52 else 0) := by
  rw [Fp64Seed.evaluate,input_high negative exponent fraction low hl]
  rcases he with rfl | rfl
  · rw [high_exponent2045 negative fraction hf]
    cases negative <;> by_cases hz : fraction = 0 <;> norm_num [hz]
  · rw [high_exponent2046 negative fraction hf]
    cases negative <;> norm_num

def finiteValue (word : Nat) : ℚ :=
  let e : Nat := word/2^52%2048
  let f : Nat := word%2^52
  (if word < 2^63 then 1 else -1)*
    (if e = 0 then (f : ℚ)*(2 : ℚ)^(-1074 : Int)
     else (2^52+(f : ℚ))*(2 : ℚ)^((e : Int)-1075))

theorem flushed_relative_error (negative : Bool) (exponent fraction low : Nat)
    (he : exponent = 2045 ∨ exponent = 2046) (hf : fraction < 2^20) (hl : low < 2^32)
    (hflush : exponent = 2046 ∨ fraction ≠ 0) :
    |finiteValue (Fp64Seed.evaluate false (inputWord negative exponent fraction low))*
      finiteValue (inputWord negative exponent fraction low)-1| = 1 := by
  rw [boundary_bits negative exponent fraction low he hf hl]
  rw [ite_eq_right (show ¬(exponent = 2045 ∧ fraction = 0) by omega)]
  cases negative <;> norm_num [finiteValue]

theorem nonflushed_relative_error (negative : Bool) (low : Nat) (hl : low < 2^32) :
    finiteValue (Fp64Seed.evaluate false (inputWord negative 2045 0 low))*
      finiteValue (inputWord negative 2045 0 low)-1 = (low : ℚ)/2^52 := by
  have he : inputWord negative 2045 0 low/2^52%2048 = 2045 := by
    cases negative <;> dsimp [inputWord] <;> omega
  have hf : inputWord negative 2045 0 low%2^52 = low := by
    cases negative <;> dsimp [inputWord] <;> omega
  have hs : inputWord negative 2045 0 low < 2^63 ↔ negative = false := by
    clear he hf
    cases negative <;> norm_num [inputWord] <;> omega
  rw [boundary_bits negative 2045 0 low (Or.inl rfl) (by decide) hl]
  simp only [finiteValue, he, hf, hs]
  cases negative <;> norm_num <;> ring

theorem nonflushed_bound (negative : Bool) (low : Nat) (hl : low < 2^32) :
    |finiteValue (Fp64Seed.evaluate false (inputWord negative 2045 0 low))*
      finiteValue (inputWord negative 2045 0 low)-1| < (1 : ℚ)/2^20 := by
  rw [nonflushed_relative_error negative low hl,abs_of_nonneg (by positivity)]
  have h : (low : ℚ) < 4294967296 := by exact_mod_cast hl
  norm_num
  linarith

#print axioms top_mantissa
#print axioms boundary_bits
#print axioms flushed_relative_error
#print axioms nonflushed_relative_error
#print axioms nonflushed_bound
end GB300Math.Fp64SeedBoundary
