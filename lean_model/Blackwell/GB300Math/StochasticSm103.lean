/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std.Tactic

namespace GB300Math.StochasticSm103

inductive Format where
  | f16 | bf16 | e4m3 | e5m2 | e2m3 | e3m2 | e2m1
  deriving DecidableEq, Repr

def isPair : Format → Bool
  | .f16 | .bf16 => true
  | _ => false

def parameters : Format → Nat × Nat × Nat × Nat
  | .f16 => (5, 10, 15, 31743)
  | .bf16 => (8, 7, 127, 32639)
  | .e4m3 => (4, 3, 7, 126)
  | .e5m2 => (5, 2, 15, 123)
  | .e2m3 => (2, 3, 1, 31)
  | .e3m2 => (3, 2, 3, 31)
  | .e2m1 => (2, 1, 1, 7)

def reverseBits : Nat → Nat → Nat
  | 0, _ => 0
  | n + 1, x => (x % 2) * 2^n + reverseBits n (x / 2)

theorem reverseBits_bound (n x : Nat) : reverseBits n x < 2^n := by
  induction n generalizing x with
  | zero => simp [reverseBits]
  | succ n ih =>
    have h := ih (x / 2)
    have hx : x % 2 = 0 ∨ x % 2 = 1 := by omega
    rcases hx with hx | hx <;> simp [reverseBits, hx, Nat.pow_succ] <;> omega

def entropyLane (f : Format) (bits lane : Nat) : Nat :=
  if isPair f then bits / 2^(16*lane) % 65536
  else
    let r := if f == .e2m1 then
        let shifted := bits / 2^(8*(lane / 2))
        shifted % 256 + (shifted / 65536 % 256) * 256
      else bits / 2^(16*(lane / 2)) % 65536
    if lane % 2 = 0 then r else reverseBits 16 r

def roundMantissa (mantissa entropy drop bits : Nat) : Nat :=
  (mantissa + (entropy % 2^bits) * 2^(drop - bits)) / 2^drop

theorem noise_bound (entropy drop bits : Nat) (h : bits ≤ drop) :
    (entropy % 2^bits) * 2^(drop - bits) < 2^drop := by
  have hpow : 2^drop = 2^bits * 2^(drop - bits) := by
    rw [← Nat.pow_add]
    congr 1
    omega
  rw [hpow]
  exact Nat.mul_lt_mul_of_pos_right (Nat.mod_lt _ (Nat.two_pow_pos _))
    (Nat.two_pow_pos _)

theorem exact_grid (q entropy drop bits : Nat) (h : bits ≤ drop) :
    roundMantissa (q * 2^drop) entropy drop bits = q := by
  have hn := noise_bound entropy drop bits h
  unfold roundMantissa
  rw [Nat.mul_comm q, Nat.mul_add_div (Nat.two_pow_pos _),
    Nat.div_eq_of_lt hn, Nat.add_zero]

theorem strict_one_step_error (mantissa entropy drop bits : Nat) (h : bits ≤ drop) :
    roundMantissa mantissa entropy drop bits * 2^drop < mantissa + 2^drop ∧
    mantissa < roundMantissa mantissa entropy drop bits * 2^drop + 2^drop := by
  have hn := noise_bound entropy drop bits h
  have hr := Nat.mod_lt
    (mantissa + (entropy % 2^bits) * 2^(drop - bits)) (Nat.two_pow_pos drop)
  have hd := Nat.mod_add_div
    (mantissa + (entropy % 2^bits) * 2^(drop - bits)) (2^drop)
  rw [Nat.mul_comm (2^drop)] at hd
  unfold roundMantissa
  omega

def lane (f : Format) (relu satfinite : Bool) (input entropy : Nat) : Nat :=
  let (exponentBits, fractionBits, bias, maximum) := parameters f
  let x := input % 2^32
  let negative := x / 2^31 != 0
  let sign := (x / 2^31) * 2^(exponentBits + fractionBits)
  let exponent := x / 2^23 % 256
  let fraction := x % 2^23
  let limit := if isPair f && !satfinite then maximum + 1 else maximum
  if exponent = 255 && fraction ≠ 0 then
    if isPair f then 0x7fff
    else if f == .e4m3 || f == .e5m2 then 127 else maximum
  else if relu && negative then 0
  else if exponent = 255 then sign + limit
  else
    let mantissa := fraction + if exponent = 0 then 0 else 2^23
    let targetExponent : Int := ((max exponent 1 : Nat) : Int) - 127 + bias
    let drop := 23 - fractionBits + (max 0 (1 - targetExponent)).toNat
    let entropyBits := if isPair f then 23 - fractionBits else 16
    let value := (max 0 (targetExponent - 1)).toNat * 2^fractionBits +
      roundMantissa mantissa entropy drop entropyBits
    sign + min value limit

def packed (f : Format) (relu satfinite : Bool) (a b c d entropy : Nat) : Nat :=
  let width := if isPair f then 16 else if f == .e2m1 then 4 else 8
  let low := lane f relu satfinite a (entropyLane f entropy 0) +
    lane f relu satfinite b (entropyLane f entropy 1) * 2^width
  if isPair f then low
  else low + lane f relu satfinite c (entropyLane f entropy 2) * 2^(2*width) +
    lane f relu satfinite d (entropyLane f entropy 3) * 2^(3*width)

theorem rectified_negative_zero : lane .f16 true false 0x80000000 0 = 0 := by decide
theorem unbounded_positive_infinity : lane .f16 false false 0x7f800000 0 = 0x7c00 := by decide
theorem saturation_preserves_nan : lane .f16 true true 0xffc00001 0 = 0x7fff := by decide
theorem shared_entropy_witness :
    entropyLane .e4m3 0x89abcdef 0 = 0xcdef ∧
    entropyLane .e4m3 0x89abcdef 1 = 0xf7b3 ∧
    entropyLane .e2m1 0x89abcdef 0 = 0xabef ∧
    entropyLane .e2m1 0x89abcdef 2 = 0x89cd := by decide

#print axioms exact_grid
#print axioms strict_one_step_error
#print axioms reverseBits_bound
#print axioms rectified_negative_zero
#print axioms unbounded_positive_infinity
#print axioms saturation_preserves_nan
#print axioms shared_entropy_witness

end GB300Math.StochasticSm103
