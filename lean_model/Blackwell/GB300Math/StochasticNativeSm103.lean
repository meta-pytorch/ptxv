/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.StochasticSm103
namespace GB300Math.StochasticNativeSm103
open GB300Math.StochasticSm103

def lane (f : Format) (relu satfinite : Bool) (input entropy : Nat) : Nat :=
  let (exponentBits, fractionBits, bias, maximum) := parameters f
  let x := input % 2^32
  let negative := x / 2^31 != 0
  let sign := (x / 2^31) * 2^(exponentBits + fractionBits)
  let exponent := x / 2^23 % 256
  let fraction := x % 2^23
  let limit := if satfinite then maximum else maximum + 1
  if exponent = 255 && fraction ≠ 0 then
    if isPair f then 0x7fff
    else if f == .e4m3 || f == .e5m2 then 127 else maximum
  else if relu && negative then 0
  else if exponent = 255 then
    if f == .e4m3 && !satfinite then 127 else sign + limit
  else
    let mantissa := fraction + if exponent = 0 then 0 else 2^23
    let targetExponent : Int := ((max exponent 1 : Nat) : Int) - 127 + bias
    let drop := 23 - fractionBits + (max 0 (1 - targetExponent)).toNat
    let entropyBits := if isPair f then 23 - fractionBits else 16
    let value := (max 0 (targetExponent - 1)).toNat * 2^fractionBits +
      roundMantissa mantissa entropy drop entropyBits
    if f == .e4m3 && !satfinite && value > maximum then 127
    else sign + min value limit

def entropy (f : Format) (c : Nat) : Nat :=
  if f == .e2m1 then c / 256 % 256 + (c / 2^24 % 256)*256
  else c / 65536 % 65536

def packed (f : Format) (relu satfinite : Bool) (a b c : Nat) : Nat :=
  let r := entropy f c
  let width := if f == .e2m1 then 4 else 8
  let mergeBits := 32-2*width
  (c % 2^mergeBits)*2^(2*width) +
    lane f relu satfinite a (reverseBits 16 r)*2^width +
    lane f relu satfinite b r

theorem e4_nan_ignores_sign : lane .e4m3 false false 0xff800000 0 = 127 := by decide +kernel
theorem fp8_entropy_partition : entropy .e4m3 0x89abcdef = 0x89ab := by decide
theorem fp4_entropy_partition : entropy .e2m1 0x89abcdef = 0x89cd := by decide
theorem merge_witness : packed .e4m3 false true 0x3f87ffff 0xbfc80001 1 = 0x138bc := by decide +kernel
#print axioms e4_nan_ignores_sign
#print axioms fp8_entropy_partition
#print axioms fp4_entropy_partition
#print axioms merge_witness

end GB300Math.StochasticNativeSm103
