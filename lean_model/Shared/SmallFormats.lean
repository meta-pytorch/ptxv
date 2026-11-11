/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FloatLib.Floats.Formats.BinaryInterchange.Configured

set_option maxRecDepth 100000
set_option maxHeartbeats 50000000

namespace GB300Math.SmallFormats

open FloatLib.Floats

inductive Format where
  | e4m3 | e5m2 | e2m3 | e3m2 | e2m1
  deriving DecidableEq, Repr

def parameters : Format → Nat × Nat × Nat
  | .e4m3 => (4, 3, 7)
  | .e5m2 => (5, 2, 15)
  | .e2m3 => (2, 3, 1)
  | .e3m2 => (3, 2, 3)
  | .e2m1 => (2, 1, 1)

def exceptional (f : Format) (e fraction : Nat) : Bool :=
  (f == .e4m3 && e == 15 && fraction == 7) || (f == .e5m2 && e == 31)

def toHalf (f : Format) (word : Nat) : Nat :=
  let (exponentBits, fractionBits, bias) := parameters f
  let e := (word / 2 ^ fractionBits) % 2 ^ exponentBits
  let fraction := word % 2 ^ fractionBits
  let sign := (word / 2 ^ (exponentBits + fractionBits)) % 2 * 32768
  if exceptional f e fraction then
    if f == .e5m2 && fraction == 0 then sign + 0x7c00 else 0x7fff
  else if f == .e5m2 then sign + e * 1024 + fraction * 256
  else if e == 0 then
    if fraction == 0 then sign else
      let leading := fraction.log2
      sign + (16 - bias - fractionBits + leading) * 1024 +
        (fraction - 2 ^ leading) * 2 ^ (10 - leading)
  else sign + (e + 15 - bias) * 1024 + fraction * 2 ^ (10 - fractionBits)

def value (f : Format) (word : Nat) : Option ℚ :=
  let (exponentBits, fractionBits, bias) := parameters f
  let e := (word / 2 ^ fractionBits) % 2 ^ exponentBits
  let fraction := word % 2 ^ fractionBits
  let sign : ℚ := if (word / 2 ^ (exponentBits + fractionBits)) % 2 == 0 then 1 else -1
  if exceptional f e fraction then none else
    let significand := if e == 0 then fraction else 2 ^ fractionBits + fraction
    let power : Int := (if e == 0 then 1 else e) - (bias : Int) - (fractionBits : Int)
    some (sign * significand * (2 : ℚ) ^ power)

def halfValue (word : Nat) : Option ℚ :=
  ExecFloat.Binary.toRat? (ExecFloat.Binary.ofNatBits word : ExecFloat.Binary 5 10)

theorem e4m3_exact : ∀ w : Fin 256, halfValue (toHalf .e4m3 w.val) = value .e4m3 w.val := by
  decide +kernel

theorem e5m2_exact : ∀ w : Fin 256, halfValue (toHalf .e5m2 w.val) = value .e5m2 w.val := by
  decide +kernel

theorem e2m3_exact : ∀ w : Fin 64, halfValue (toHalf .e2m3 w.val) = value .e2m3 w.val := by
  decide +kernel

theorem e3m2_exact : ∀ w : Fin 64, halfValue (toHalf .e3m2 w.val) = value .e3m2 w.val := by
  decide +kernel

theorem e2m1_exact : ∀ w : Fin 16, halfValue (toHalf .e2m1 w.val) = value .e2m1 w.val := by
  decide +kernel

def e8m0ToBf16 (word : Nat) : Nat :=
  let code := word % 256
  if code == 255 then 0x7fff else if code == 0 then 64 else code * 128

theorem e8m0_exact : ∀ w : Fin 255,
    ExecFloat.Binary.toRat? (ExecFloat.Binary.ofNatBits (e8m0ToBf16 w.val) : ExecFloat.Binary 8 7) =
      some ((2 : ℚ) ^ ((w.val : Int) - 127)) := by
  decide +kernel

#print axioms e8m0_exact
#print axioms e4m3_exact
#print axioms e5m2_exact
#print axioms e2m3_exact
#print axioms e3m2_exact
#print axioms e2m1_exact

end GB300Math.SmallFormats

open Lean GB300Math.SmallFormats

def main : IO Unit := do
  for (name, format) in [("E4M3", Format.e4m3), ("E5M2", .e5m2), ("E2M3", .e2m3), ("E3M2", .e3m2), ("E2M1", .e2m1)] do
    let table := (List.range 256).map (toHalf format)
    IO.println s!"SMALL_FORMAT {name} {(toJson table).compress}"
  IO.println s!"SMALL_FORMAT e8m0 {(toJson ((List.range 256).map e8m0ToBf16)).compress}"
