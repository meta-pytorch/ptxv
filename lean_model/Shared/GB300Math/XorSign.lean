/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.XorSign

structure Format where
  width : Nat
  fraction : Nat
  deriving Repr

def format (name : String) : Option Format :=
  match name with
  | "f16" | "f16x2" => some ⟨16, 10⟩
  | "bf16" | "bf16x2" => some ⟨16, 7⟩
  | "f32" => some ⟨32, 23⟩
  | _ => none

def magnitude (f : Format) (ftz : Bool) (a : Nat) : Nat :=
  let value := a % 2^(f.width-1)
  if ftz && value < 2^f.fraction then 0 else value

def lane (f : Format) (maximum propagate ftz : Bool) (a b : Nat) : Nat :=
  let sign := 2^(f.width-1)
  let infinity := sign-2^f.fraction
  let x := magnitude f ftz a
  let y := magnitude f ftz b
  let nx := x > infinity
  let ny := y > infinity
  if (propagate && (nx || ny)) || (nx && ny) then sign-1
  else
    let result := if nx then y else if ny then x else if maximum then max x y else min x y
    result + (((a/sign)%2 + (b/sign)%2)%2)*sign

def evaluate (precision op : String) (ftz propagate : Bool) (a b : Nat) : Option Nat := do
  let f ← format precision
  unless op = "min" || op = "max" do none
  if (precision = "bf16" || precision = "bf16x2") && ftz then none else do
    let lo := lane f (op = "max") propagate ftz a b
    if precision = "f16x2" || precision = "bf16x2" then
      return lo + 65536*lane f (op = "max") propagate ftz (a/65536) (b/65536)
    return lo

theorem negative_zero_preserved :
    evaluate "f32" "min" false false 0x80000000 0x3f800000 = some 0x80000000 := by decide

theorem two_negative_inputs_positive :
    evaluate "f16" "max" false false 0xbc00 0xc000 = some 0x4000 := by decide

theorem discarded_nan_sign_participates :
    evaluate "bf16" "min" false false 0xffff 0x3f80 = some 0xbf80 := by decide

theorem nan_output_ignores_xor_sign :
    evaluate "bf16" "min" false true 0xffff 0x3f80 = some 0x7fff := by decide

theorem packed_signs_are_separate :
    evaluate "f16x2" "min" false false 0xbc003c00 0x4000c000 = some 0xbc00bc00 := by decide

#print axioms negative_zero_preserved
#print axioms two_negative_inputs_positive
#print axioms discarded_nan_sign_participates
#print axioms nan_output_ignores_xor_sign
#print axioms packed_signs_are_separate

end GB300Math.XorSign
