/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.Mufu

/- Empirical native SM103 MUFU semantics. PTX wrappers may add conversions or
argument reduction. Corpus and compiler provenance: results/blackwell/scalar/gb300-mufu-formats.md. -/
namespace GB300Math.MufuSm103

def architecture : Nat := 103

def operations : List String :=
  ["ex2", "rcp", "rsq", "sqrt", "lg2", "sin", "cos", "tanh"]

def evaluateBf16 (operation : String) (input : Nat) : Option Nat := do
  if !operations.contains operation then none
  else
    let y ← Mufu.evaluate32 operation (input % 2 ^ 16 * 2 ^ 16)
    return y / 2 ^ 16

def evaluate (operation : String) (input : Nat) : Option Nat :=
  if operation.endsWith ".bf16" then
    evaluateBf16 (operation.dropEnd 5).toString input
  else if operation.endsWith ".f16" then
    if operations.contains (operation.dropEnd 4).toString then
      Mufu.evaluate operation input
    else none
  else if operations.contains operation then Mufu.evaluate32 operation input
  else none

theorem bf16_upper_bits_ignored (operation : String) (input upper : Nat) :
    evaluateBf16 operation (input + upper * 2 ^ 16) =
      evaluateBf16 operation input := by
  simp [evaluateBf16, Nat.add_mod]

theorem tanh_bf16_truncates :
    evaluate "tanh.bf16" 0x3b81 = some 0x3b80 ∧
    let y := (Mufu.evaluate32 "tanh" 0x3b810000).getD 0
    (y + 0x7fff + y / 2 ^ 16 % 2) / 2 ^ 16 = 0x3b81 := by
  decide +kernel

theorem unsupported_operation : evaluate "tanh16.bf16" 0 = none := by
  decide +kernel

#print axioms bf16_upper_bits_ignored
#print axioms tanh_bf16_truncates
#print axioms unsupported_operation

end GB300Math.MufuSm103
