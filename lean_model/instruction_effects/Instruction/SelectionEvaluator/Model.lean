/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionProgram.Composition

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction.SegEvaluator

abbrev IR := PtxvSim.ptxv_ir.Op

def modified (bits : U32) (negative absolute : Bool) : U32 :=
  (if absolute then bits &&& 0x7fffffff#u32 else bits) ^^^
    (if negative then 0x80000000#u32 else 0#u32)

@[simp] theorem xor_zero (bits : U32) : bits ^^^ 0#u32 = bits := by
  apply UScalar.eq_of_val_eq
  simp

def combine (operation : PtxvSim.ptxv_ir.BoolOp) (comparison condition : Bool) : Bool :=
  match operation with
  | .And => comparison && condition
  | .Or => comparison || condition
  | .Xor => comparison != condition

def flags (array : Array Bool 2#usize) (index : Nat) : Bool :=
  array.val[index]?.getD false

def expected {E : Type} (callbacks : PtxvSim.seg.ComparisonEvaluator E)
    (operation : IR) (initial : E) : Result (Option U32 × E) := do
  match operation with
  | .Fsetp _ comparison a b negative absolute condition boolean ftz =>
    let (left, afterLeft) ← callbacks.read initial a
    let (right, afterRight) ← callbacks.read afterLeft b
    let (answer, afterComparison) ← callbacks.compare afterRight
      ⟨comparison, modified left (flags negative 0) (flags absolute 0),
        modified right (flags negative 1) (flags absolute 1), ftz⟩
    let (predicate, final) ← callbacks.predicate afterComparison condition
    .ok (some (if combine boolean answer predicate then 1#u32 else 0#u32), final)
  | .Sel _ a b predicate =>
    let (selected, afterPredicate) ← callbacks.predicate initial predicate
    let (bits, final) ← callbacks.read afterPredicate (if selected then a else b)
    .ok (some bits, final)
  | _ => .ok (none, initial)

end Instruction.SegEvaluator
