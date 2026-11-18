/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionEvaluator.Refinement

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction.SegEvaluator

inductive Observation where
  | read : PtxvSim.ptxv_ir.Val → Observation
  | predicate : PtxvSim.ptxv_ir.Pred → Observation
  | compare : PtxvSim.ptxv_ir.FCmp → U32 → U32 → Bool → Observation

def observedCallbacks
    (value : PtxvSim.ptxv_ir.Val → U32)
    (predicate : PtxvSim.ptxv_ir.Pred → Bool)
    (compare : PtxvSim.ptxv_ir.FCmp → U32 → U32 → Bool → Bool) :
    PtxvSim.seg.ComparisonEvaluator (List Observation) where
  read trace operand := .ok (value operand, trace ++ [.read operand])
  predicate trace condition := .ok (predicate condition, trace ++ [.predicate condition])
  compare trace request := .ok (compare request.comparison request.left request.right request.ftz,
    trace ++ [.compare request.comparison request.left request.right request.ftz])

theorem actual_comparison_observations
    (value : PtxvSim.ptxv_ir.Val → U32)
    (predicate : PtxvSim.ptxv_ir.Pred → Bool)
    (compare : PtxvSim.ptxv_ir.FCmp → U32 → U32 → Bool → Bool)
    (destination : PtxvSim.ptxv_ir.Reg) (comparison : PtxvSim.ptxv_ir.FCmp)
    (a b : PtxvSim.ptxv_ir.Val) (negative absolute : Array Bool 2#usize)
    (condition : PtxvSim.ptxv_ir.Pred) (boolean : PtxvSim.ptxv_ir.BoolOp)
    (ftz : Bool) (trace : List Observation) :
    let left := modified (value a) (flags negative 0) (flags absolute 0)
    let right := modified (value b) (flags negative 1) (flags absolute 1)
    PtxvSim.seg.comparison_instruction (observedCallbacks value predicate compare)
      (.Fsetp destination comparison a b negative absolute condition boolean ftz) trace =
    .ok (some (if combine boolean (compare comparison left right ftz) (predicate condition)
      then 1#u32 else 0#u32),
      trace ++ [.read a, .read b, .compare comparison left right ftz, .predicate condition]) := by
  dsimp only
  rw [actual_evaluator_exact_callbacks]
  simp [expected, observedCallbacks, List.append_assoc]

theorem actual_selection_observations
    (value : PtxvSim.ptxv_ir.Val → U32)
    (predicate : PtxvSim.ptxv_ir.Pred → Bool)
    (compare : PtxvSim.ptxv_ir.FCmp → U32 → U32 → Bool → Bool)
    (destination : PtxvSim.ptxv_ir.Reg) (a b : PtxvSim.ptxv_ir.Val)
    (condition : PtxvSim.ptxv_ir.Pred) (trace : List Observation) :
    PtxvSim.seg.comparison_instruction (observedCallbacks value predicate compare)
      (.Sel destination a b condition) trace =
    .ok (some (value (if predicate condition then a else b)),
      trace ++ [.predicate condition, .read (if predicate condition then a else b)]) := by
  rw [actual_evaluator_exact_callbacks]
  simp [expected, observedCallbacks, List.append_assoc]

def selected : IR → Bool
  | .Fsetp .. | .Sel .. => true
  | _ => false

theorem actual_unsupported_has_no_callbacks {E : Type}
    (callbacks : PtxvSim.seg.ComparisonEvaluator E) (operation : IR) (initial : E)
    (outside : selected operation = false) :
    PtxvSim.seg.comparison_instruction callbacks operation initial = .ok (none, initial) := by
  rw [actual_evaluator_exact_callbacks]
  cases operation <;> simp_all [selected, expected]

end Instruction.SegEvaluator
