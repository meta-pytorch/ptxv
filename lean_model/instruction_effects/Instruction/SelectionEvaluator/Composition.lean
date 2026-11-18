/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionEvaluator.Observations

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction.SegEvaluator

open SegProgram

def pairValue (left right : U32) : PtxvSim.ptxv_ir.Val → U32
  | .Reg register _ =>
      if registerKey register = registerKey xreg then left
      else if registerKey register = registerKey yreg then right else 0#u32
  | _ => 0#u32

def pairPredicate (answer : Bool) (predicate : PtxvSim.ptxv_ir.Pred) : Bool :=
  (match predicate.reg with
    | none => true
    | some register => if registerKey register = registerKey preg then answer else true) != predicate.neg

def pairCallbacks (compare : PtxvIr.FCmp → U32 → U32 → Bool)
    (left right : U32) (answer : Bool) : PtxvSim.seg.ComparisonEvaluator Unit where
  read state operand := .ok (pairValue left right operand, state)
  predicate state condition := .ok (pairPredicate answer condition, state)
  compare state request := .ok
    (compare (comparatorView request.comparison) request.left request.right, state)

def executesPair (compare : PtxvIr.FCmp → U32 → U32 → Bool)
    (plan : PtxvSim.ptxv_ir.comparison_select.Plan) (left right : U32)
    (pair : PtxvSim.ptxv_ir.Op × PtxvSim.ptxv_ir.Op) : Prop :=
  let answer := compare (comparatorView plan.comparison) left right
  PtxvSim.seg.comparison_instruction (pairCallbacks compare left right false) pair.1 () =
      .ok (some (if answer then 1#u32 else 0#u32), ()) ∧
  PtxvSim.seg.comparison_instruction (pairCallbacks compare left right answer) pair.2 () =
      .ok (some (comparisonSelectionBits compare (planView plan) left right), ())

theorem actual_pair_evaluator_all_bits (compare : PtxvIr.FCmp → U32 → U32 → Bool)
    (plan : PtxvSim.ptxv_ir.comparison_select.Plan) (left right : U32) :
    executesPair compare plan left right (opcodePair plan) := by
  unfold executesPair opcodePair
  constructor
  · rw [actual_evaluator_exact_callbacks]
    simp [expected, pairCallbacks, pairValue, pairPredicate, flags, modified, combine,
      registerKey, xreg, yreg, preg]
  · rw [actual_evaluator_exact_callbacks]
    cases hneg : plan.negate <;> cases hswap : plan.swap <;>
      cases hanswer : compare (comparatorView plan.comparison) left right <;>
      simp [expected, pairCallbacks, pairValue, pairPredicate, registerKey,
        xreg, yreg, preg, planView, comparisonSelectionBits, hneg, hswap, hanswer]

@[step] theorem actual_generated_pair_evaluates (compare : PtxvIr.FCmp → U32 → U32 → Bool)
    (plan : PtxvSim.ptxv_ir.comparison_select.Plan) (left right : U32) :
    PtxvSim.ptxv_ir.comparison_select.operations plan ⦃ pair =>
      executesPair compare plan left right pair ⦄ := by
  step with actual_opcode_pair plan as ⟨pair, hpair⟩
  rw [hpair]
  exact actual_pair_evaluator_all_bits compare plan left right


def producedPair : Parts → Option (PtxvSim.ptxv_ir.Op × PtxvSim.ptxv_ir.Op)
  | (_, [⟨_, .Inst comparison _ _ _⟩, ⟨_, .Inst selection _ _ _⟩], _) =>
      some (comparison, selection)
  | _ => none

@[step] theorem actual_constructed_program_evaluates
    (compare : PtxvIr.FCmp → U32 → U32 → Bool) (valuation : U32 → U32)
    (plan : PtxvSim.ptxv_ir.comparison_select.Plan) (left right comparisonPC selectionPC : U32) :
    PtxvSim.seg.comparison_program_parts ⟨plan, left, right, comparisonPC, selectionPC⟩
      ⦃ parts => ∃ pair, producedPair (observe parts) = some pair ∧
        executesPair compare plan (valuation left) (valuation right) pair ⦄ := by
  step with actual_program_shape plan left right comparisonPC selectionPC as ⟨parts, hparts⟩
  refine ⟨opcodePair plan, ?_, actual_pair_evaluator_all_bits compare plan (valuation left) (valuation right)⟩
  rw [hparts]
  rfl

@[step] theorem actual_op_constructor_evaluator_source_selection {R : Type}
    (readerInst : PtxvIr.comparison_select.NodeReader R) (reader : R)
    (nodeOf : PtxvIr.Val → Option U32) (hreader : NodeReaderContract readerInst reader nodeOf)
    (compare : PtxvIr.FCmp → U32 → U32 → Bool) (valuation : U32 → U32)
    (front : PtxvIr.Front) (comparison : PtxvIr.Op) (mode : PtxvIr.FCmp)
    (destination : PtxvIr.Reg) (first second : PtxvIr.Val) (predicate : PtxvIr.Pred)
    (left right comparisonPC selectionPC : U32) (hmode : plainComparison comparison = some mode) :
    (do
      let plan ← PtxvIr.comparison_select.from_pair readerInst front comparison
        (.Sel destination first second predicate) left right reader
      match plan with
      | none => .ok none
      | some plan =>
        let parts ← PtxvSim.seg.comparison_program_parts
          ⟨importPlan plan, left, right, comparisonPC, selectionPC⟩
        .ok (some (plan, parts))) ⦃ produced => match produced with
          | none => True
          | some (plan, parts) => ∃ pair,
            producedPair (observe parts) = some pair ∧
            executesPair compare (importPlan plan) (valuation left) (valuation right) pair ∧
            PtxvSim.seg.comparison_instruction
              (pairCallbacks compare (valuation left) (valuation right)
                (compare plan.comparison (valuation left) (valuation right))) pair.2 () =
              .ok (selectedNodeBits compare valuation mode predicate left right first second nodeOf, ()) ⦄ := by
  step with actual_selection_preserves_all_node_valuations readerInst reader nodeOf hreader compare
    front comparison mode destination first second predicate left right hmode as ⟨produced, hproduced⟩
  cases produced with
  | none => simp
  | some plan =>
    step with actual_constructed_program_evaluates compare valuation (importPlan plan)
      left right comparisonPC selectionPC as ⟨parts, pair, hpair, hexec⟩
    refine ⟨pair, hpair, hexec, ?_⟩
    have hbits : some (comparisonSelectionBits compare (planView (importPlan plan))
        (valuation left) (valuation right)) =
        selectedNodeBits compare valuation mode predicate left right first second nodeOf := by
      rw [imported_plan_view]
      exact hproduced valuation
    have hselection := hexec.2
    rw [hbits] at hselection
    simpa only [importPlan, comparator_view_import] using hselection

end Instruction.SegEvaluator
