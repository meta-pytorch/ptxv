/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.IEEEComparison.Primitive

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction.IEEEComparison

open SegProgram

def concreteEvaluator {F : Type} (fallback : F) : SegBound.Evaluator PtxvSim.seg.bound.IeeeComparison F :=
  ⟨(), fallback⟩

@[step] theorem actual_op_source_ieee_execution {R F : Type}
    (readerInst : PtxvIr.comparison_select.NodeReader R) (reader : R)
    (nodeOf : PtxvIr.Val → Option U32) (hreader : NodeReaderContract readerInst reader nodeOf)
    (fallbackInst : PtxvSim.seg.bound.FallbackEvaluator F) (fallback : F)
    (valuation : U32 → U32)
    (front : PtxvIr.Front) (comparison : PtxvIr.Op) (mode : PtxvIr.FCmp)
    (destination : PtxvIr.Reg) (first second : PtxvIr.Val) (predicate : PtxvIr.Pred)
    (left right comparisonPC selectionPC : U32) (values : alloc.vec.Vec U64)
    (hmode : plainComparison comparison = some mode) :
    (do
      let plan ← PtxvIr.comparison_select.from_pair readerInst front comparison
        (.Sel destination first second predicate) left right reader
      match plan with
      | none => .ok none
      | some plan =>
        let parts ← PtxvSim.seg.comparison_program_parts
          ⟨importPlan plan, left, right, comparisonPC, selectionPC⟩
        let (answer, scratch, evaluator) ← SegBound.executeBound primitive primitiveCopy
          fallbackInst (concreteEvaluator fallback) valuation parts values
        .ok (some (answer, scratch, evaluator))) ⦃ produced => match produced with
          | none => True
          | some (answer, _, evaluator) => ∃ bits, answer = .Ok bits ∧
            evaluator = concreteEvaluator fallback ∧
            some bits = selectedNodeBits (fun mode a b => oracle mode a b false)
              valuation mode predicate left right first second nodeOf ⦄ := by
  convert SegBound.actual_op_bound_evaluator_execution readerInst reader nodeOf hreader
    primitive primitiveCopy fallbackInst (concreteEvaluator fallback) oracle concrete_primitive_contract
    valuation front comparison mode destination first second predicate left right comparisonPC selectionPC values hmode using 1
  funext produced
  cases produced with
  | none => rfl
  | some payload =>
    rcases payload with ⟨answer, scratch, evaluator⟩
    rfl

end Instruction.IEEEComparison
