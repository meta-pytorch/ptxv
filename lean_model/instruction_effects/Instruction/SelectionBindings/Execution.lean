/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionBindings.Wrappers

open Aeneas Aeneas.Std Result
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction.SegBinding

open SegProgram

abbrev NativeParts := alloc.vec.Vec U32 × alloc.vec.Vec PtxvSim.seg.SOp × PtxvSim.seg.SVal
abbrev Comparison := PtxvIr.FCmp → U32 → U32 → Bool → Bool

def arguments (valuation : U32 → U32) (nodes : alloc.vec.Vec U32) : Slice U32 :=
  Slice.from (nodes.val.map valuation) (by simp)

def emptyPrior : Slice U64 := Slice.from [] (by simp)
def onePrior (first : U64) : Slice U64 := Slice.from [first] (by simp; scalar_tac)
def twoPrior (first second : U64) : Slice U64 := Slice.from [first, second] (by simp; scalar_tac)

def callbacks (compare : Comparison) (args : Slice U32) (prior : Slice U64)
    (bindings : Slice Binding) : PtxvSim.seg.ComparisonEvaluator Unit where
  read state value := do
    match value with
    | .Reg register _ =>
        let bits ← PtxvSim.seg.binding.register_bits register bindings ⟨args, prior⟩
        .ok (UScalar.cast .U32 bits, state)
    | .Imm bits => .ok (UScalar.cast .U32 bits, state)
    | _ => .fail .undef
  predicate state condition := do
    let bits ← PtxvSim.seg.binding.read_predicate condition bindings ⟨args, prior⟩
    .ok (bits, state)
  compare state request := .ok
    (compare (comparatorView request.comparison) request.left request.right request.ftz, state)

def executeInstruction (compare : Comparison) (args : Slice U32) (prior : Slice U64)
    (instruction : PtxvSim.seg.SOp) : Result U64 := do
  match instruction.kind with
  | .Inst operation _ bindings _ =>
    let (result, _) ← PtxvSim.seg.comparison_instruction
      (callbacks compare args prior bindings.slice) operation ()
    match result with
    | none => .fail .undef
    | some bits => .ok (UScalar.cast .U64 bits)
  | _ => .fail .undef

def executeProducedPair (compare : Comparison) (valuation : U32 → U32)
    (parts : NativeParts) : Result U32 := do
  let args := arguments valuation parts.1
  match parts.2.1.val with
  | [first, second] =>
    let firstBits ← executeInstruction compare args emptyPrior first
    let secondBits ← executeInstruction compare args (onePrior firstBits) second
    let output ← PtxvSim.seg.binding.read_sval parts.2.2 args (twoPrior firstBits secondBits)
    .ok (UScalar.cast .U32 output)
  | _ => .fail .undef

@[simp] theorem binding_slice_val {α : Type} (bindings : alloc.vec.Vec α) :
    bindings.slice.val = bindings.val := rfl

@[simp] theorem cast32_roundtrip (bits : U32) :
    UScalar.cast .U32 (UScalar.cast .U64 bits) = bits := by
  apply UScalar.eq_of_val_eq
  scalar_tac

@[simp] theorem cast64_zero : UScalar.cast .U64 (0#u32) = 0#u64 := by
  apply UScalar.eq_of_val_eq
  scalar_tac

@[simp] theorem cast64_one : UScalar.cast .U64 (1#u32) = 1#u64 := by
  apply UScalar.eq_of_val_eq
  scalar_tac

theorem expected_program_actual_bindings (compare : Comparison) (valuation : U32 → U32)
    (plan : PtxvSim.ptxv_ir.comparison_select.Plan) (left right comparisonPC selectionPC : U32)
    (parts : NativeParts) (shape : observe parts = expected plan left right comparisonPC selectionPC) :
    executeProducedPair compare valuation parts = .ok
      (comparisonSelectionBits (fun mode a b => compare mode a b false)
        (planView plan) (valuation left) (valuation right)) := by
  have hargs := congrArg (fun p : Parts => p.1) shape
  have hops := congrArg (fun p : Parts => p.2.1) shape
  have hout := congrArg (fun p : Parts => p.2.2) shape
  dsimp only [observe, expected] at hargs hops hout
  unfold executeProducedPair
  rw [hops, hout]
  cases hneg : plan.negate <;> cases hswap : plan.swap <;>
    cases hanswer : compare (comparatorView plan.comparison) (valuation left) (valuation right) false <;>
    simp [executeInstruction, callbacks, SegEvaluator.actual_evaluator_exact_callbacks,
      SegEvaluator.expected, arguments, emptyPrior, onePrior, twoPrior,
      source_register_bits_exact, source_read_predicate_exact, source_read_sval_exact,
      registerBits, predicateBits, registerPredicateBits, predicateSource, lookup, svalBits, checked,
      pairVec, tripleVec, opcodePair, registerKey, xreg, yreg, preg, dreg,
      SegEvaluator.flags, SegEvaluator.modified, SegEvaluator.combine,
      planView, comparisonSelectionBits, hargs, hneg, hswap, hanswer]

@[step] theorem actual_constructor_source_execution (compare : Comparison) (valuation : U32 → U32)
    (plan : PtxvSim.ptxv_ir.comparison_select.Plan) (left right comparisonPC selectionPC : U32) :
    PtxvSim.seg.comparison_program_parts ⟨plan, left, right, comparisonPC, selectionPC⟩
      ⦃ parts => executeProducedPair compare valuation parts = .ok
        (comparisonSelectionBits (fun mode a b => compare mode a b false)
          (planView plan) (valuation left) (valuation right)) ⦄ := by
  step with actual_program_shape plan left right comparisonPC selectionPC as ⟨parts, hshape⟩
  exact expected_program_actual_bindings compare valuation plan left right comparisonPC selectionPC parts hshape

end Instruction.SegBinding
