/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionBindings.Execution

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction.SegBinding

open SegProgram

def referenceValid (args prior : Nat) : PtxvSim.seg.SVal → Prop
  | .Arg index => index.val < args
  | .Const _ => True
  | .Op index => index.val < prior

def operandPresent (bindings : List Binding) : PtxvSim.ptxv_ir.Val → Prop
  | .Reg register _ => (lookup bindings register).isSome = true
  | .Imm _ => True
  | _ => False

def predicatePresent (bindings : List Binding) (predicate : PtxvSim.ptxv_ir.Pred) : Prop :=
  match predicate.reg with
  | none => True
  | some register => (lookup bindings register).isSome = true

def noMissingSources (bindings : List Binding) : PtxvSim.ptxv_ir.Op → Prop
  | .Fsetp _ _ a b _ _ condition _ _ =>
      operandPresent bindings a ∧ operandPresent bindings b ∧ predicatePresent bindings condition
  | .Sel _ a b condition =>
      operandPresent bindings a ∧ operandPresent bindings b ∧ predicatePresent bindings condition
  | _ => False

def instructionReady (args prior : Nat) (instruction : PtxvSim.seg.SOp) : Prop :=
  match instruction.kind with
  | .Inst operation _ bindings _ =>
    (∀ binding ∈ bindings.val, referenceValid args prior binding.2) ∧
      noMissingSources bindings.val operation
  | _ => False

def codeReady (args : Nat) : Nat → List PtxvSim.seg.SOp → Prop
  | _, [] => True
  | prior, instruction :: rest => instructionReady args prior instruction ∧ codeReady args (prior + 1) rest

def programReady (parts : Parts) : Prop :=
  codeReady parts.1.length 0 parts.2.1 ∧ referenceValid parts.1.length parts.2.1.length parts.2.2

@[step] theorem actual_constructor_no_defaults_or_bad_references
    (plan : PtxvSim.ptxv_ir.comparison_select.Plan) (left right comparisonPC selectionPC : U32) :
    PtxvSim.seg.comparison_program_parts ⟨plan, left, right, comparisonPC, selectionPC⟩
      ⦃ parts => programReady (observe parts) ⦄ := by
  step with actual_program_shape plan left right comparisonPC selectionPC as ⟨parts, hshape⟩
  rw [hshape]
  cases hswap : plan.swap <;>
    simp [programReady, codeReady, instructionReady, referenceValid, noMissingSources,
      operandPresent, predicatePresent, expected, opcodePair, pairVec, tripleVec,
      lookup, registerKey, xreg, yreg, preg, dreg, hswap]

end Instruction.SegBinding
