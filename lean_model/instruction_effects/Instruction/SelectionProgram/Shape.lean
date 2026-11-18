/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionProgram.Model

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction.SegProgram

def pairVec {α : Type} (a b : α) : alloc.vec.Vec α :=
  alloc.vec.Vec.from [a, b] (by simp; scalar_tac)
def tripleVec {α : Type} (a b c : α) : alloc.vec.Vec α :=
  alloc.vec.Vec.from [a, b, c] (by simp; scalar_tac)

def opcodePair (plan : PtxvSim.ptxv_ir.comparison_select.Plan) :
    PtxvSim.ptxv_ir.Op × PtxvSim.ptxv_ir.Op :=
  (.Fsetp preg plan.comparison (.Reg xreg 1#u8) (.Reg yreg 1#u8)
    (Array.repeat 2#usize false) (Array.repeat 2#usize false)
    ⟨none, false⟩ .And false,
   .Sel dreg (if plan.swap then .Reg yreg 1#u8 else .Reg xreg 1#u8)
     (if plan.swap then .Reg xreg 1#u8 else .Reg yreg 1#u8)
     ⟨some preg, plan.negate⟩)

def expected (plan : PtxvSim.ptxv_ir.comparison_select.Plan)
    (x y comparisonPC selectionPC : U32) : Parts :=
  ([x, y], [
    ⟨comparisonPC, .Inst (opcodePair plan).1 preg
      (pairVec (xreg, .Arg 0#u16) (yreg, .Arg 1#u16)) (alloc.vec.Vec.new _)⟩,
    ⟨selectionPC, .Inst (opcodePair plan).2 dreg
      (tripleVec (xreg, .Arg 0#u16) (yreg, .Arg 1#u16) (preg, .Op 0#u32))
      (alloc.vec.Vec.new _)⟩], .Op 1#u32)

def observe (parts : alloc.vec.Vec U32 × alloc.vec.Vec PtxvSim.seg.SOp × PtxvSim.seg.SVal) :
    Parts := (parts.1.val, parts.2.1.val, parts.2.2)

@[step] theorem actual_opcode_pair (plan : PtxvSim.ptxv_ir.comparison_select.Plan) :
    PtxvSim.ptxv_ir.comparison_select.operations plan ⦃ pair => pair = opcodePair plan ⦄ := by
  cases h : plan.swap <;>
    simp [PtxvSim.ptxv_ir.comparison_select.operations, opcodePair, xreg, yreg, preg, dreg, h]

@[step] theorem actual_program_shape (plan : PtxvSim.ptxv_ir.comparison_select.Plan)
    (x y comparisonPC selectionPC : U32) :
    PtxvSim.seg.comparison_program_parts ⟨plan, x, y, comparisonPC, selectionPC⟩ ⦃ parts =>
      observe parts = expected plan x y comparisonPC selectionPC ⦄ := by
  unfold PtxvSim.seg.comparison_program_parts
  step with actual_opcode_pair plan as ⟨comparison, selection, hpair⟩
  have hcomparison := congrArg Prod.fst hpair
  have hselection := congrArg Prod.snd hpair
  dsimp only at hcomparison hselection
  subst comparison selection
  have hsmall1 : 1 ≤ U32.max := by scalar_tac
  have hlt1 : 1 < U32.max := by scalar_tac
  have hlt2 : 2 < U32.max := by scalar_tac
  simp [alloc.vec.Vec.with_capacity, alloc.vec.Vec.new, alloc.vec.Vec.push,
    observe, expected, pairVec, tripleVec,
    xreg, yreg, preg, dreg, hsmall1, hlt1, hlt2]

end Instruction.SegProgram
