/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionProgram.Code.Funs
import Instruction.ComparisonSelectionSoundness

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction
namespace SegProgram

abbrev Parts := List U32 × List PtxvSim.seg.SOp × PtxvSim.seg.SVal

def comparatorView : PtxvSim.ptxv_ir.FCmp → PtxvIr.FCmp
  | .Eq => .Eq | .Ne => .Ne | .Lt => .Lt | .Le => .Le | .Gt => .Gt | .Ge => .Ge
  | .Num => .Num | .Nan => .Nan | .Equ => .Equ | .Neu => .Neu | .Ltu => .Ltu
  | .Leu => .Leu | .Gtu => .Gtu | .Geu => .Geu | .F => .F | .T => .T

def planView (plan : PtxvSim.ptxv_ir.comparison_select.Plan) : ComparisonSelection :=
  ⟨comparatorView plan.comparison, plan.negate, plan.swap⟩

def registerKey (register : PtxvSim.ptxv_ir.Reg) : Nat × Nat :=
  (match register.file with | .V => 0 | .R => 1 | .UR => 2 | .P => 3 | .UP => 4,
   register.idx.val)

def xreg : PtxvSim.ptxv_ir.Reg := ⟨.R, 0#u16⟩
def yreg : PtxvSim.ptxv_ir.Reg := ⟨.R, 1#u16⟩
def preg : PtxvSim.ptxv_ir.Reg := ⟨.P, 0#u16⟩
def dreg : PtxvSim.ptxv_ir.Reg := ⟨.R, 2#u16⟩

def sourceBinding (args prior : List U32) : PtxvSim.seg.SVal → Option U32
  | .Arg index => args[index.val]?
  | .Op index => prior[index.val]?
  | .Const _ => none

def registerBinding (args prior : List U32)
    (bindings : List (PtxvSim.ptxv_ir.Reg × PtxvSim.seg.SVal))
    (register : PtxvSim.ptxv_ir.Reg) : Option U32 := do
  let binding ← bindings.find? (fun b => registerKey b.1 == registerKey register)
  sourceBinding args prior binding.2

def operandBinding (args prior : List U32)
    (bindings : List (PtxvSim.ptxv_ir.Reg × PtxvSim.seg.SVal)) :
    PtxvSim.ptxv_ir.Val → Option U32
  | .Reg register _ => registerBinding args prior bindings register
  | .Imm _ => none
  | _ => none

def instructionResult (compare : PtxvIr.FCmp → U32 → U32 → Bool)
    (args prior : List U32) (operation : PtxvSim.seg.SOp) : Option U32 :=
  match operation.kind with
  | .Inst op _ bindings _ =>
    match op with
    | .Fsetp _ comparison a b _ _ _ _ _ => do
        let left ← operandBinding args prior bindings.val a
        let right ← operandBinding args prior bindings.val b
        some (if compare (comparatorView comparison) left right then 1#u32 else 0#u32)
    | .Sel _ a b predicate => do
        let raw ← match predicate.reg with
          | none => some true
          | some register => do
            let bits ← registerBinding args prior bindings.val register
            some (bits != 0#u32)
        operandBinding args prior bindings.val (if raw != predicate.neg then a else b)
    | _ => none
  | _ => none

def runOps (compare : PtxvIr.FCmp → U32 → U32 → Bool) (args : List U32) :
    List PtxvSim.seg.SOp → List U32 → Option (List U32)
  | [], prior => some prior
  | op :: rest, prior => do
      let bits ← instructionResult compare args prior op
      runOps compare args rest (prior ++ [bits])

def result (compare : PtxvIr.FCmp → U32 → U32 → Bool)
    (valuation : U32 → U32) (parts : Parts) : Option U32 := do
  let args := parts.1.map valuation
  let prior ← runOps compare args parts.2.1 []
  sourceBinding args prior parts.2.2

end SegProgram
end Instruction
