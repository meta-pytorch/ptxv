/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionDriver.Composition

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction.SegBound

abbrev Comparison := SegBinding.Comparison
abbrev Sources (C : Type) := PtxvSim.seg.bound.Sources C
abbrev Evaluator (C F : Type) := PtxvSim.seg.bound.Evaluator C F
abbrev Inputs := PtxvSim.seg.binding.Inputs

def directValue : PtxvSim.ptxv_ir.Val → Bool
  | .Reg _ _ | .Imm _ => true
  | _ => false

def eligibleOperation : PtxvSim.ptxv_ir.Op → Bool
  | .Fsetp _ _ a b _ _ _ _ _ | .Sel _ a b _ => directValue a && directValue b
  | _ => false

def eligibleInstruction (instruction : PtxvSim.seg.SOp) : Bool :=
  match instruction.kind with
  | .Inst operation _ _ _ => eligibleOperation operation
  | _ => false

def PrimitiveContract {C : Type} (inst : PtxvSim.seg.bound.ComparisonPrimitive C)
    (primitive : C) (compare : Comparison) : Prop :=
  ∀ request, inst.compare primitive request = .ok
    (compare (SegProgram.comparatorView request.comparison) request.left request.right request.ftz)

def valueBits (inputs : Inputs) (bindings : Slice SegBinding.Binding)
    (value : PtxvSim.ptxv_ir.Val) : Result U32 :=
  match value with
  | .Reg register _ => do
      let bits ← SegBinding.checked
        (SegBinding.registerBits register bindings.val inputs.args.val inputs.prior.val)
      .ok (UScalar.cast .U32 bits)
  | .Imm bits => .ok (UScalar.cast .U32 bits)
  | _ => .fail .undef

def predicateBits (inputs : Inputs) (bindings : Slice SegBinding.Binding)
    (predicate : PtxvSim.ptxv_ir.Pred) : Result Bool :=
  SegBinding.checked
    (SegBinding.predicateBits predicate bindings.val inputs.args.val inputs.prior.val)

end Instruction.SegBound
