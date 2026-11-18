/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionBindings.Reads

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction.SegDriver

abbrev Output (E : Type) := core.result.Result U32 String × alloc.vec.Vec U64 × E

def emptyValues : alloc.vec.Vec U64 := alloc.vec.Vec.from [] (by simp)

@[simp] theorem deref_eq_slice {α : Type} (values : alloc.vec.Vec α) :
    alloc.vec.Vec.deref values = values.slice := by
  apply Slice.ext
  simp only [alloc.vec.Vec.deref, Slice.from_val, alloc.vec.Vec.val]

def finish {E : Type} (args : Slice U32) (out : PtxvSim.seg.SVal)
    (values : alloc.vec.Vec U64) (evaluator : E) : Result (Output E) := do
  let bits ← SegBinding.checked (SegBinding.svalBits args.val values.val out)
  .ok (.Ok (UScalar.cast .U32 bits), values, evaluator)

def interpret {E : Type} (inst : PtxvSim.seg.program.ProgramEvaluator E)
    (args : Slice U32) (out : PtxvSim.seg.SVal) :
    List PtxvSim.seg.SOp → alloc.vec.Vec U64 → E → Result (Output E)
  | [], values, evaluator => finish args out values evaluator
  | instruction :: rest, values, evaluator => do
      let (answer, next) ← inst.evaluate evaluator instruction args values.slice
      match answer with
      | .Err error => .ok (.Err error, values, next)
      | .Ok bits =>
          let values ← alloc.vec.Vec.push values bits
          interpret inst args out rest values next

def expected {E : Type} (inst : PtxvSim.seg.program.ProgramEvaluator E)
    (program : PtxvSim.seg.program.Program) (evaluator : E) : Result (Output E) :=
  interpret inst program.args program.out program.ops.val emptyValues evaluator

theorem bind_continuation_congr {α β : Type} (computation : Result α)
    (first second : α → Result β) (same : ∀ value, first value = second value) :
    (do let value ← computation; first value) = (do let value ← computation; second value) := by
  exact congrArg (fun continuation => Std.bind computation continuation) (funext same)

theorem exact_result_exists_of_spec {α : Type} (computation : Result α) (post : α → Prop)
    (specification : computation ⦃ post ⦄) :
    ∃ value, computation = .ok value ∧ post value := by
  cases computation using Result.cases with
  | ret value => exact ⟨value, rfl, (WP.spec_ok value).mp specification⟩
  | vis effect continuation => exact False.elim ((WP.spec_vis effect continuation).mp specification)
  | div => exact False.elim (WP.spec_div.mp specification)

end Instruction.SegDriver
