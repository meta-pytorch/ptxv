/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionDriver.Selected

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction.SegDriver

theorem source_scratch_independence {E : Type} (inst : PtxvSim.seg.program.ProgramEvaluator E)
    (program : PtxvSim.seg.program.Program) (first second : alloc.vec.Vec U64) (evaluator : E) :
    PtxvSim.seg.program.execute inst program first evaluator =
      PtxvSim.seg.program.execute inst program second evaluator := by
  rw [source_execute_exact, source_execute_exact]

theorem source_empty_no_callbacks {E : Type} (inst : PtxvSim.seg.program.ProgramEvaluator E)
    (program : PtxvSim.seg.program.Program) (values : alloc.vec.Vec U64) (evaluator : E)
    (empty : program.ops.val = []) :
    PtxvSim.seg.program.execute inst program values evaluator =
      finish program.args program.out emptyValues evaluator := by
  rw [source_execute_exact]
  simp only [expected, empty, interpret]

theorem source_first_error {E : Type} (inst : PtxvSim.seg.program.ProgramEvaluator E)
    (program : PtxvSim.seg.program.Program) (values : alloc.vec.Vec U64) (evaluator next : E)
    (first : PtxvSim.seg.SOp) (rest : List PtxvSim.seg.SOp) (error : String)
    (ops : program.ops.val = first :: rest)
    (callback : inst.evaluate evaluator first program.args emptyValues.slice = .ok (.Err error, next)) :
    PtxvSim.seg.program.execute inst program values evaluator = .ok (.Err error, emptyValues, next) := by
  rw [source_execute_exact]
  simp only [expected, ops, interpret, callback, bind_ok, uncurry_apply_pair]

theorem source_second_error {E : Type} (inst : PtxvSim.seg.program.ProgramEvaluator E)
    (program : PtxvSim.seg.program.Program) (values : alloc.vec.Vec U64) (evaluator middle next : E)
    (first second : PtxvSim.seg.SOp) (rest : List PtxvSim.seg.SOp) (bits : U64) (error : String)
    (ops : program.ops.val = first :: second :: rest)
    (initial : inst.evaluate evaluator first program.args emptyValues.slice = .ok (.Ok bits, middle))
    (following : inst.evaluate middle second program.args (oneValue bits).slice = .ok (.Err error, next)) :
    PtxvSim.seg.program.execute inst program values evaluator = .ok (.Err error, oneValue bits, next) := by
  rw [source_execute_exact]
  simp only [expected, ops, interpret, initial, bind_ok, source_push_empty, following, uncurry_apply_pair]

theorem source_first_abort {E : Type} (inst : PtxvSim.seg.program.ProgramEvaluator E)
    (program : PtxvSim.seg.program.Program) (values : alloc.vec.Vec U64) (evaluator : E)
    (first : PtxvSim.seg.SOp) (rest : List PtxvSim.seg.SOp) (error : Error)
    (ops : program.ops.val = first :: rest)
    (callback : inst.evaluate evaluator first program.args emptyValues.slice = .fail error) :
    PtxvSim.seg.program.execute inst program values evaluator = .fail error := by
  rw [source_execute_exact]
  simp only [expected, ops, interpret, callback, bind_fail]

theorem source_first_divergence {E : Type} (inst : PtxvSim.seg.program.ProgramEvaluator E)
    (program : PtxvSim.seg.program.Program) (values : alloc.vec.Vec U64) (evaluator : E)
    (first : PtxvSim.seg.SOp) (rest : List PtxvSim.seg.SOp)
    (ops : program.ops.val = first :: rest)
    (callback : inst.evaluate evaluator first program.args emptyValues.slice = .div) :
    PtxvSim.seg.program.execute inst program values evaluator = .div := by
  rw [source_execute_exact]
  simp only [expected, ops, interpret, callback, bind_div]

theorem source_empty_invalid_output {E : Type} (inst : PtxvSim.seg.program.ProgramEvaluator E)
    (program : PtxvSim.seg.program.Program) (values : alloc.vec.Vec U64) (evaluator : E)
    (empty : program.ops.val = [])
    (invalid : SegBinding.svalBits program.args.val [] program.out = none) :
    PtxvSim.seg.program.execute inst program values evaluator = .fail .arrayOutOfBounds := by
  rw [source_empty_no_callbacks inst program values evaluator empty]
  simp [finish, emptyValues, invalid, SegBinding.checked]

end Instruction.SegDriver
