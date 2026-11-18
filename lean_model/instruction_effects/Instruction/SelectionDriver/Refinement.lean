/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionDriver.Model

open Aeneas Aeneas.Std Result
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction.SegDriver

theorem source_loop_exact {E : Type} (inst : PtxvSim.seg.program.ProgramEvaluator E)
    (args : Slice U32) (ops : Slice PtxvSim.seg.SOp) (out : PtxvSim.seg.SVal)
    (values : alloc.vec.Vec U64) (evaluator : E) (index : Usize)
    (bounded : index.val ≤ ops.val.length) :
    PtxvSim.seg.program.execute_loop inst args ops out values evaluator index =
      interpret inst args out (ops.val.drop index.val) values evaluator := by
  have auxiliary : ∀ remaining : Nat, ∀ index : Usize,
      ops.val.length - index.val = remaining → index.val ≤ ops.val.length →
      ∀ (values : alloc.vec.Vec U64) (evaluator : E),
      PtxvSim.seg.program.execute_loop inst args ops out values evaluator index =
        interpret inst args out (ops.val.drop index.val) values evaluator := by
    intro remaining
    induction remaining using Nat.strong_induction_on with
    | h remaining induction =>
      intro index hremaining hbound values evaluator
      unfold PtxvSim.seg.program.execute_loop
      rw [loop]
      unfold PtxvSim.seg.program.execute_loop.body
      by_cases within : index.val < ops.val.length
      · have htest : index < Slice.len ops := by scalar_tac
        simp only [htest, ite_true]
        have hread : Slice.index_usize ops index = .ok ops.val[index.val] :=
          SegBinding.exact_eq_of_spec _ _ (Slice.index_usize_spec ops index within)
        rw [hread, List.drop_eq_getElem_cons within]
        simp only [bind_ok, interpret, Std.bind_assoc, deref_eq_slice]
        apply bind_continuation_congr
        intro answer
        rcases answer with ⟨answer, next⟩
        cases answer with
        | Err error => simp
        | Ok bits =>
          simp only [uncurry_apply_pair, Std.bind_assoc, bind_ok]
          apply bind_continuation_congr
          intro nextValues
          have hplus : (index + 1#usize : Result Usize) ⦃ next => next.val = index.val + 1 ⦄ := by
            step; scalar_tac
          obtain ⟨nextIndex, hnextIndex, hnextVal⟩ := exact_result_exists_of_spec _ _ hplus
          rw [hnextIndex]
          simp only [bind_ok]
          simpa only [hnextVal, PtxvSim.seg.program.execute_loop,
            PtxvSim.seg.program.execute_loop.body, deref_eq_slice, uncurry] using induction (ops.val.length - nextIndex.val)
            (by omega) nextIndex rfl (by omega) nextValues next
      · have htest : ¬ index < Slice.len ops := by scalar_tac
        have hdrop : ops.val.drop index.val = [] := List.drop_eq_nil_iff.mpr (by omega)
        simp only [htest, ite_false, hdrop, interpret, finish,
          SegBinding.source_read_sval_exact, deref_eq_slice, Std.bind_assoc, lift, bind_ok]
        rfl
  exact auxiliary (ops.val.length - index.val) index rfl bounded values evaluator

theorem source_execute_exact {E : Type} (inst : PtxvSim.seg.program.ProgramEvaluator E)
    (program : PtxvSim.seg.program.Program) (values : alloc.vec.Vec U64) (evaluator : E) :
    PtxvSim.seg.program.execute inst program values evaluator = expected inst program evaluator := by
  unfold PtxvSim.seg.program.execute
  simp only [alloc.vec.Vec.clear, bind_ok]
  rw [source_loop_exact inst program.args program.ops program.out _ evaluator 0#usize (by simp)]
  rfl

end Instruction.SegDriver
