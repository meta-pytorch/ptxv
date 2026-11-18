/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.ControlFlowRows

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 8000000
set_option maxRecDepth 10000

namespace Instruction.ControlFlow

@[step]
theorem rust_successors (operation : Op) (guard : Pred) (context : control_flow.Context)
    (space : (successors operation guard (contextView context)).length ≤ Usize.max) :
    control_flow.successors operation guard context ⦃ result =>
      RowPost (successors operation guard (contextView context)) result ⦄ := by
  unfold control_flow.successors
  rw [rust_pred_always]
  simp only [bind_ok]
  cases operation <;> simp only [successors] at space ⊢
  all_goals try exact rust_next context
  case Call target => exact rust_target_next target context
  case Bra target any tid0 all condition divergent =>
    cases condition <;> cases divergent <;> cases any <;> cases tid0 <;> cases all <;>
      cases observed : always guard <;> cases folded : context.fold <;>
      simp only [folded, core.option.Option.is_none, Option.isNone_none, Option.isNone_some,
        Bool.not_false, Bool.not_true, Bool.false_and, Bool.true_and,
        Bool.false_eq_true, ite_false, ite_true, contextView]
    all_goals try first
      | exact rust_target target
      | exact rust_target_next target context
    all_goals split <;> simp_all only []
    all_goals first
      | exact rust_next context
      | exact rust_target target
  case Brx base offset targets =>
    simpa only [alloc.vec.Vec.deref, Slice.from_val, List.length_map, List.length_append] using
      rust_copy_optional_next (alloc.vec.Vec.deref targets) context (always guard)
        (by simpa only [alloc.vec.Vec.deref, Slice.from_val, List.length_map, List.length_append] using space)
  case Ret address =>
    cases observed : context.returns with
    | none =>
        have capacity : context.count.val + (if always guard then [] else next (contextView context)).length ≤ Usize.max := by
          simpa only [returnTargets, contextView, observed, Option.map_none, Option.getD_none,
            List.length_append, List.length_range] using space
        simpa only [returnTargets, contextView, observed, Option.map_none, Option.getD_none] using
          rust_range_optional_next context (always guard) capacity
    | some values =>
        have capacity : values.val.length + (if always guard then [] else next (contextView context)).length ≤ Usize.max := by
          simpa only [returnTargets, contextView, observed, Option.map_some, Option.getD_some,
            List.length_append, List.length_map] using space
        simpa only [returnTargets, contextView, observed, Option.map_some, Option.getD_some] using
          rust_copy_optional_next values context (always guard) capacity
  case Exit | Trap =>
    simpa only [alloc.vec.Vec.new, alloc.vec.Vec.from_val, List.map_nil, List.nil_append] using
      rust_optional_next (alloc.vec.Vec.new Usize) context (always guard) (by
        simpa only [alloc.vec.Vec.new, alloc.vec.Vec.from_val, List.length_nil, Nat.zero_add] using space)
  case ClusterWait =>
    simp only [lift, bind_ok]
    by_cases cut : context.cut &&& 1#u8 = 0#u8
    · simpa only [contextView, cut, bne_self_eq_false, Bool.false_eq_true, ite_false,
        ne_eq, not_true_eq_false] using rust_next context
    · have test : ((context.cut &&& 1#u8) != 0#u8) = true := by simpa only [bne_iff_ne] using cut
      simp only [test, ite_true, contextView, cut, ne_eq, not_false_eq_true, WP.spec_ok, RowPost,
        alloc.vec.Vec.new, alloc.vec.Vec.from_val, List.map_nil]
  case BarSync id count arrive aligned blocking reduction =>
    cases count <;> cases arrive <;> simp only
    all_goals try exact rust_next context
    simp only [lift, bind_ok]
    by_cases cut : context.cut &&& 2#u8 = 0#u8 <;> cases observed : always guard <;>
      simp only [contextView, cut, bne_iff_ne, ite_self, bne_self_eq_false, Bool.false_eq_true, ite_false,
        ite_true, ne_eq, not_true_eq_false, not_false_eq_true, decide_true, decide_false,
        Bool.false_and, Bool.true_and, WP.spec_ok, RowPost,
        alloc.vec.Vec.new, alloc.vec.Vec.from_val, List.map_nil]
    all_goals exact rust_next context

end Instruction.ControlFlow
