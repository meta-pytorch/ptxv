/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.EffectsComposition

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 4000000

namespace MemoryWrite.ActionEffects

theorem source_loop_sequence (trace : Slice effects.Effect)
    (state : effects.FoldState) (index : Usize) :
    effects.fold_from_loop state trace index = sequence (trace.val.drop index.val) state := by
  generalize fuelEq : trace.val.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing state index with
  | h fuel ih =>
    have stepEq : effects.fold_from_loop state trace index = (do
        let result ← effects.fold_from_loop.body trace state index
        match result with
        | .cont pair => effects.fold_from_loop pair.1 trace pair.2
        | .done output => .ok output) := by
      unfold effects.fold_from_loop
      rw [loop.eq_def]
      congr 1
      funext result
      cases result with
      | cont pair => rcases pair with ⟨after, next⟩; rfl
      | done output => rfl
    rw [stepEq]
    by_cases inside : index.val < trace.val.length
    · have bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := Slice.length_ineq trace
        scalar_tac
      let next : Usize := Usize.ofNatCore (index.val + 1) bound
      have nextVal : next.val = index.val + 1 := by simp [next]
      have platformBound : index.val + 1 < 2 ^ System.Platform.numBits := bound
      have increment : (index + 1#usize : Result Usize) = .ok next := by
        simp [HAdd.hAdd, UScalar.add, UScalar.tryMk, UScalar.tryMkOpt,
          UScalar.check_bounds, platformBound, next, Result.ofOption]
        rfl
      have lookup : Slice.index_usize trace index = .ok trace.val[index.val] := by
        simp [Slice.index_usize, Slice.getElem?_Usize_eq,
          List.getElem?_eq_getElem inside]
      have test : index < Slice.len trace := by scalar_tac
      have body : effects.fold_from_loop.body trace state index = (do
          let after ← effects.apply state trace.val[index.val]
          .ok (.cont (after, next))) := by
        simp only [effects.fold_from_loop.body, test, ite_true, lookup, bind_ok]
        simp only [increment, bind_ok]
        rfl
      rw [body]
      simp only [Std.bind_assoc, bind_ok]
      rw [List.drop_eq_getElem_cons inside]
      simp only [sequence]
      congr 1
      funext after
      apply ih (trace.val.length - next.val) (by omega) after next rfl
    · have test : ¬index < Slice.len trace := by scalar_tac
      simp [effects.fold_from_loop.body, test, sequence,
        List.drop_eq_nil_of_le (show trace.val.length ≤ index.val by omega)]

theorem source_fold_sequence (state : effects.FoldState) (trace : Slice effects.Effect) :
    effects.fold_from state trace = sequence trace.val state := by
  simpa [effects.fold_from] using source_loop_sequence trace state 0#usize

theorem source_fold_append (state : effects.FoldState)
    (first second combined : Slice effects.Effect)
    (combinedEvents : combined.val = first.val ++ second.val) :
    effects.fold_from state combined = (do
      let middle ← effects.fold_from state first
      effects.fold_from middle second) := by
  simp only [source_fold_sequence, combinedEvents, sequence_append]

theorem source_fault_after_prefix (state : effects.FoldState)
    (prior faulted : Slice effects.Effect)
    (events : faulted.val = prior.val ++ [.Fault]) :
    effects.fold_from state faulted = (do
      let beforeFault ← effects.fold_from state prior
      .ok { beforeFault with
        summary := { beforeFault.summary with all := true }, invalidated := true }) := by
  simp [source_fold_sequence, events, sequence_append, sequence, source_fault]

theorem source_finding_after_prefix (state : effects.FoldState)
    (prior reported : Slice effects.Effect)
    (events : reported.val = prior.val ++ [.Finding]) :
    effects.fold_from state reported = (do
      let beforeFinding ← effects.fold_from state prior
      .ok { beforeFinding with invalidated := true }) := by
  simp [source_fold_sequence, events, sequence_append, sequence, source_finding]

theorem fault_keeps_successful_prefix (state beforeFault : effects.FoldState)
    (prior faulted : Slice effects.Effect)
    (events : faulted.val = prior.val ++ [.Fault])
    (prefixSuccess : effects.fold_from state prior = .ok beforeFault) :
    effects.fold_from state faulted = .ok {
      beforeFault with summary := { beforeFault.summary with all := true }, invalidated := true
    } := by
  simp [source_fault_after_prefix state prior faulted events, prefixSuccess]

theorem fault_keeps_prefix_failure (state : effects.FoldState) (error : Error)
    (prior faulted : Slice effects.Effect)
    (events : faulted.val = prior.val ++ [.Fault])
    (prefixFailure : effects.fold_from state prior = .fail error) :
    effects.fold_from state faulted = .fail error := by
  simp [source_fault_after_prefix state prior faulted events, prefixFailure]

end MemoryWrite.ActionEffects
