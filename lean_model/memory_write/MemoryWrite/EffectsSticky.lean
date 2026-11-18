/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.EffectsInvalidation
import MemoryWrite.EffectsFold

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 4000000

namespace MemoryWrite.ActionEffects

theorem source_apply_invalidation_commutes
    (state : effects.FoldState) (event : effects.Effect) :
    effects.apply { state with invalidated := true } event = (do
      let next ← effects.apply state event
      .ok { next with invalidated := true }) := by
  cases event with
  | Access resource kind visibility =>
    unfold effects.apply
    simp only [Std.bind_assoc]
    congr 1
    funext poison
    congr 1
    funext visible
    cases visible with
    | false => simp
    | true =>
      simp only [ite_true, source_resource_range, bind_ok]
      cases kind with
      | Read =>
        simp only [Std.bind_assoc, bind_ok, Bool.true_or]
      | Write =>
        simp only [Std.bind_assoc, bind_ok, Bool.true_or]
  | Release key =>
    simp only [effects.apply, Std.bind_assoc, bind_ok]
  | Acquire key =>
    simp only [effects.apply, Std.bind_assoc, bind_ok]
  | Arrival key arrival => exact record_arrival_invalidation_commutes state key arrival
  | Finding => simp only [source_finding, bind_ok]
  | Fault => simp only [source_fault, bind_ok]

theorem sequence_invalidation_commutes (trace : List effects.Effect)
    (state : effects.FoldState) :
    sequence trace { state with invalidated := true } = (do
      let next ← sequence trace state
      .ok { next with invalidated := true }) := by
  induction trace generalizing state with
  | nil => simp only [sequence, bind_ok]
  | cons event rest ih =>
    simp only [sequence, source_apply_invalidation_commutes, Std.bind_assoc, bind_ok, ih]

theorem source_fold_invalidation_commutes (state : effects.FoldState)
    (trace : Slice effects.Effect) :
    effects.fold_from { state with invalidated := true } trace = (do
      let next ← effects.fold_from state trace
      .ok { next with invalidated := true }) := by
  simp only [source_fold_sequence, sequence_invalidation_commutes]

theorem source_finding_survives_suffix (state : effects.FoldState)
    (prior suffix combined : Slice effects.Effect)
    (events : combined.val = prior.val ++ .Finding :: suffix.val) :
    effects.fold_from state combined = (do
      let beforeFinding ← effects.fold_from state prior
      let afterSuffix ← effects.fold_from beforeFinding suffix
      .ok { afterSuffix with invalidated := true }) := by
  simp [source_fold_sequence, events, sequence_append, sequence, source_finding,
    sequence_invalidation_commutes]

end MemoryWrite.ActionEffects
