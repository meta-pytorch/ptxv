/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MetadataPlanModel
import MemoryWrite.Code.Funs

open Aeneas Aeneas.Std Result PtxvMemoryWrite
set_option autoImplicit false

namespace MemoryWrite.Symbolic

def destinationPlanView (plan : symbolic.DestinationPlan) : DestinationPlan :=
  { touched := plan.touched
    clearHistory := plan.clear_history
    forgetHalf := plan.forget_half
    clearBits := plan.clear_bits
    activeTaint := plan.active_taint
    forceTaint := plan.force_taint }

@[step] theorem destination_plan_spec (active dependent hookWrote sourceTainted : Bool) :
    symbolic.destination_plan active dependent hookWrote sourceTainted ⦃ result =>
      destinationPlanView result = destinationPlan active dependent hookWrote sourceTainted ⦄ := by
  cases active <;> cases dependent <;> cases hookWrote <;>
    simp [symbolic.destination_plan, destinationPlanView, destinationPlan, WP.spec_ok]

theorem actual_destination_plan_refines_assignment (active dependent hookWrote sourceTainted : Bool)
    (source old : Metadata) (htaint : source.tainted = sourceTainted)
    (hhistory : hookWrote = false → source.history = none) :
    symbolic.destination_plan active dependent hookWrote sourceTainted ⦃ result =>
      applyDestinationPlan (destinationPlanView result) source old =
        assignMetadata active dependent source old ⦄ := by
  step with destination_plan_spec active dependent hookWrote sourceTainted as ⟨result, hview⟩
  rw [hview]
  exact destination_plan_refines_assignment active dependent hookWrote sourceTainted source old htaint hhistory

theorem actual_destination_plan_preserves_soundness {Input : Type} (sem : Semantics Input)
    (meaning : Meaning Input) (source old : Metadata) (sourceValues oldValues : Input → Value)
    (hsource : Sound sem meaning source sourceValues) (hold : Sound sem meaning old oldValues)
    (guard : Input → Bool) (dependent hookWrote sourceTainted : Bool)
    (hguard : GuardSound sem guard dependent) (htaint : source.tainted = sourceTainted)
    (hhistory : hookWrote = false → source.history = none) :
    symbolic.destination_plan (guard sem.capture) dependent hookWrote sourceTainted ⦃ result =>
      Sound sem meaning (applyDestinationPlan (destinationPlanView result) source old)
        (guardedValues guard sourceValues oldValues) ⦄ := by
  step with destination_plan_spec (guard sem.capture) dependent hookWrote sourceTainted as ⟨result, hview⟩
  rw [hview]
  exact destination_plan_preserves_soundness sem meaning source old sourceValues oldValues
    hsource hold guard dependent hookWrote sourceTainted hguard htaint hhistory

theorem actual_destination_plan_native_projection (active dependent hookWrote sourceTainted : Bool)
    (source old : Metadata) (current : Option U32)
    (hhook : active = true → hookWrote = true → current = source.history)
    (hinactive : active = false → current = old.history)
    (hmissing : hookWrote = false → source.history = none)
    (htaint : source.tainted = sourceTainted) :
    symbolic.destination_plan active dependent hookWrote sourceTainted ⦃ result =>
      nativeHistoryAfterPlan (destinationPlanView result) current =
        (assignMetadata active dependent source old).history ∧
      nativeTaintAfterPlan (destinationPlanView result) old.tainted =
        (assignMetadata active dependent source old).tainted ⦄ := by
  step with destination_plan_spec active dependent hookWrote sourceTainted as ⟨result, hview⟩
  rw [hview]
  exact ⟨native_history_refines_assignment active dependent hookWrote sourceTainted source old
    current hhook hinactive hmissing,
    native_taint_refines_assignment active dependent hookWrote sourceTainted source old htaint⟩

end MemoryWrite.Symbolic
