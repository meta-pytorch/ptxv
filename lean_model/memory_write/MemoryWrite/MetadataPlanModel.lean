/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.SymbolicState

set_option autoImplicit false

namespace MemoryWrite.Symbolic

structure DestinationPlan where
  touched : Bool
  clearHistory : Bool
  forgetHalf : Bool
  clearBits : Bool
  activeTaint : Option Bool
  forceTaint : Bool
  deriving DecidableEq

def destinationPlan (active dependent hookWrote sourceTainted : Bool) : DestinationPlan :=
  let touched := active || dependent
  { touched
    clearHistory := touched && (dependent || !hookWrote)
    forgetHalf := dependent
    clearBits := touched
    activeTaint := if active then some sourceTainted else none
    forceTaint := dependent }

def applyDestinationPlan (plan : DestinationPlan) (source old : Metadata) : Metadata :=
  let selected := if plan.activeTaint.isSome then source else old
  { selected with
    history := if plan.clearHistory then none else selected.history
    tainted := if plan.forceTaint then true else plan.activeTaint.getD old.tainted
    defined := if plan.forceTaint then false else selected.defined }

theorem destination_plan_refines_assignment (active dependent hookWrote sourceTainted : Bool)
    (source old : Metadata) (htaint : source.tainted = sourceTainted)
    (hhistory : hookWrote = false → source.history = none) :
    applyDestinationPlan (destinationPlan active dependent hookWrote sourceTainted) source old =
      assignMetadata active dependent source old := by
  cases source
  cases old
  cases active <;> cases dependent <;> cases hookWrote <;>
    simp_all [applyDestinationPlan, destinationPlan, assignMetadata]

theorem inactive_independent_plan_preserves_metadata (hookWrote sourceTainted : Bool)
    (source old : Metadata) :
    applyDestinationPlan (destinationPlan false false hookWrote sourceTainted) source old = old := by
  cases old
  simp [applyDestinationPlan, destinationPlan]

theorem dependent_plan_invalidates_history (active hookWrote sourceTainted : Bool)
    (source old : Metadata) :
    (applyDestinationPlan (destinationPlan active true hookWrote sourceTainted) source old).history = none ∧
    (applyDestinationPlan (destinationPlan active true hookWrote sourceTainted) source old).tainted = true ∧
    (applyDestinationPlan (destinationPlan active true hookWrote sourceTainted) source old).defined = false := by
  cases active <;> simp [applyDestinationPlan, destinationPlan]

theorem destination_plan_preserves_soundness {Input : Type} (sem : Semantics Input)
    (meaning : Meaning Input) (source old : Metadata) (sourceValues oldValues : Input → Value)
    (hsource : Sound sem meaning source sourceValues) (hold : Sound sem meaning old oldValues)
    (guard : Input → Bool) (dependent hookWrote sourceTainted : Bool)
    (hguard : GuardSound sem guard dependent) (htaint : source.tainted = sourceTainted)
    (hhistory : hookWrote = false → source.history = none) :
    Sound sem meaning
      (applyDestinationPlan (destinationPlan (guard sem.capture) dependent hookWrote sourceTainted) source old)
      (guardedValues guard sourceValues oldValues) := by
  rw [destination_plan_refines_assignment _ _ _ _ _ _ htaint hhistory]
  exact assign_sound sem meaning source old sourceValues oldValues hsource hold guard dependent hguard

def nativeHistoryAfterPlan (plan : DestinationPlan) (current : Option Aeneas.Std.U32) :
    Option Aeneas.Std.U32 := if plan.clearHistory then none else current

def nativeTaintAfterPlan (plan : DestinationPlan) (old : Bool) : Bool :=
  if plan.forceTaint then true else plan.activeTaint.getD old

theorem native_history_refines_assignment (active dependent hookWrote sourceTainted : Bool)
    (source old : Metadata) (current : Option Aeneas.Std.U32)
    (hhook : active = true → hookWrote = true → current = source.history)
    (hinactive : active = false → current = old.history)
    (hmissing : hookWrote = false → source.history = none) :
    nativeHistoryAfterPlan (destinationPlan active dependent hookWrote sourceTainted) current =
      (assignMetadata active dependent source old).history := by
  cases active <;> cases dependent <;> cases hookWrote <;>
    simp_all [nativeHistoryAfterPlan, destinationPlan, assignMetadata]

theorem native_taint_refines_assignment (active dependent hookWrote sourceTainted : Bool)
    (source old : Metadata) (htaint : source.tainted = sourceTainted) :
    nativeTaintAfterPlan (destinationPlan active dependent hookWrote sourceTainted) old.tainted =
      (assignMetadata active dependent source old).tainted := by
  cases active <;> cases dependent <;>
    simp_all [nativeTaintAfterPlan, destinationPlan, assignMetadata]

theorem native_inactive_independent_frame (hookWrote sourceTainted : Bool)
    (current : Option Aeneas.Std.U32) (oldTaint : Bool) :
    nativeHistoryAfterPlan (destinationPlan false false hookWrote sourceTainted) current = current ∧
    nativeTaintAfterPlan (destinationPlan false false hookWrote sourceTainted) oldTaint = oldTaint ∧
    (destinationPlan false false hookWrote sourceTainted).clearBits = false := by
  simp [nativeHistoryAfterPlan, nativeTaintAfterPlan, destinationPlan]

end MemoryWrite.Symbolic
