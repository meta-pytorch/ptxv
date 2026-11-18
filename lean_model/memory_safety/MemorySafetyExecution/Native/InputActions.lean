/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.SourceDependencies

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

open ValueFlow ValueFlow.Source

private theorem scalar_value_same {ty : UScalarTy} (left right : UScalar ty) : left.val=right.val ↔ left=right :=
  ⟨UScalar.eq_of_val_eq,congrArg UScalar.val⟩

private theorem pointer_model_same (left right : V.Pointer) : pointerModel left=pointerModel right ↔ left=right := by
  constructor
  · intro same
    have allocation := UScalar.eq_of_val_eq (congrArg ValueFlow.Pointer.allocation same)
    have offset := UScalar.eq_of_val_eq (congrArg ValueFlow.Pointer.offset same)
    cases left; cases right; simp_all
  · rintro rfl
    rfl

def NativeActionsRelated (tainted : Bool) (left right : V.Action) : Prop :=
  match left,right with
  | .WriteVirtual destination value,.WriteVirtual otherDestination otherValue =>
      destination=otherDestination ∧ ValuesRelated tainted (valueModel value) (valueModel otherValue)
  | .WritePredicate destination value,.WritePredicate otherDestination otherValue =>
      destination=otherDestination ∧ (tainted=false → value=otherValue)
  | .Store bytes pointer value,.Store otherBytes otherPointer otherValue =>
      bytes=otherBytes ∧ pointer=otherPointer ∧ (tainted=false → value=otherValue)
  | _,_ => left=right

theorem source_actions_inputs (left right : V.Action) (tainted : Bool)
    (related : ActionsRelated tainted (actionModel left) (actionModel right)) :
    NativeActionsRelated tainted left right := by
  cases left <;> cases right <;>
    simp only [NativeActionsRelated,ActionsRelated,actionModel,reduceCtorEq,ValueFlow.Action.jump.injEq,ValueFlow.Action.load.injEq,
      value_flow.Action.Jump.injEq,value_flow.Action.Load.injEq,scalar_value_same,pointer_model_same] at related ⊢
  all_goals exact related

end MemorySafetyExecution.Native
