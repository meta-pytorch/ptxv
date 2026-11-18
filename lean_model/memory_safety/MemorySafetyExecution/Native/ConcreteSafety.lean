/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.Placement
import MemorySafetyExecution.Native.Preparation
import MemorySafety.Semantics

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

structure ConcreteSpan (program : CoreProgram) (state : CoreState)
    (request : ptxv_memory_write.memory_safety.Request) where
  allocation : Usize
  offset : Nat
  bytes : Nat
  layout : ptxv_memory_write.value_flow.AllocationLayout
  memory : AllocationMemory
  layoutSelected : program.layouts.val[allocation.val]? = some layout
  memorySelected : state.allocations.val[allocation.val]? = some memory
  address : request.address.val = layout.base.val + offset
  width : request.width.val = bytes
  alignment : request.alignment.val = bytes
  objectAccess : ValueFlow.ObjectAccess
    { bytes := layout.bytes.val, alignment := layout.alignment.val } offset bytes
  physicalAccess : ValueFlow.PhysicalAccess
    { bytes := layout.bytes.val, alignment := layout.alignment.val } layout.base.val offset bytes
  live : memory.live = true
  storageBounds : offset + bytes ≤ memory.bytes.val.length
  definedShape : memory.defined.val.length = memory.bytes.val.length
  initialized : request.kind = .Load → ∀ index, index < bytes →
    memory.defined.val[offset + index]? = some true

private theorem concrete_span_from_resolution
    (program : CoreProgram) (state : CoreState) (pointer : Pointer) (width : U8)
    (write : Bool) (access : ResolvedAccess)
    (descriptor : ExactDescriptor pointer width write access)
    (range : OriginRange program state pointer width access) :
    ∃ span : ConcreteSpan program state access.request,
      span.allocation = access.allocation ∧ span.offset = access.offset.val ∧
      span.bytes = access.width.val := by
  obtain ⟨layout, memory, layoutSelected, memorySelected, widthKnown, offsetVal, addressVal,
    live, originBound, alignment, aligned, offsetAligned, baseAligned, physicalBound,
    storageBound, definedShape, initialized⟩ := range
  have objectAccess : ValueFlow.ObjectAccess
      { bytes := layout.bytes.val, alignment := layout.alignment.val }
      pointer.offset.val width.val :=
    ⟨widthKnown, alignment, Nat.dvd_of_mod_eq_zero aligned,
      Nat.dvd_of_mod_eq_zero offsetAligned, originBound⟩
  have physicalAccess : ValueFlow.PhysicalAccess
      { bytes := layout.bytes.val, alignment := layout.alignment.val }
      layout.base.val pointer.offset.val width.val := by
    change width.val ∣ layout.base.val + pointer.offset.val ∧
      layout.base.val ≤ layout.base.val + pointer.offset.val ∧
      layout.base.val + pointer.offset.val + width.val ≤ layout.base.val + layout.bytes.val ∧
      layout.base.val + pointer.offset.val + width.val ≤ 2^64
    refine ⟨Nat.dvd_add (Nat.dvd_of_mod_eq_zero baseAligned)
      (Nat.dvd_of_mod_eq_zero offsetAligned), by omega, by omega, physicalBound⟩
  let span : ConcreteSpan program state access.request := {
    allocation := access.allocation
    offset := access.offset.val
    bytes := access.width.val
    layout := layout
    memory := memory
    layoutSelected := by simpa only [descriptor.1] using layoutSelected
    memorySelected := by simpa only [descriptor.1] using memorySelected
    address := by simpa only [offsetVal] using addressVal
    width := by rw [descriptor.2.1, descriptor.2.2.1]
    alignment := by rw [descriptor.2.1, descriptor.2.2.2.1]
    objectAccess := by simpa only [offsetVal, descriptor.2.1] using objectAccess
    physicalAccess := by simpa only [offsetVal, descriptor.2.1] using physicalAccess
    live := live
    storageBounds := by simpa only [offsetVal, descriptor.2.1] using storageBound
    definedShape := definedShape
    initialized := by simpa only [offsetVal, descriptor.2.1] using initialized
  }
  exact ⟨span, rfl, rfl, rfl⟩

def ConcreteMemorySafety (program : CoreProgram) (before : CoreState)
    (event : ptxv_memory_write.memory_safety.Event) (after : CoreState) : Prop :=
  match event with
  | .Control => after.allocations = before.allocations
  | .Free _ _ => False
  | .Access _ request => ∃ span : ConcreteSpan program before request,
      (request.kind = .Load → after.allocations = before.allocations) ∧
      (request.kind = .Store true → ∃ memory,
        after.allocations = before.allocations.set span.allocation memory ∧
        ByteFrame span.memory memory span.offset span.bytes)

theorem native_effect_concrete_safety (program : CoreProgram) (before after : CoreState)
    (event : ptxv_memory_write.memory_safety.Event)
    (effect : NativeMemoryEffect program before event after) :
    ConcreteMemorySafety program before event after := by
  cases effect with
  | control after frame => exact frame
  | load actor after pointer width access descriptor range frame =>
    obtain ⟨span, _, _, _⟩ := concrete_span_from_resolution program before pointer width false
      access descriptor range
    refine ⟨span, by intros; exact frame, ?_⟩
    intro store
    have kind := descriptor.2.2.2.2
    simp only [Bool.false_eq_true, ite_false] at kind
    rw [kind] at store
    cases store
  | store actor after pointer width access oldMemory newMemory descriptor range selected frame bytes =>
    obtain ⟨span, allocationEq, offsetEq, countEq⟩ := concrete_span_from_resolution program before
      pointer width true access descriptor range
    refine ⟨span, ?_, ?_⟩
    · intro load
      have kind := descriptor.2.2.2.2
      simp only [ite_true] at kind
      rw [kind] at load
      cases load
    · intro store
      have memoryEq : span.memory = oldMemory := by
        apply Option.some.inj
        rw [← span.memorySelected, allocationEq, selected]
      exact ⟨newMemory, by simpa only [allocationEq] using frame,
        by simpa only [memoryEq, offsetEq, countEq] using bytes⟩

theorem source_execute_concrete_safety (program : CoreProgram) (before after : CoreState)
    (actor : Usize) (event : ptxv_memory_write.memory_safety.Event)
    (accepted : ptxv_memory_write.execution_core.execute program before actor =
      .ok (.Ok event, after)) : ConcreteMemorySafety program before event after :=
  native_effect_concrete_safety program before after event
    (source_execute_memory_effect program before after actor event accepted)

theorem core_step_concrete_safety (program : CoreProgram) (before after : CoreState)
    (event : ptxv_memory_write.memory_safety.Event) (step : CoreStep program before event after) :
    ConcreteMemorySafety program before event after :=
  native_effect_concrete_safety program before after event
    (core_step_memory_effect program before after event step)

def touches {program : CoreProgram} {state : CoreState}
    {request : ptxv_memory_write.memory_safety.Request}
    (span : ConcreteSpan program state request) (address : Nat) : Prop :=
  ∃ index, index < span.bytes ∧ address = span.layout.base.val + span.offset + index

theorem touches_exactly_request {program : CoreProgram} {state : CoreState}
    {request : ptxv_memory_write.memory_safety.Request}
    (span : ConcreteSpan program state request) (address : Nat) :
    touches span address ↔ request.address.val ≤ address ∧
      address < request.address.val + request.width.val := by
  rw [span.address, span.width]
  constructor
  · rintro ⟨index, bound, rfl⟩
    omega
  · rintro ⟨lower, upper⟩
    refine ⟨address - (span.layout.base.val + span.offset), by omega, by omega⟩

theorem disjoint_requests_have_no_common_native_byte
    {program : CoreProgram} {before other : CoreState}
    {first second : ptxv_memory_write.memory_safety.Request}
    (left : ConcreteSpan program before first) (right : ConcreteSpan program other second)
    (disjoint : ¬MemorySafety.Overlap first.address.val first.width.val
      second.address.val second.width.val) :
    ∀ address, ¬(touches left address ∧ touches right address) := by
  intro address both
  obtain ⟨leftLow, leftHigh⟩ := (touches_exactly_request left address).1 both.1
  obtain ⟨rightLow, rightHigh⟩ := (touches_exactly_request right address).1 both.2
  apply disjoint
  unfold MemorySafety.Overlap
  omega

end MemorySafetyExecution.Native
