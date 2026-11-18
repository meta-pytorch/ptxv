/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.SourceVectors
import MemorySafety.SourcePrimitives

open Aeneas Aeneas.Std Aeneas.Std.WP Result PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option backward.isDefEq.respectTransparency false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyIR

theorem source_resolved_event (resolved : memory_event.ResolvedGlobalEvent) :
    memory_event.ResolvedGlobalEvent.impl.event resolved = .ok resolved.event := rfl

theorem source_resolved_location (resolved : memory_event.ResolvedGlobalEvent) :
    memory_event.ResolvedGlobalEvent.impl.location resolved = .ok resolved.location := rfl

theorem source_resolve_allocation_preserves_event
    (event : memory_safety.Event) (index : Usize) (base : U64) (bytes : Usize)
    (resolved : memory_event.ResolvedGlobalEvent)
    (accepted : memory_event.resolve_allocation event index base bytes = .ok (some resolved)) :
    resolved.event = event ∧ ∃ offset, resolved.location = exec.Loc.Global index offset := by
  cases event with
  | Free thread allocation => simp [memory_event.resolve_allocation] at accepted
  | Control => simp [memory_event.resolve_allocation] at accepted
  | Access thread request =>
    unfold memory_event.resolve_allocation at accepted
    obtain ⟨_, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
    obtain ⟨_, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
    obtain ⟨_, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
    obtain ⟨_, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
    obtain ⟨_, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
    obtain ⟨_, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
    obtain ⟨_, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
    obtain ⟨_, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
    split at accepted
    · split at accepted
      · split at accepted
        · obtain ⟨offset, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
          simp only [ok.injEq, Option.some.injEq] at accepted
          cases accepted
          exact ⟨rfl, offset, rfl⟩
        · simp at accepted
      · simp at accepted
    · simp at accepted

def ExactGlobalResolution (event : memory_safety.Event) (index : Usize)
    (base : U64) (bytes : Usize) (resolved : memory_event.ResolvedGlobalEvent) : Prop :=
  resolved.event = event ∧ ∃ actor request offset,
    event = memory_safety.Event.Access actor request ∧
    resolved.location = exec.Loc.Global index offset ∧
    request.address.val = base.val + offset.val ∧
    offset.val + request.width.val ≤ bytes.val ∧
    base.val + bytes.val ≤ 2 ^ 64

theorem source_resolve_allocation_bounds
    (event : memory_safety.Event) (index : Usize) (base : U64) (bytes : Usize) :
    memory_event.resolve_allocation event index base bytes
      ⦃ output => ∀ resolved, output = some resolved →
        ExactGlobalResolution event index base bytes resolved ⦄ := by
  cases event with
  | Free actor allocation => simp [memory_event.resolve_allocation]
  | Control => simp [memory_event.resolve_allocation]
  | Access actor request =>
    have bytesBound : bytes.val ≤ U64.max := by
      rcases Usize.bounds_eq with narrow | wide <;> scalar_tac
    unfold memory_event.resolve_allocation
    step with UScalar.cast_inBounds_spec .U128 request.address (by scalar_tac) as ⟨address, addressEq⟩
    step with UScalar.cast_inBounds_spec .U128 request.width (by scalar_tac) as ⟨width, widthEq⟩
    step as ⟨finish, finishEq⟩
    step with UScalar.cast_inBounds_spec .U128 base (by scalar_tac) as ⟨start, startEq⟩
    step with UScalar.cast_inBounds_spec .U128 bytes (by scalar_tac) as ⟨size, sizeEq⟩
    step as ⟨endAllocation, endEq⟩
    step with UScalar.cast_inBounds_spec .U128 core.num.U64.MAX
      (by simp [core.num.U64.MAX, U64.rMax]; scalar_tac) as ⟨maximum, maximumEq⟩
    step as ⟨limit, limitEq⟩
    · simp only [U64.rMax] at maximumEq
      scalar_tac
    · split
      · split
        · split
          · step as ⟨offset, offsetEq⟩
            simp only [Option.some.injEq, forall_eq']
            refine ⟨rfl, actor, request, offset, rfl, rfl, ?_, ?_, ?_⟩ <;>
              simp only [U64.rMax] at maximumEq <;> scalar_tac
          · simp
        · simp
      · simp

def InGlobalAllocations (allocations : Slice machine.Alloc) (event : memory_safety.Event)
    (resolved : memory_event.ResolvedGlobalEvent) : Prop :=
  ∃ index : Usize, ∃ inside : index.val < allocations.val.length,
    ExactGlobalResolution event index allocations.val[index.val].base
      (alloc.vec.Vec.len allocations.val[index.val].init) resolved

theorem source_resolve_global_event_loop
    (actor : Usize) (request : memory_safety.Request) (allocations : Slice machine.Alloc)
    (index : Usize) (found : Option memory_event.ResolvedGlobalEvent)
    (initial : ∀ resolved, found = some resolved →
      InGlobalAllocations allocations (.Access actor request) resolved) :
    memory_event.resolve_global_event_loop actor request allocations index found
      ⦃ output => ∀ resolved, output = some resolved →
        InGlobalAllocations allocations (.Access actor request) resolved ⦄ := by
  unfold memory_event.resolve_global_event_loop
  apply loop.spec_decr_nat
    (fun current : Usize × Option memory_event.ResolvedGlobalEvent =>
      allocations.val.length - current.1.val)
    (fun current : Usize × Option memory_event.ResolvedGlobalEvent =>
      ∀ resolved, current.2 = some resolved →
        InGlobalAllocations allocations (.Access actor request) resolved)
  · intro current invariant
    rcases current with ⟨cursor, previous⟩
    unfold memory_event.resolve_global_event_loop.body
    by_cases inside : cursor.val < allocations.val.length
    · have test : cursor < Slice.len allocations := by scalar_tac
      simp only [test, ite_true, core.option.Option.is_none]
      cases previous with
      | some previous => simpa using invariant
      | none =>
        simp only [Option.isNone, ite_true]
        step as ⟨allocation, allocationEq⟩
        step with source_resolve_allocation_bounds (.Access actor request) cursor allocation.base
          (alloc.vec.Vec.len allocation.init) as ⟨output, outputSafe⟩
        step as ⟨next, nextEq⟩
        refine ⟨?_, ?_⟩
        · intro resolved accepted
          have resolution := outputSafe resolved accepted
          have rowEq : allocation = allocations.val[cursor.val] := allocationEq
          refine ⟨cursor, inside, ?_⟩
          simpa [ExactGlobalResolution, rowEq, alloc.vec.Vec.len, Slice.len] using resolution
        · scalar_tac
    · have test : ¬cursor < Slice.len allocations := by scalar_tac
      simpa only [test, ite_false, WP.spec_ok] using invariant
  · exact initial

theorem source_resolve_global_event_bounds
    (event : memory_safety.Event) (allocations : Slice machine.Alloc)
    (resolved : memory_event.ResolvedGlobalEvent)
    (accepted : memory_event.resolve_global_event event allocations = .ok (some resolved)) :
    InGlobalAllocations allocations event resolved := by
  cases event with
  | Free actor allocation => simp [memory_event.resolve_global_event] at accepted
  | Control => simp [memory_event.resolve_global_event] at accepted
  | Access actor request =>
    unfold memory_event.resolve_global_event at accepted
    obtain ⟨allowed, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
    cases allowed with
    | false => simp at accepted
    | true =>
      simp only [ite_true] at accepted
      split at accepted
      · simp at accepted
      · obtain ⟨remainder, _, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
        split at accepted
        · simp at accepted
        · have safe := source_resolve_global_event_loop actor request allocations 0#usize none
            (by simp)
          simpa only [accepted, WP.spec_ok, Option.some.injEq, forall_eq'] using safe

end MemorySafetyIR
