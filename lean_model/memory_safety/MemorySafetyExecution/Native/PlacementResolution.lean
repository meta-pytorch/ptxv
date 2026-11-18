/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.Placement
import MemoryWrite.TotalPrimitives

open Aeneas Aeneas.Std Result PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option maxHeartbeats 8000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option debug.skipKernelTC false
set_option backward.isDefEq.respectTransparency false
set_option backward.split false

namespace MemorySafetyExecution.Native

abbrev PlacementLayout := ptxv_memory_write.value_flow.AllocationLayout
abbrev PlacementMemory := ptxv_memory_write.execution_core.AllocationBytes

def ProgramPlacements (left right : CoreProgram) : Prop :=
  left.layouts.val.length = right.layouts.val.length ∧
  (∀ (index : Nat) (old fresh : PlacementLayout), left.layouts.val[index]? = some old → right.layouts.val[index]? = some fresh →
    fresh.bytes = old.bytes ∧ fresh.alignment = old.alignment ∧
      ValueFlow.LegalPlacement { bytes := fresh.bytes.val, alignment := fresh.alignment.val } fresh.base.val) ∧
  right.layouts.val.Pairwise (fun first second =>
    first.base.val + first.bytes.val ≤ second.base.val ∨ second.base.val + second.bytes.val ≤ first.base.val)

@[step]
private theorem initialized_complete (write : Bool) (values : alloc.vec.Vec Bool)
    (offset count cursor : Usize) (defined : Bool)
    (bound : cursor.val ≤ count.val) (fits : offset.val + count.val ≤ values.val.length)
    (trueFlag : defined = true)
    (initialized : write = false → ∀ index, index < count.val → values.val[offset.val + index]? = some true) :
    ptxv_memory_write.execution_core.resolve_access_loop write values offset count cursor defined
      ⦃ result => result = true ⦄ := by
  unfold ptxv_memory_write.execution_core.resolve_access_loop
  apply loop.spec_decr_nat
    (fun current : Usize × Bool => count.val - current.1.val)
    (fun current : Usize × Bool => current.1.val ≤ count.val ∧ current.2 = true)
  · rintro ⟨atIndex, flag⟩ ⟨insideCount, flagTrue⟩
    unfold ptxv_memory_write.execution_core.resolve_access_loop.body
    have flagValue : flag = true := flagTrue
    subst flag
    by_cases inside : atIndex.val < count.val
    · have test : atIndex < count := by scalar_tac
      simp only [test, ite_true]
      cases write with
      | true => simp
      | false =>
        simp only [Bool.false_eq_true, ite_false, ite_true]
        step as ⟨selected, selectedVal⟩
        have selectedBound : selected.val < values.val.length := by scalar_tac
        step as ⟨entry, entryAt⟩
        have entryTrue : entry = true := by
          have indexed := initialized rfl atIndex.val inside
          rw [←selectedVal, List.getElem?_eq_getElem selectedBound] at indexed
          have equal : values.val[selected.val] = true := Option.some.inj indexed
          exact entryAt.trans equal
        subst entry
        step as ⟨next, nextVal⟩
        exact ⟨by scalar_tac, entryTrue, by scalar_tac⟩
    · have test : ¬atIndex < count := by scalar_tac
      simp [test]
  · exact ⟨bound, trueFlag⟩

private theorem width_supported (width : U8) (supported : ValueFlow.supportedWidth width.val) :
    ptxv_memory_write.execution_core.supported_width width = .ok true := by
  rcases supported with one | two | four | eight
  · have same : width = 1#u8 := by scalar_tac
    subst width
    rfl
  · have same : width = 2#u8 := by scalar_tac
    subst width
    rfl
  · have same : width = 4#u8 := by scalar_tac
    subst width
    rfl
  · have same : width = 8#u8 := by scalar_tac
    subst width
    rfl

@[step]
private theorem width_u64 (width : U8) :
    lift (core.convert.num.FromU64U8.from width) ⦃ result => result.val = width.val ⦄ := by
  simp [lift]

@[step]
private theorem width_usize (width : U8) :
    lift (core.convert.num.FromUsizeU8.from width) ⦃ result => result.val = width.val ⦄ := by
  simp [lift]

@[step]
theorem source_resolve_from_origin (program : CoreProgram) (state : CoreState) (pointer : Pointer)
    (width : U8) (write : Bool) (layout : PlacementLayout) (memory : PlacementMemory)
    (layoutAt : program.layouts.val[pointer.allocation.val]? = some layout)
    (memoryAt : state.allocations.val[pointer.allocation.val]? = some memory)
    (origin : ValueFlow.ObjectAccess { bytes := layout.bytes.val, alignment := layout.alignment.val } pointer.offset.val width.val)
    (placement : ValueFlow.LegalPlacement { bytes := layout.bytes.val, alignment := layout.alignment.val } layout.base.val)
    (live : memory.live = true)
    (storage : pointer.offset.val + width.val ≤ memory.bytes.val.length)
    (shape : memory.defined.val.length = memory.bytes.val.length)
    (initialized : write = false → ∀ index, index < width.val → memory.defined.val[pointer.offset.val + index]? = some true) :
    ptxv_memory_write.execution_core.resolve_access program state pointer width write ⦃ result =>
      ∃ access, result = .Ok access ∧ ExactDescriptor pointer width write access ∧
        access.offset.val = pointer.offset.val ∧ access.request.address.val = layout.base.val + pointer.offset.val ⦄ := by
  have supported := origin.1
  have widthPositive : 0 < width.val := by rcases supported with h | h | h | h <;> omega
  have extentBound := origin.2.2.2.2
  have alignmentNonzero := origin.2.1
  have widthAlignment := origin.2.2.1
  have widthOffset := origin.2.2.2.1
  have alignmentBase := placement.1
  have physical := ValueFlow.object_access_all_placements _ _ _ origin layout.base.val placement
  have widthBase : width.val ∣ layout.base.val := dvd_trans widthAlignment alignmentBase
  have layoutInside := (List.getElem?_eq_some_iff.mp layoutAt).1
  have memoryInside := (List.getElem?_eq_some_iff.mp memoryAt).1
  have layoutRead : alloc.vec.Vec.index_usize program.layouts pointer.allocation = .ok layout := by
    simp [alloc.vec.Vec.index_usize, layoutAt]
  have memoryRead : alloc.vec.Vec.index_usize state.allocations pointer.allocation = .ok memory := by
    simp [alloc.vec.Vec.index_usize, memoryAt]
  unfold ptxv_memory_write.execution_core.resolve_access
  rw [width_supported width supported]
  simp only [bind_ok, ite_true]
  have allocationInLayout : ¬pointer.allocation >= alloc.vec.Vec.len program.layouts := by scalar_tac
  have allocationInMemory : ¬pointer.allocation >= alloc.vec.Vec.len state.allocations := by scalar_tac
  simp only [allocationInLayout, allocationInMemory, ite_false, alloc.vec.Vec.index_slice_index,
    layoutRead, memoryRead, bind_ok, live, ite_true]
  step as ⟨bytes, bytesVal⟩
  have bytesEq : bytes.val = width.val := by simpa using bytesVal
  have offsetWithin : ¬pointer.offset > layout.bytes := by scalar_tac
  simp only [offsetWithin, ite_false]
  step as ⟨remaining, remainingVal⟩
  have fitsOrigin : ¬bytes > remaining := by scalar_tac
  have alignmentNotZero : ¬layout.alignment = 0#u64 := by scalar_tac
  simp only [fitsOrigin, alignmentNotZero, ite_false]
  step as ⟨alignmentMod, alignmentModVal⟩
  have aligned : alignmentMod = 0#u64 := by
    have modulo := Nat.mod_eq_zero_of_dvd widthAlignment
    scalar_tac
  simp only [aligned, bne_self_eq_false, Bool.false_eq_true, ite_false]
  step as ⟨offsetMod, offsetModVal⟩
  have alignedOffset : offsetMod = 0#u64 := by
    have modulo := Nat.mod_eq_zero_of_dvd widthOffset
    scalar_tac
  simp only [alignedOffset, bne_self_eq_false, Bool.false_eq_true, ite_false]
  step as ⟨baseMod, baseModVal⟩
  have alignedBase : baseMod = 0#u64 := by
    have modulo := Nat.mod_eq_zero_of_dvd widthBase
    scalar_tac
  simp only [alignedBase, bne_self_eq_false, Bool.false_eq_true, ite_false]
  step with UScalar.cast_inBounds_spec .U128 layout.base (by scalar_tac) as ⟨base, baseVal⟩
  step with UScalar.cast_inBounds_spec .U128 pointer.offset (by scalar_tac) as ⟨relative, relativeVal⟩
  step as ⟨address, addressVal⟩
  step with UScalar.cast_inBounds_spec .U128 bytes (by scalar_tac) as ⟨size, sizeVal⟩
  step as ⟨finish, finishVal⟩
  step with UScalar.cast_inBounds_spec .U128 core.num.U64.MAX (by simp [core.num.U64.MAX, U64.rMax]; scalar_tac) as ⟨maximum, maximumVal⟩
  change maximum.val = 18446744073709551615 at maximumVal
  step with UScalar.add_spec (x := maximum) (y := 1#u128) (by scalar_tac) as ⟨limit, limitVal⟩
  have max64Val : core.num.U64.MAX.val = 2^64 - 1 := by decide
  have physicalEnd := physical.2.2.2
  have physicalInside : ¬finish > limit := by scalar_tac
  simp only [physicalInside, ite_false]
  step with UScalar.cast_inBounds_spec .U128 pointer.offset (by scalar_tac) as ⟨relativeAgain, relativeAgainVal⟩
  have machineMaximumBound : core.num.Usize.MAX.val ≤ UScalar.max .U128 := by
    rcases Usize.bounds_eq with narrow | wide <;> simp [core.num.Usize.MAX] <;> scalar_tac
  step with UScalar.cast_inBounds_spec .U128 core.num.Usize.MAX machineMaximumBound as ⟨machineMax, machineMaxVal⟩
  have memoryBound := alloc.vec.Vec.len_ineq memory.bytes
  have machineInside : ¬relativeAgain > machineMax := by scalar_tac
  simp only [machineInside, ite_false]
  step with UScalar.cast_inBounds_spec .Usize pointer.offset (by scalar_tac) as ⟨offset, offsetVal⟩
  step as ⟨count, countVal⟩
  have countEq : count.val = width.val := by simpa using countVal
  have storageOffset : ¬offset > alloc.vec.Vec.len memory.bytes := by scalar_tac
  simp only [storageOffset, ite_false]
  step as ⟨storageRemaining, storageRemainingVal⟩
  have storageWidth : ¬count > storageRemaining := by scalar_tac
  have storageShape : ¬alloc.vec.Vec.len memory.defined != alloc.vec.Vec.len memory.bytes := by scalar_tac
  simp only [storageWidth, storageShape, Bool.false_eq_true, ite_false]
  step with initialized_complete write memory.defined offset count 0#usize true
    (by scalar_tac) (by scalar_tac) rfl (by simpa only [offsetVal, countEq] using initialized) as ⟨defined, definedTrue⟩
  subst defined
  simp only [ite_true]
  step with UScalar.cast_inBounds_spec .U64 address (by scalar_tac) as ⟨physicalAddress, physicalAddressVal⟩
  cases write <;> simp only [Bool.false_eq_true, ite_false, ite_true, WP.spec_ok]
  all_goals refine ⟨_, rfl, ?_, offsetVal, ?_⟩
  all_goals try simp only [ExactDescriptor, Bool.false_eq_true, ite_false, ite_true]
  all_goals repeat' apply And.intro
  all_goals first | rfl | scalar_tac


def StorageShapesRelated (left right : CoreState) : Prop :=
  left.allocations.val.length = right.allocations.val.length ∧
  ∀ (index : Nat) (old fresh : PlacementMemory), left.allocations.val[index]? = some old → right.allocations.val[index]? = some fresh →
    old.bytes.val.length = fresh.bytes.val.length ∧ old.defined = fresh.defined ∧ old.live = fresh.live

theorem source_resolve_placements (leftProgram rightProgram : CoreProgram) (left right : CoreState)
    (pointer : Pointer) (width : U8) (write : Bool) (oldAccess : ResolvedAccess)
    (placements : ProgramPlacements leftProgram rightProgram)
    (storageShapes : StorageShapesRelated left right)
    (computed : ptxv_memory_write.execution_core.resolve_access leftProgram left pointer width write = .ok (.Ok oldAccess)) :
    ∃ access layout, ptxv_memory_write.execution_core.resolve_access rightProgram right pointer width write = .ok (.Ok access) ∧
      rightProgram.layouts.val[pointer.allocation.val]? = some layout ∧
      access.allocation = oldAccess.allocation ∧ access.offset = oldAccess.offset ∧ access.width = oldAccess.width ∧
      access.request.width = oldAccess.request.width ∧ access.request.alignment = oldAccess.request.alignment ∧
      access.request.kind = oldAccess.request.kind ∧ access.request.address.val = layout.base.val + pointer.offset.val := by
  obtain ⟨oldDescriptor, range⟩ := source_resolve_range leftProgram left pointer width write oldAccess computed
  obtain ⟨oldLayout, oldMemory, oldLayoutAt, oldMemoryAt, supported, oldOffset, oldAddress,
    oldLive, originBound, nonzero, widthAlignment, offsetAligned, baseAligned, physicalBound,
    storageBound, definedShape, initialized⟩ := range
  have layoutLengths := placements.1
  have allocationLengths := storageShapes.1
  have oldLayoutInside := (List.getElem?_eq_some_iff.mp oldLayoutAt).1
  have layoutInside : pointer.allocation.val < rightProgram.layouts.val.length := by omega
  let layout := rightProgram.layouts.val[pointer.allocation.val]
  have layoutAt : rightProgram.layouts.val[pointer.allocation.val]? = some layout := List.getElem?_eq_getElem layoutInside
  obtain ⟨sameBytes, sameAlignment, legal⟩ := placements.2.1 pointer.allocation.val oldLayout layout oldLayoutAt layoutAt
  have oldMemoryInside := (List.getElem?_eq_some_iff.mp oldMemoryAt).1
  have memoryInside : pointer.allocation.val < right.allocations.val.length := by omega
  let memory := right.allocations.val[pointer.allocation.val]
  have memoryAt : right.allocations.val[pointer.allocation.val]? = some memory := List.getElem?_eq_getElem memoryInside
  obtain ⟨bytesShape, sameDefined, sameLive⟩ := storageShapes.2 pointer.allocation.val oldMemory memory oldMemoryAt memoryAt
  have accessMeaning : ValueFlow.ObjectAccess { bytes := layout.bytes.val, alignment := layout.alignment.val }
      pointer.offset.val width.val := by
    refine ⟨supported, ?_, ?_, Nat.dvd_of_mod_eq_zero offsetAligned, ?_⟩
    · simpa only [sameAlignment] using nonzero
    · simpa only [sameAlignment] using Nat.dvd_of_mod_eq_zero widthAlignment
    · simpa only [sameBytes] using originBound
  have loadFlags : write = false → ∀ index, index < width.val → memory.defined.val[pointer.offset.val + index]? = some true := by
    intro read index bound
    rw [←sameDefined]
    apply initialized
    · simp only [ExactDescriptor] at oldDescriptor
      rw [oldDescriptor.2.2.2.2, read]
      rfl
    · exact bound
  have specification := source_resolve_from_origin rightProgram right pointer width write layout memory
    layoutAt memoryAt accessMeaning legal (by rw [←sameLive]; exact oldLive)
    (by omega) (by rw [←sameDefined]; omega) loadFlags
  obtain ⟨result, returned, access, resultEq, descriptor, offsetEq, addressEq⟩ :=
    MemoryWrite.spec_returns _ _ specification
  subst result
  refine ⟨access, layout, returned, layoutAt, descriptor.1.trans oldDescriptor.1.symm, ?_, ?_, ?_, ?_, ?_, addressEq⟩
  · apply UScalar.eq_of_val_eq
    have widthValue := descriptor.2.1
    have oldWidthValue := oldDescriptor.2.1
    have requestWidth := descriptor.2.2.1
    have oldRequestWidth := oldDescriptor.2.2.1
    have requestAlignment := descriptor.2.2.2.1
    have oldRequestAlignment := oldDescriptor.2.2.2.1
    omega
  · apply UScalar.eq_of_val_eq
    have widthValue := descriptor.2.1
    have oldWidthValue := oldDescriptor.2.1
    have requestWidth := descriptor.2.2.1
    have oldRequestWidth := oldDescriptor.2.2.1
    have requestAlignment := descriptor.2.2.2.1
    have oldRequestAlignment := oldDescriptor.2.2.2.1
    omega
  · apply UScalar.eq_of_val_eq
    have widthValue := descriptor.2.1
    have oldWidthValue := oldDescriptor.2.1
    have requestWidth := descriptor.2.2.1
    have oldRequestWidth := oldDescriptor.2.2.1
    have requestAlignment := descriptor.2.2.2.1
    have oldRequestAlignment := oldDescriptor.2.2.2.1
    omega
  · apply UScalar.eq_of_val_eq
    have widthValue := descriptor.2.1
    have oldWidthValue := oldDescriptor.2.1
    have requestWidth := descriptor.2.2.1
    have oldRequestWidth := oldDescriptor.2.2.1
    have requestAlignment := descriptor.2.2.2.1
    have oldRequestAlignment := oldDescriptor.2.2.2.1
    omega
  · exact descriptor.2.2.2.2.trans oldDescriptor.2.2.2.2.symm

end MemorySafetyExecution.Native
