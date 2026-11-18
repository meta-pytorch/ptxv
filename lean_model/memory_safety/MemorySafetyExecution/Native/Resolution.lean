/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.Core
import MemorySafetyExecution.Native.Initialized
import MemorySafety.SourceVectors

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 8000000
set_option maxRecDepth 8192
set_option pp.deepTerms false
set_option pp.proofs false
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

abbrev ResolvedAccess := ptxv_memory_write.execution_core.ResolvedAccess
abbrev Pointer := ptxv_memory_write.value_flow.Pointer

/-- The native byte span and the emitted descriptor use the same width and origin. -/
def ExactDescriptor (pointer : Pointer) (width : U8) (write : Bool)
    (access : ResolvedAccess) : Prop :=
  access.allocation = pointer.allocation ∧
  access.width.val = width.val ∧
  access.request.width.val = width.val ∧
  access.request.alignment.val = width.val ∧
  access.request.kind = (if write then .Store true else .Load)


def OriginRange (program : CoreProgram) (state : CoreState) (pointer : Pointer)
    (width : U8) (access : ResolvedAccess) : Prop :=
  ∃ layout memory,
    program.layouts.val[pointer.allocation.val]? = some layout ∧
    state.allocations.val[pointer.allocation.val]? = some memory ∧
    (width.val = 1 ∨ width.val = 2 ∨ width.val = 4 ∨ width.val = 8) ∧
    access.offset.val = pointer.offset.val ∧
    access.request.address.val = layout.base.val + pointer.offset.val ∧
    memory.live = true ∧
    pointer.offset.val + width.val ≤ layout.bytes.val ∧
    layout.alignment.val ≠ 0 ∧
    layout.alignment.val % width.val = 0 ∧
    pointer.offset.val % width.val = 0 ∧
    layout.base.val % width.val = 0 ∧
    layout.base.val + pointer.offset.val + width.val ≤ 2^64 ∧
    pointer.offset.val + width.val ≤ memory.bytes.val.length ∧
    memory.defined.val.length = memory.bytes.val.length ∧
    (access.request.kind = .Load → ∀ index, index < width.val →
      memory.defined.val[pointer.offset.val + index]? = some true)

private theorem source_width (width : U8) :
    ptxv_memory_write.execution_core.supported_width width =
      .ok (decide (width.val = 1 ∨ width.val = 2 ∨ width.val = 4 ∨ width.val = 8)) := by
  have one : (width = 1#u8) ↔ width.val = 1 := by scalar_tac
  have two : (width = 2#u8) ↔ width.val = 2 := by scalar_tac
  have four : (width = 4#u8) ↔ width.val = 4 := by scalar_tac
  have eight : (width = 8#u8) ↔ width.val = 8 := by scalar_tac
  simp only [ptxv_memory_write.execution_core.supported_width, one, two, four, eight]
  split
  · simp_all
  · split
    · simp_all
    · split <;> simp_all

private theorem successful_false {α : Type} (condition : Prop) [Decidable condition]
    (reason : ptxv_memory_write.execution_core.Fault)
    (next : Result (core.result.Result α ptxv_memory_write.execution_core.Fault)) (out : α)
    (accepted : (if condition then .ok (.Err reason) else next) = .ok (.Ok out)) :
    ¬condition ∧ next = .ok (.Ok out) := by
  by_cases h : condition <;> simp_all

private theorem successful_true {α : Type} (condition : Prop) [Decidable condition]
    (reason : ptxv_memory_write.execution_core.Fault)
    (next : Result (core.result.Result α ptxv_memory_write.execution_core.Fault)) (out : α)
    (accepted : (if condition then next else .ok (.Err reason)) = .ok (.Ok out)) :
    condition ∧ next = .ok (.Ok out) := by
  by_cases h : condition <;> simp_all

theorem source_resolve_range
    (program : CoreProgram) (state : CoreState) (pointer : Pointer)
    (width : U8) (write : Bool) (access : ResolvedAccess)
    (accepted : ptxv_memory_write.execution_core.resolve_access program state pointer width write =
      .ok (.Ok access)) : ExactDescriptor pointer width write access ∧ OriginRange program state pointer width access := by
  unfold ptxv_memory_write.execution_core.resolve_access at accepted
  obtain ⟨supported, supportedEq, next⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
  clear accepted
  obtain ⟨supportedTrue, accepted⟩ := successful_true _ _ _ _ next
  clear next
  obtain ⟨layoutBound, next⟩ := successful_false _ _ _ _ accepted
  clear accepted
  obtain ⟨memoryBound, accepted⟩ := successful_false _ _ _ _ next
  clear next
  obtain ⟨layout, layoutEq, next⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
  clear accepted
  obtain ⟨memory, memoryEq, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ next
  clear next
  obtain ⟨live, next⟩ := successful_true _ _ _ _ accepted
  clear accepted
  obtain ⟨bytes, bytesEq, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ next
  clear next
  obtain ⟨originOffset, next⟩ := successful_false _ _ _ _ accepted
  clear accepted
  obtain ⟨remaining, remainingEq, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ next
  clear next
  obtain ⟨originWidth, next⟩ := successful_false _ _ _ _ accepted
  clear accepted
  obtain ⟨nonzeroAlign, accepted⟩ := successful_false _ _ _ _ next
  clear next
  obtain ⟨alignRemainder, alignEq, next⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
  clear accepted
  obtain ⟨alignZero, accepted⟩ := successful_false _ _ _ _ next
  clear next
  obtain ⟨offsetRemainder, offsetModEq, next⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
  clear accepted
  obtain ⟨offsetZero, accepted⟩ := successful_false _ _ _ _ next
  clear next
  obtain ⟨baseRemainder, baseModEq, next⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
  clear accepted
  obtain ⟨baseZero, accepted⟩ := successful_false _ _ _ _ next
  clear next
  obtain ⟨base, baseEq, next⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
  clear accepted
  obtain ⟨relative, relativeEq, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ next
  clear next
  obtain ⟨address, addressEq, next⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
  clear accepted
  obtain ⟨size, sizeEq, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ next
  clear next
  obtain ⟨finish, finishEq, next⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
  clear accepted
  obtain ⟨maximum, maximumEq, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ next
  clear next
  obtain ⟨limit, limitEq, next⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
  clear accepted
  obtain ⟨physicalBound, accepted⟩ := successful_false _ _ _ _ next
  clear next
  obtain ⟨relativeAgain, relativeAgainEq, next⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
  clear accepted
  obtain ⟨machineMax, machineMaxEq, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ next
  clear next
  obtain ⟨machineBound, next⟩ := successful_false _ _ _ _ accepted
  clear accepted
  obtain ⟨offset, offsetEq, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ next
  clear next
  obtain ⟨count, countEq, next⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
  clear accepted
  obtain ⟨byteOffset, accepted⟩ := successful_false _ _ _ _ next
  clear next
  obtain ⟨storageRemaining, storageRemainingEq, next⟩ := MemorySafety.Source.source_bind_success _ _ _ accepted
  clear accepted
  obtain ⟨storageWidth, accepted⟩ := successful_false _ _ _ _ next
  clear next
  obtain ⟨definedShape, next⟩ := successful_false _ _ _ _ accepted
  clear accepted
  obtain ⟨defined, definedEq, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ next
  clear next
  obtain ⟨definedTrue, next⟩ := successful_true _ _ _ _ accepted
  clear accepted
  obtain ⟨physical, physicalEq, accepted⟩ := MemorySafety.Source.source_bind_success _ _ _ next
  clear next
  have bytesIdentity : bytes = core.convert.num.FromU64U8.from width := by
    simpa only [lift, ok.injEq] using bytesEq.symm
  have countIdentity : count = core.convert.num.FromUsizeU8.from width := by
    simpa only [lift, ok.injEq] using countEq.symm

  have widthKnown : width.val = 1 ∨ width.val = 2 ∨ width.val = 4 ∨ width.val = 8 := by
    rw [source_width] at supportedEq
    simpa only [supportedTrue, ok.injEq, decide_eq_true_eq] using supportedEq
  have bytesVal : bytes.val = width.val := by simp [bytesIdentity]
  have countVal : count.val = width.val := by simp [countIdentity]
  have layoutInside : pointer.allocation.val < program.layouts.val.length := by scalar_tac
  have memoryInside : pointer.allocation.val < state.allocations.val.length := by scalar_tac
  have layoutValue : layout = program.layouts.val[pointer.allocation.val] := by
    change alloc.vec.Vec.index_usize program.layouts pointer.allocation = .ok layout at layoutEq
    have spec := alloc.vec.Vec.index_usize_spec program.layouts pointer.allocation layoutInside
    simpa only [layoutEq, WP.spec_ok] using spec
  have memoryValue : memory = state.allocations.val[pointer.allocation.val] := by
    change alloc.vec.Vec.index_usize state.allocations pointer.allocation = .ok memory at memoryEq
    have spec := alloc.vec.Vec.index_usize_spec state.allocations pointer.allocation memoryInside
    simpa only [memoryEq, WP.spec_ok] using spec
  have baseVal : base.val = layout.base.val := by
    have spec := UScalar.cast_inBounds_spec .U128 layout.base (by scalar_tac)
    simpa only [baseEq, WP.spec_ok] using spec
  have relativeVal : relative.val = pointer.offset.val := by
    have spec := UScalar.cast_inBounds_spec .U128 pointer.offset (by scalar_tac)
    simpa only [relativeEq, WP.spec_ok] using spec
  have addressVal : address.val = layout.base.val + pointer.offset.val := by
    have spec := UScalar.add_spec (x := base) (y := relative) (by scalar_tac)
    have equality : address.val = base.val + relative.val := by
      simpa only [addressEq, WP.spec_ok] using spec
    omega
  have sizeVal : size.val = width.val := by
    have spec := UScalar.cast_inBounds_spec .U128 bytes (by scalar_tac)
    have equality : size.val = bytes.val := by simpa only [sizeEq, WP.spec_ok] using spec
    omega
  have finishVal : finish.val = address.val + size.val := by
    have spec := UScalar.add_spec (x := address) (y := size) (by scalar_tac)
    simpa only [finishEq, WP.spec_ok] using spec
  have maximumVal : maximum.val = U64.max := by
    have spec := UScalar.cast_inBounds_spec .U128 core.num.U64.MAX
      (by simp [core.num.U64.MAX, U64.rMax]; scalar_tac)
    have equality : maximum.val = core.num.U64.MAX.val := by
      simpa only [maximumEq, WP.spec_ok] using spec
    simpa [core.num.U64.MAX, U64.rMax, U64.max, U64.numBits] using equality
  have limitVal : limit.val = 2^64 := by
    have spec := UScalar.add_spec (x := maximum) (y := 1#u128) (by scalar_tac)
    have equality : limit.val = maximum.val + 1 := by
      simpa [limitEq] using spec
    scalar_tac
  have physicalVal : physical.val = layout.base.val + pointer.offset.val := by
    have spec := UScalar.cast_inBounds_spec .U64 address (by scalar_tac)
    have equality : physical.val = address.val := by simpa only [physicalEq, WP.spec_ok] using spec
    omega
  have relativeAgainVal : relativeAgain.val = pointer.offset.val := by
    have spec := UScalar.cast_inBounds_spec .U128 pointer.offset (by scalar_tac)
    simpa only [relativeAgainEq, WP.spec_ok] using spec
  have machineMaxVal : machineMax.val = Usize.max := by
    have bound : core.num.Usize.MAX.val ≤ UScalar.max .U128 := by
      rcases Usize.bounds_eq with narrow | wide <;>
        simp [core.num.Usize.MAX] <;> scalar_tac
    have spec := UScalar.cast_inBounds_spec .U128 core.num.Usize.MAX bound
    have equality : machineMax.val = core.num.Usize.MAX.val := by
      simpa only [machineMaxEq, WP.spec_ok] using spec
    simpa [core.num.Usize.MAX] using equality
  have offsetVal : offset.val = pointer.offset.val := by
    have spec := UScalar.cast_inBounds_spec .Usize pointer.offset (by scalar_tac)
    simpa only [offsetEq, WP.spec_ok] using spec
  have remainingVal : remaining.val = layout.bytes.val - pointer.offset.val := by
    have spec := UScalar.sub_spec (x := layout.bytes) (y := pointer.offset) (by scalar_tac)
    have facts : remaining.val = layout.bytes.val - pointer.offset.val ∧
        pointer.offset.val ≤ layout.bytes.val := by
      simpa only [remainingEq, WP.spec_ok] using spec
    exact facts.1
  have storageRemainingVal : storageRemaining.val = memory.bytes.val.length - offset.val := by
    have spec := UScalar.sub_spec (x := alloc.vec.Vec.len memory.bytes) (y := offset) (by scalar_tac)
    have facts : storageRemaining.val = memory.bytes.val.length - offset.val ∧
        offset.val ≤ memory.bytes.val.length := by
      simpa [storageRemainingEq] using spec
    exact facts.1
  have bytesNonzero : bytes.val ≠ 0 := by omega
  have alignmentMod : layout.alignment.val % width.val = 0 := by
    have spec := UScalar.rem_spec layout.alignment (y := bytes) bytesNonzero
    have equality : alignRemainder.val = layout.alignment.val % bytes.val := by
      simpa only [alignEq, WP.spec_ok] using spec
    scalar_tac
  have offsetMod : pointer.offset.val % width.val = 0 := by
    have spec := UScalar.rem_spec pointer.offset (y := bytes) bytesNonzero
    have equality : offsetRemainder.val = pointer.offset.val % bytes.val := by
      simpa only [offsetModEq, WP.spec_ok] using spec
    scalar_tac
  have baseMod : layout.base.val % width.val = 0 := by
    have spec := UScalar.rem_spec layout.base (y := bytes) bytesNonzero
    have equality : baseRemainder.val = layout.base.val % bytes.val := by
      simpa only [baseModEq, WP.spec_ok] using spec
    scalar_tac
  have loadFlags (read : write = false) : ∀ index, index < width.val →
      memory.defined.val[pointer.offset.val + index]? = some true := by
    have fits : offset.val + count.val ≤ memory.defined.val.length := by scalar_tac
    have run : ptxv_memory_write.execution_core.resolve_access_loop false memory.defined
        offset count 0#usize true = .ok true := by
      simpa only [read, definedTrue] using definedEq
    have initialized := source_initialized_range memory.defined offset count fits run
    simpa only [offsetVal, countVal] using initialized
  have range : OriginRange program state pointer width
      { allocation := pointer.allocation, offset, width := count,
        request := { address := physical, width := bytes, alignment := bytes, kind := (if write then .Store true else .Load) } } := by
    refine ⟨layout, memory, ?_, ?_, widthKnown, offsetVal, physicalVal, live, ?_, ?_,
      alignmentMod, offsetMod, baseMod, ?_, ?_, ?_, ?_⟩
    · simpa only [layoutValue] using List.getElem?_eq_getElem layoutInside
    · simpa only [memoryValue] using List.getElem?_eq_getElem memoryInside
    · scalar_tac
    · scalar_tac
    · scalar_tac
    · scalar_tac
    · scalar_tac
    · intro loadKind
      have read : write = false := by
        cases write with
        | false => rfl
        | true => simp at loadKind
      exact loadFlags read
  cases write <;> simp only [Bool.false_eq_true, ite_false, ite_true, ok.injEq,
    core.result.Result.Ok.injEq] at accepted <;> subst access <;>
    refine ⟨by simp [ExactDescriptor, bytesIdentity, countIdentity], ?_⟩ <;>
    simpa only [Bool.false_eq_true, ite_false, ite_true] using range

theorem source_resolve_descriptor
    (program : CoreProgram) (state : CoreState) (pointer : Pointer)
    (width : U8) (write : Bool) (access : ResolvedAccess)
    (accepted : ptxv_memory_write.execution_core.resolve_access program state pointer width write =
      .ok (.Ok access)) : ExactDescriptor pointer width write access :=
  (source_resolve_range program state pointer width write access accepted).1

end MemorySafetyExecution.Native
