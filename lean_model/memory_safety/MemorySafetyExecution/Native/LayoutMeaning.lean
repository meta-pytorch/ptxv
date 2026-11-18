/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.LayoutGeometry
import MemorySafetyExecution.Native.CompatibleMeaning

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
open MemorySafety.Source
set_option autoImplicit false
set_option maxHeartbeats 8000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option debug.skipKernelTC false
namespace MemorySafetyExecution.Native

private theorem layout_loop_unroll (program : CoreProgram) (allocations : alloc.vec.Vec execution_core.AllocationBytes)
    (flag : Bool) (cursor : Usize) :
    execution_core.layout_valid_loop program.code program.layouts program.initial program.outputs allocations flag cursor =
      (do let control ← execution_core.layout_valid_loop.body program.code program.layouts program.initial program.outputs allocations flag cursor
          match control with
          | .cont pair => execution_core.layout_valid_loop program.code program.layouts program.initial program.outputs allocations pair.1 pair.2
          | .done result => .ok result) := by
  unfold execution_core.layout_valid_loop
  rw [loop.eq_def]
  congr 1
  funext control
  cases control with
  | cont pair => cases pair; rfl
  | done result => rfl

private theorem layout_loop_false (program : CoreProgram) (allocations : alloc.vec.Vec execution_core.AllocationBytes) (cursor : Usize) :
    execution_core.layout_valid_loop program.code program.layouts program.initial program.outputs allocations false cursor=.ok false := by
  rw [layout_loop_unroll]
  simp [execution_core.layout_valid_loop.body]

theorem source_layout_loop_meaning (program : CoreProgram) (allocations : alloc.vec.Vec execution_core.AllocationBytes)
    (flag : Bool) (cursor : Usize)
    (computed : execution_core.layout_valid_loop program.code program.layouts program.initial program.outputs allocations flag cursor=.ok true) :
    flag=true ∧ ∀ (index : Nat) (layout : value_flow.AllocationLayout) (memory : execution_core.AllocationBytes), cursor.val ≤ index →
      program.layouts.val[index]?=some layout → allocations.val[index]?=some memory →
      StorageExtent layout (alloc.vec.Vec.len memory.bytes) (alloc.vec.Vec.len memory.defined) ∧
      ∀ earlier other, earlier < index → program.layouts.val[earlier]?=some other → LayoutDisjoint layout other := by
  generalize fuelEq : program.layouts.val.length-cursor.val=fuel
  induction fuel using Nat.strong_induction_on generalizing flag cursor with
  | h fuel ih =>
    cases flag with
    | false => simp only [layout_loop_false,ok.injEq,Bool.false_eq_true] at computed
    | true =>
      refine ⟨rfl,?_⟩
      rw [layout_loop_unroll] at computed
      by_cases inside : cursor.val < program.layouts.val.length
      · have test : cursor < alloc.vec.Vec.len program.layouts := by scalar_tac
        simp only [execution_core.layout_valid_loop.body,test,ite_true,Std.bind_assoc] at computed
        obtain ⟨layout,layoutRead,computed⟩ := source_bind_success _ _ _ computed
        obtain ⟨memory,memoryRead,computed⟩ := source_bind_success _ _ _ computed
        obtain ⟨shape,shapeRun,computed⟩ := source_bind_success _ _ _ computed
        obtain ⟨compatible,compatibleRun,computed⟩ := source_bind_success _ _ _ computed
        obtain ⟨next,nextRun,computed⟩ := source_bind_success _ _ _ computed
        have nextVal : next.val=cursor.val+1 := by
          have spec:=UScalar.add_spec (x:=cursor) (y:=1#usize) (by
            have bound:=alloc.vec.Vec.len_ineq program.layouts
            scalar_tac)
          simpa [nextRun,WP.spec_ok] using spec
        have recursiveMeaning := ih (program.layouts.val.length-next.val) (by omega) compatible next
          (by simpa only [bind_ok] using computed) rfl
        have compatibleTrue : compatible=true := recursiveMeaning.1
        subst compatible
        have shapeTrue : shape=true := by
          cases shape <;> simp_all
        subst shape
        simp only [ite_true] at compatibleRun
        have layoutAt := vec_index_success program.layouts cursor layout (by
          simpa only [alloc.vec.Vec.index_slice_index] using layoutRead)
        have memoryAt := vec_index_success allocations cursor memory (by
          simpa only [alloc.vec.Vec.index_slice_index] using memoryRead)
        have extentSpec:=source_allocation_shape_extent layout (alloc.vec.Vec.len memory.bytes) (alloc.vec.Vec.len memory.defined)
        rw [shapeRun] at extentSpec
        have extent : StorageExtent layout (alloc.vec.Vec.len memory.bytes) (alloc.vec.Vec.len memory.defined) := by
          simpa only [WP.spec_ok,true_implies] using extentSpec
        obtain ⟨selected,selectedAt,disjoint⟩ := source_allocation_compatible_meaning program cursor compatibleRun
        have selectedEq : selected=layout := by simpa only [layoutAt,Option.some.injEq] using selectedAt.symm
        subst selected
        intro index old storage lower oldAt storageAt
        by_cases same : index=cursor.val
        · rw [same,layoutAt] at oldAt
          rw [same,memoryAt] at storageAt
          have oldEq : old=layout := by simpa only [Option.some.injEq] using oldAt.symm
          have storageEq : storage=memory := by simpa only [Option.some.injEq] using storageAt.symm
          subst old
          subst storage
          exact ⟨extent,by simpa only [same] using disjoint⟩
        · exact recursiveMeaning.2 index old storage (by omega) oldAt storageAt
      · intro index layout memory lower layoutAt memoryAt
        have bound:=(List.getElem?_eq_some_iff.mp layoutAt).1
        omega

theorem source_layout_valid_meaning (program : CoreProgram) (state : CoreState)
    (computed : execution_core.layout_valid program state=.ok true) :
    program.layouts.val.length=state.allocations.val.length ∧
      (∀ (index : Nat) (layout : value_flow.AllocationLayout) (memory : execution_core.AllocationBytes), program.layouts.val[index]?=some layout → state.allocations.val[index]?=some memory →
        StorageExtent layout (alloc.vec.Vec.len memory.bytes) (alloc.vec.Vec.len memory.defined)) ∧
      SeparatedLayouts program.layouts := by
  unfold execution_core.layout_valid at computed
  have meaning:=source_layout_loop_meaning program state.allocations _ 0#usize computed
  have lengthEq : program.layouts.val.length=state.allocations.val.length := by
    have flag:=meaning.1
    scalar_tac
  refine ⟨lengthEq,?_,?_⟩
  · intro index layout memory layoutAt memoryAt
    exact (meaning.2 index layout memory (by scalar_tac) layoutAt memoryAt).1
  · intro firstIndex secondIndex first second firstAt secondAt different
    have firstInside:=(List.getElem?_eq_some_iff.mp firstAt).1
    have secondInside:=(List.getElem?_eq_some_iff.mp secondAt).1
    have firstMemoryInside : firstIndex < state.allocations.val.length := by omega
    have secondMemoryInside : secondIndex < state.allocations.val.length := by omega
    rcases Nat.lt_or_gt_of_ne different with before|after
    · have geometry:=(meaning.2 secondIndex second state.allocations.val[secondIndex] (by scalar_tac) secondAt
        (List.getElem?_eq_getElem secondMemoryInside)).2 firstIndex first before firstAt
      exact geometry.symm
    · exact (meaning.2 firstIndex first state.allocations.val[firstIndex] (by scalar_tac) firstAt
        (List.getElem?_eq_getElem firstMemoryInside)).2 secondIndex second after secondAt

end MemorySafetyExecution.Native
