/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.AllocationCompatible

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
open MemorySafety.Source
set_option autoImplicit false
set_option maxHeartbeats 8000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option debug.skipKernelTC false
namespace MemorySafetyExecution.Native

def LayoutDisjoint (left right : value_flow.AllocationLayout) : Prop :=
  left.base.val+left.bytes.val ≤ right.base.val ∨ right.base.val+right.bytes.val ≤ left.base.val

private theorem compatible_unroll (layouts : Layouts) (index : Usize) (layout : value_flow.AllocationLayout)
    (flag : Bool) (cursor : Usize) :
    execution_core.allocation_compatible_loop layouts index layout flag cursor =
      (do let control ← execution_core.allocation_compatible_loop.body layouts index layout flag cursor
          match control with
          | .cont pair => execution_core.allocation_compatible_loop layouts index layout pair.1 pair.2
          | .done result => .ok result) := by
  unfold execution_core.allocation_compatible_loop
  rw [loop.eq_def]
  congr 1
  funext control
  cases control with
  | cont pair => cases pair; rfl
  | done result => rfl

private theorem compatible_false (layouts : Layouts) (index : Usize) (layout : value_flow.AllocationLayout) (cursor : Usize) :
    execution_core.allocation_compatible_loop layouts index layout false cursor=.ok false := by
  rw [compatible_unroll]
  simp [execution_core.allocation_compatible_loop.body]

theorem source_compatible_meaning (layouts : Layouts) (index : Usize) (layout : value_flow.AllocationLayout)
    (flag : Bool) (cursor : Usize)
    (computed : execution_core.allocation_compatible_loop layouts index layout flag cursor=.ok true) :
    flag=true ∧ ∀ target other, cursor.val ≤ target → target < index.val →
      layouts.val[target]?=some other → LayoutDisjoint layout other := by
  generalize fuelEq : index.val-cursor.val=fuel
  induction fuel using Nat.strong_induction_on generalizing flag cursor with
  | h fuel ih =>
    cases flag with
    | false => simp only [compatible_false,ok.injEq,Bool.false_eq_true] at computed
    | true =>
      refine ⟨rfl,?_⟩
      rw [compatible_unroll] at computed
      by_cases inside : cursor.val < index.val
      · have test : cursor < index := by scalar_tac
        simp only [execution_core.allocation_compatible_loop.body,test,ite_true,Std.bind_assoc] at computed
        obtain ⟨other,otherRead,computed⟩ := source_bind_success _ _ _ computed
        obtain ⟨base,baseRun,computed⟩ := source_bind_success _ _ _ computed
        obtain ⟨otherBase,otherBaseRun,computed⟩ := source_bind_success _ _ _ computed
        obtain ⟨otherSize,otherSizeRun,computed⟩ := source_bind_success _ _ _ computed
        obtain ⟨otherEnd,otherEndRun,computed⟩ := source_bind_success _ _ _ computed
        obtain ⟨overlap,overlapRun,computed⟩ := source_bind_success _ _ _ computed
        obtain ⟨next,nextRun,computed⟩ := source_bind_success _ _ _ computed
        have nextVal : next.val=cursor.val+1 := by
          have spec := UScalar.add_spec (x:=cursor) (y:=1#usize) (by
            have bound:=index.hmax
            scalar_tac)
          simpa [nextRun,WP.spec_ok] using spec
        have recursiveMeaning := ih (index.val-next.val) (by omega) (decide (¬overlap=true)) next (by simpa only [bind_ok] using computed) rfl
        have overlapFalse : overlap=false := by cases overlap <;> simp_all
        have baseVal : base.val=layout.base.val := by
          have spec:=UScalar.cast_inBounds_spec .U128 layout.base (by scalar_tac)
          simpa only [baseRun,WP.spec_ok] using spec
        have otherBaseVal : otherBase.val=other.base.val := by
          have spec:=UScalar.cast_inBounds_spec .U128 other.base (by scalar_tac)
          simpa only [otherBaseRun,WP.spec_ok] using spec
        have otherSizeVal : otherSize.val=other.bytes.val := by
          have spec:=UScalar.cast_inBounds_spec .U128 other.bytes (by scalar_tac)
          simpa only [otherSizeRun,WP.spec_ok] using spec
        have otherEndVal : otherEnd.val=otherBase.val+otherSize.val := by
          have spec:=UScalar.add_spec (x:=otherBase) (y:=otherSize) (by scalar_tac)
          simpa only [otherEndRun,WP.spec_ok] using spec
        have disjoint : LayoutDisjoint layout other := by
          by_cases first : base < otherEnd
          · simp only [first,ite_true] at overlapRun
            obtain ⟨again,againRun,overlapRun⟩ := source_bind_success _ _ _ overlapRun
            obtain ⟨baseAgain,baseAgainRun,overlapRun⟩ := source_bind_success _ _ _ overlapRun
            obtain ⟨size,sizeRun,overlapRun⟩ := source_bind_success _ _ _ overlapRun
            obtain ⟨finish,finishRun,overlapRun⟩ := source_bind_success _ _ _ overlapRun
            have againVal : again.val=other.base.val := by
              have spec:=UScalar.cast_inBounds_spec .U128 other.base (by scalar_tac)
              simpa only [againRun,WP.spec_ok] using spec
            have baseAgainVal : baseAgain.val=layout.base.val := by
              have spec:=UScalar.cast_inBounds_spec .U128 layout.base (by scalar_tac)
              simpa only [baseAgainRun,WP.spec_ok] using spec
            have sizeVal : size.val=layout.bytes.val := by
              have spec:=UScalar.cast_inBounds_spec .U128 layout.bytes (by scalar_tac)
              simpa only [sizeRun,WP.spec_ok] using spec
            have finishVal : finish.val=baseAgain.val+size.val := by
              have spec:=UScalar.add_spec (x:=baseAgain) (y:=size) (by scalar_tac)
              simpa only [finishRun,WP.spec_ok] using spec
            have noOverlap : ¬again < finish := by simpa [overlapFalse] using overlapRun.symm
            unfold LayoutDisjoint
            left
            scalar_tac
          · unfold LayoutDisjoint
            right
            scalar_tac
        intro target selected lower upper selectedAt
        by_cases same : target=cursor.val
        · have otherAt := vec_index_success layouts cursor other (by
            simpa only [alloc.vec.Vec.index_slice_index] using otherRead)
          rw [same,otherAt] at selectedAt
          have sameOther : selected=other := by simpa only [Option.some.injEq] using selectedAt.symm
          subst selected
          exact disjoint
        · exact recursiveMeaning.2 target selected (by omega) upper selectedAt
      · intro target selected lower upper selectedAt
        omega

theorem source_allocation_compatible_meaning (program : CoreProgram) (index : Usize)
    (computed : execution_core.allocation_compatible program index=.ok true) :
    ∃ layout, program.layouts.val[index.val]?=some layout ∧
      ∀ target other, target < index.val → program.layouts.val[target]?=some other → LayoutDisjoint layout other := by
  unfold execution_core.allocation_compatible at computed
  obtain ⟨layout,read,computed⟩ := source_bind_success _ _ _ computed
  have atIndex := vec_index_success program.layouts index layout (by
    simpa only [alloc.vec.Vec.index_slice_index] using read)
  have meaning:=source_compatible_meaning program.layouts index layout true 0#usize computed
  exact ⟨layout,atIndex,fun target other bound selected => meaning.2 target other (by scalar_tac) bound selected⟩

end MemorySafetyExecution.Native
