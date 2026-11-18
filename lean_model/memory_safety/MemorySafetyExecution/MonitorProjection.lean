/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.SourceVectors
import MemoryWrite.TotalPrimitives

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Source

private def uniformSuffix (flags : List Bool) (defined : Bool) (index : Nat) : Option Bool :=
  if (flags.drop index).all (fun flag => flag == defined) then some defined else none

private theorem uniform_loop_spec (flags : Slice Bool) (defined : Bool) (index : Usize) :
    checked_execution.uniform_defined_loop flags defined index
      ⦃ output => output = uniformSuffix flags.val defined index.val ⦄ := by
  unfold checked_execution.uniform_defined_loop
  apply loop.spec_decr_nat
    (fun current : Usize => flags.val.length - current.val)
    (fun current : Usize =>
      uniformSuffix flags.val defined current.val = uniformSuffix flags.val defined index.val)
  · intro current invariant
    unfold checked_execution.uniform_defined_loop.body
    by_cases inside : current.val < flags.val.length
    · have test : current < Slice.len flags := by scalar_tac
      simp only [test, ite_true]
      step as ⟨entry, entryEq⟩
      by_cases equal : entry = defined
      · simp only [equal, bne_self_eq_false, Bool.false_eq_true, ite_false]
        step as ⟨next, nextVal⟩
        refine ⟨?_, by scalar_tac⟩
        rw [nextVal]
        unfold uniformSuffix at invariant ⊢
        rw [List.drop_eq_getElem_cons inside] at invariant
        simpa only [List.all_cons, ← entryEq, equal, beq_self_eq_true, Bool.true_and] using invariant
      · have different : (entry != defined) = true := by simp [equal]
        simp only [different, ite_true, WP.spec_ok]
        have unequal : (entry == defined) = false := by simp [equal]
        unfold uniformSuffix at invariant ⊢
        rw [List.drop_eq_getElem_cons inside] at invariant
        simpa only [List.all_cons, ← entryEq, unequal, Bool.false_and,
          Bool.false_eq_true, ite_false] using invariant
    · have test : ¬current < Slice.len flags := by scalar_tac
      simp only [test, ite_false, WP.spec_ok]
      simpa only [uniformSuffix, List.drop_eq_nil_of_le (show flags.val.length ≤ current.val by omega),
        List.all_nil, ite_true] using invariant
  · rfl

theorem source_uniform_defined (flags : Slice Bool) (defined : Bool)
    (computed : checked_execution.uniform_defined flags = .ok (some defined)) :
    flags.val ≠ [] ∧ ∀ flag ∈ flags.val, flag = defined := by
  unfold checked_execution.uniform_defined at computed
  simp only [core.slice.Slice.is_empty, bind_ok, Slice.length] at computed
  by_cases empty : flags.val = []
  · simp [empty] at computed
  · have nonempty : ¬flags.val.length = 0 := by simpa using empty
    simp only [nonempty] at computed
    obtain ⟨first, read, remaining⟩ := MemorySafety.Source.source_bind_success _ _ _ computed
    have specification := uniform_loop_spec flags first 1#usize
    have suffix : some defined = uniformSuffix flags.val first 1 := by
      simpa [remaining, WP.spec_ok] using specification
    have inside : 0 < flags.val.length := List.length_pos_iff.mpr empty
    have lookup : Slice.index_usize flags 0#usize = .ok flags.val[0] := by
      simp [Slice.index_usize, Slice.getElem?_Usize_eq, List.getElem?_eq_getElem inside]
    have firstEq : flags.val[0] = first := by simpa only [lookup, ok.injEq] using read
    unfold uniformSuffix at suffix
    split at suffix
    · rename_i all
      have same : defined = first := Option.some.inj suffix
      refine ⟨empty, ?_⟩
      intro flag member
      have decomposition : flags.val = flags.val[0] :: flags.val.drop 1 := by
        simpa only [List.drop_zero] using List.drop_eq_getElem_cons inside
      rw [decomposition] at member
      rcases List.mem_cons.mp member with member | member
      · exact member.trans (firstEq.trans same.symm)
      · have equal := List.all_eq_true.mp all flag member
        have equal' : flag = first := by simpa using equal
        exact equal'.trans same.symm
    · cases suffix

theorem source_monitor_allocation (layout : value_flow.AllocationLayout)
    (memory : execution_core.AllocationBytes) (allocation : memory_safety.Allocation)
    (computed : checked_execution.monitor_allocation layout memory = .ok (some allocation)) :
    allocation.base = layout.base ∧ allocation.bytes = layout.bytes ∧
      memory.defined.val ≠ [] ∧ ∀ flag ∈ memory.defined.val, flag = allocation.initially_defined := by
  unfold checked_execution.monitor_allocation at computed
  obtain ⟨result, checked, remaining⟩ := MemorySafety.Source.source_bind_success _ _ _ computed
  cases result with
  | none => simp at remaining
  | some defined =>
    simp only [ok.injEq, Option.some.injEq] at remaining
    subst allocation
    obtain ⟨nonempty, uniform⟩ := source_uniform_defined memory.defined.deref defined checked
    exact ⟨rfl, rfl, by simpa only [alloc.vec.Vec.deref, Slice.from_val] using nonempty, by simpa only [alloc.vec.Vec.deref, Slice.from_val] using uniform⟩

end MemorySafetyExecution.Source
