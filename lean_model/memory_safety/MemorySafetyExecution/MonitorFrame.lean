/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.MonitorProjection

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 5000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Source

open MemorySafety.Source

def ProjectedAllocation (program : execution_core.Program) (index : Nat)
    (allocation : memory_safety.Allocation) : Prop :=
  ∃ layout memory,
    program.layouts.val[index]? = some layout ∧
    program.initial.allocations.val[index]? = some memory ∧
    allocation.base = layout.base ∧ allocation.bytes = layout.bytes ∧
    memory.defined.val ≠ [] ∧ ∀ flag ∈ memory.defined.val, flag = allocation.initially_defined

private theorem source_allocations_loop (program : execution_core.Program)
    (accumulated output : alloc.vec.Vec memory_safety.Allocation) (index : Usize)
    (bound : index.val ≤ program.layouts.val.length)
    (correct : List.Forall₂ (ProjectedAllocation program) (List.range index.val) accumulated.val)
    (computed : checked_execution.monitor_allocations_loop program accumulated index = .ok (some output)) :
    List.Forall₂ (ProjectedAllocation program) (List.range program.layouts.val.length) output.val := by
  generalize fuelEq : program.layouts.val.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing accumulated index with
  | h fuel inductionResult =>
    have unroll : checked_execution.monitor_allocations_loop program accumulated index = (do
        let result ← checked_execution.monitor_allocations_loop.body program accumulated index
        match result with
        | .cont after => checked_execution.monitor_allocations_loop program after.1 after.2
        | .done result => .ok result) := by
      unfold checked_execution.monitor_allocations_loop
      rw [loop.eq_def]
      congr 1
      funext result
      cases result with
      | cont pair => cases pair; rfl
      | done result => rfl
    rw [unroll] at computed
    by_cases inside : index.val < program.layouts.val.length
    · have test : index < alloc.vec.Vec.len program.layouts := by scalar_tac
      simp only [checked_execution.monitor_allocations_loop.body, test, ite_true,
        alloc.vec.Vec.index_slice_index, Std.bind_assoc] at computed
      obtain ⟨layout, layoutRead, computed⟩ := source_bind_success _ _ _ computed
      obtain ⟨memory, memoryRead, computed⟩ := source_bind_success _ _ _ computed
      obtain ⟨result, projected, computed⟩ := source_bind_success _ _ _ computed
      cases result with
      | none => simp at computed
      | some allocation =>
        simp only [Std.bind_assoc, bind_ok] at computed
        obtain ⟨afterPrefix, pushed, computed⟩ := source_bind_success _ _ _ computed
        obtain ⟨afterIndex, advanced, computed⟩ := source_bind_success _ _ _ computed
        have incrementBound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
          have size := alloc.vec.Vec.len_ineq program.layouts
          scalar_tac
        have increment := source_usize_increment index incrementBound
        have afterVal : afterIndex.val = index.val + 1 := by
          have equal : Usize.ofNatCore (index.val + 1) incrementBound = afterIndex := by
            simpa only [increment, ok.injEq] using advanced
          rw [← equal]
          rfl
        have relation : ProjectedAllocation program index.val allocation :=
          ⟨layout, memory, vec_index_success _ _ _ layoutRead,
            vec_index_success _ _ _ memoryRead, source_monitor_allocation layout memory allocation projected⟩
        have correctAfter : List.Forall₂ (ProjectedAllocation program)
            (List.range afterIndex.val) afterPrefix.val := by
          rw [afterVal, List.range_succ, vec_push_success _ _ _ pushed]
          exact List.rel_append correct (.cons relation .nil)
        exact inductionResult (program.layouts.val.length - afterIndex.val) (by omega)
          afterPrefix afterIndex (by omega) correctAfter computed rfl
    · have test : ¬index < alloc.vec.Vec.len program.layouts := by scalar_tac
      simp only [checked_execution.monitor_allocations_loop.body, test, ite_false, bind_ok,
        ok.injEq, Option.some.injEq] at computed
      have equal : index.val = program.layouts.val.length := by omega
      rw [← computed, ← equal]
      exact correct

theorem source_monitor_allocations (program : execution_core.Program)
    (allocations : alloc.vec.Vec memory_safety.Allocation)
    (computed : checked_execution.monitor_allocations program = .ok (some allocations)) :
    List.Forall₂ (ProjectedAllocation program) (List.range program.layouts.val.length) allocations.val := by
  apply source_allocations_loop program (alloc.vec.Vec.new _) allocations 0#usize
  · simp
  · exact .nil
  · exact computed

private theorem source_threads_loop (count : Usize)
    (threads result : alloc.vec.Vec (alloc.vec.Vec memory_safety.Instruction)) (thread : Usize)
    (computed : checked_execution.monitor_threads_loop count threads thread = .ok result) :
    result.val.map (fun code => code.val) = threads.val.map (fun code => code.val) ++
      List.replicate (count.val - thread.val) ([] : List memory_safety.Instruction) := by
  generalize fuelEq : count.val - thread.val = fuel
  induction fuel using Nat.strong_induction_on generalizing threads thread with
  | h fuel ih =>
    have unroll : checked_execution.monitor_threads_loop count threads thread = (do
        let result ← checked_execution.monitor_threads_loop.body count threads thread
        match result with
        | .cont pair => checked_execution.monitor_threads_loop count pair.1 pair.2
        | .done result => .ok result) := by
      unfold checked_execution.monitor_threads_loop
      rw [loop.eq_def]
      congr 1
      funext result
      cases result with
      | cont pair => cases pair; rfl
      | done output => rfl
    rw [unroll] at computed
    by_cases inside : thread.val < count.val
    · have bound : thread.val + 1 < 2 ^ UScalarTy.Usize.numBits := by scalar_tac
      let after : Usize := Usize.ofNatCore (thread.val + 1) bound
      have afterVal : after.val = thread.val + 1 := by simp [after]
      have increment := source_usize_increment thread bound
      have test : thread < count := by scalar_tac
      simp only [checked_execution.monitor_threads_loop.body, test, ite_true] at computed
      simp only [Std.bind_assoc, bind_ok] at computed
      obtain ⟨next, pushed, finished⟩ := source_bind_success _ _ _ computed
      rw [increment] at finished
      simp only [bind_ok] at finished
      have tail := ih (count.val - after.val) (by omega) next after finished rfl
      rw [tail, vec_push_success threads next (alloc.vec.Vec.new _) pushed, List.map_append, List.map_cons,
        List.map_nil, alloc.vec.Vec.new, alloc.vec.Vec.from_val]
      have countEq : count.val - thread.val = (count.val - after.val) + 1 := by omega
      rw [← fuelEq, countEq, List.replicate_succ]
      simp [List.append_assoc]
    · have test : ¬thread < count := by scalar_tac
      simp only [checked_execution.monitor_threads_loop.body, test, ite_false,
        bind_ok, ok.injEq] at computed
      rw [← computed]
      have fuelZero : fuel = 0 := by omega
      simp [fuelZero]


theorem source_monitor_threads (count : Usize)
    (threads : alloc.vec.Vec (alloc.vec.Vec memory_safety.Instruction))
    (computed : checked_execution.monitor_threads count = .ok threads) :
    threads.val.map (fun code => code.val) = List.replicate count.val [] := by
  simpa using source_threads_loop count (alloc.vec.Vec.new _) threads 0#usize computed


theorem source_monitor_program (program : execution_core.Program) (frame : memory_safety.Program)
    (computed : checked_execution.monitor_program program = .ok (some frame)) :
    execution_core.layout_valid program program.initial = .ok true ∧
      List.Forall₂ (ProjectedAllocation program) (List.range program.layouts.val.length)
        frame.allocations.val ∧
      frame.threads.val.map (fun code => code.val) =
        List.replicate program.initial.threads.val.length [] := by
  unfold checked_execution.monitor_program at computed
  obtain ⟨valid, tested, computed⟩ := source_bind_success _ _ _ computed
  cases valid with
  | false => simp at computed
  | true =>
    simp only [ite_true] at computed
    obtain ⟨result, allocationsComputed, computed⟩ := source_bind_success _ _ _ computed
    cases result with
    | none => simp at computed
    | some allocations =>
      obtain ⟨threads, threadsComputed, computed⟩ := source_bind_success _ _ _ computed
      simp only [ok.injEq, Option.some.injEq] at computed
      subst frame
      exact ⟨tested, source_monitor_allocations program allocations allocationsComputed,
        by simpa only [alloc.vec.Vec.len_val] using
          source_monitor_threads (alloc.vec.Vec.len program.initial.threads) threads threadsComputed⟩

end MemorySafetyExecution.Source
