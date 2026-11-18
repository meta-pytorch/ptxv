/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlowDependencies.Comparison

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace ValueFlowDependencies.Source

theorem source_metadata_equal (left right : Metadata) :
    value_flow_dependencies.metadata_equal left right = .ok (decide (left = right)) := by
  rcases left with ⟨captured, history, tainted, defined⟩
  rcases right with ⟨otherCaptured, otherHistory, otherTainted, otherDefined⟩
  by_cases c : captured = otherCaptured <;> by_cases t : tainted = otherTainted <;>
    by_cases d : defined = otherDefined <;> cases history <;> cases otherHistory <;>
    simp_all [value_flow_dependencies.metadata_equal]

private theorem metadata_body (left right : Slice Metadata)
    (sameLength : left.val.length = right.val.length) (equal : Bool) (index : Usize) :
    value_flow_dependencies.metadata_slice_equal_loop.body left right equal index =
      comparisonBody left.val right.val equal index := by
  unfold value_flow_dependencies.metadata_slice_equal_loop.body comparisonBody
  by_cases inside : index.val < left.val.length
  · have rightInside : index.val < right.val.length := by omega
    have test : index < Slice.len left := by scalar_tac
    simp only [test, inside, ite_true]
    cases equal with
    | false => simp
    | true =>
      have leftRead : Slice.index_usize left index = .ok left.val[index.val] := by
        simp [Slice.index_usize, Slice.getElem?_Usize_eq, List.getElem?_eq_getElem inside]
        rfl
      have rightRead : Slice.index_usize right index = .ok right.val[index.val] := by
        simp [Slice.index_usize, Slice.getElem?_Usize_eq, List.getElem?_eq_getElem rightInside]
      simp only [ite_true, leftRead, rightRead, bind_ok, source_metadata_equal,
        List.getElem?_eq_getElem inside, List.getElem?_eq_getElem rightInside,
        Option.some.injEq, Bool.true_and]
      rfl
  · have test : ¬index < Slice.len left := by scalar_tac
    simp only [test, inside, ite_false]

theorem source_metadata_slice_equal (left right : Slice Metadata) :
    value_flow_dependencies.metadata_slice_equal left right = .ok (decide (left.val = right.val)) := by
  unfold value_flow_dependencies.metadata_slice_equal
  by_cases sameLength : left.val.length = right.val.length
  · have lengths : Slice.len left = Slice.len right := by scalar_tac
    simp only [lengths, bne_self_eq_false, Bool.false_eq_true, ite_false]
    unfold value_flow_dependencies.metadata_slice_equal_loop
    rw [comparison_loop_from_body left.val right.val (by have := Slice.length_ineq left; scalar_tac) sameLength
      _ (metadata_body left right sameLength)]
    simp
  · have lengths : Slice.len left ≠ Slice.len right := by
      intro equality
      exact sameLength (by simpa using congrArg UScalar.val equality)
    have different : left.val ≠ right.val := by
      intro equality
      exact sameLength (congrArg List.length equality)
    simp [lengths, different]

theorem source_registers_equal (left right : Registers) :
    value_flow_dependencies.registers_equal left right = .ok (decide (left = right)) := by
  unfold value_flow_dependencies.registers_equal
  rw [source_metadata_slice_equal]
  cases left with
  | mk virtuals predicates =>
    cases right with
    | mk otherVirtuals otherPredicates =>
      by_cases v : virtuals = otherVirtuals <;> by_cases p : predicates = otherPredicates <;>
        simp_all [source_metadata_slice_equal, alloc.vec.Vec.deref, alloc.vec.Vec.eq_iff]

private theorem threads_body (left right : alloc.vec.Vec (Registers))
    (sameLength : left.val.length = right.val.length) (equal : Bool) (index : Usize) :
    value_flow_dependencies.state_equal_loop0.body left right equal index =
      comparisonBody left.val right.val equal index := by
  unfold value_flow_dependencies.state_equal_loop0.body comparisonBody
  by_cases inside : index.val < left.val.length
  · have rightInside : index.val < right.val.length := by omega
    have test : index < alloc.vec.Vec.len left := by scalar_tac
    simp only [test, inside, ite_true]
    cases equal with
    | false => simp
    | true =>
      have leftRead : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (Registers)) left index = .ok left.val[index.val] := by
        change alloc.vec.Vec.index_usize left index = .ok left.val[index.val]
        simp [alloc.vec.Vec.index_usize,
          List.getElem?_eq_getElem inside]
        rfl
      have rightRead : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (Registers)) right index = .ok right.val[index.val] := by
        change alloc.vec.Vec.index_usize right index = .ok right.val[index.val]
        simp [alloc.vec.Vec.index_usize,
          List.getElem?_eq_getElem rightInside]
      simp only [ite_true, leftRead, rightRead, bind_ok, source_registers_equal,
        List.getElem?_eq_getElem inside, List.getElem?_eq_getElem rightInside,
        Option.some.injEq, Bool.true_and]
      rfl
  · have test : ¬index < alloc.vec.Vec.len left := by scalar_tac
    simp only [test, inside, ite_false]

private theorem threads_loop (left right : alloc.vec.Vec (Registers))
    (sameLength : left.val.length = right.val.length) (equal : Bool) (index : Usize) :
    value_flow_dependencies.state_equal_loop0 left right equal index =
      .ok (equal && decide (left.val.drop index.val = right.val.drop index.val)) := by
  unfold value_flow_dependencies.state_equal_loop0
  exact comparison_loop_from_body left.val right.val
    (by have := alloc.vec.Vec.len_ineq left; scalar_tac) sameLength
    _ (threads_body left right sameLength) equal index

private theorem memory_body (left right : alloc.vec.Vec (alloc.vec.Vec Metadata))
    (sameLength : left.val.length = right.val.length) (equal : Bool) (index : Usize) :
    value_flow_dependencies.state_equal_loop1.body left right equal index =
      comparisonBody left.val right.val equal index := by
  unfold value_flow_dependencies.state_equal_loop1.body comparisonBody
  by_cases inside : index.val < left.val.length
  · have rightInside : index.val < right.val.length := by omega
    have test : index < alloc.vec.Vec.len left := by scalar_tac
    simp only [test, inside, ite_true]
    cases equal with
    | false => simp
    | true =>
      have leftRead : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (alloc.vec.Vec Metadata)) left index = .ok left.val[index.val] := by
        change alloc.vec.Vec.index_usize left index = .ok left.val[index.val]
        simp [alloc.vec.Vec.index_usize,
          List.getElem?_eq_getElem inside]
        rfl
      have rightRead : alloc.vec.Vec.index (core.slice.index.SliceIndexUsizeSlice (alloc.vec.Vec Metadata)) right index = .ok right.val[index.val] := by
        change alloc.vec.Vec.index_usize right index = .ok right.val[index.val]
        simp [alloc.vec.Vec.index_usize,
          List.getElem?_eq_getElem rightInside]
      simp only [ite_true, leftRead, rightRead, bind_ok, source_metadata_slice_equal, alloc.vec.Vec.deref, Slice.from_val, alloc.vec.Vec.eq_iff,
        List.getElem?_eq_getElem inside, List.getElem?_eq_getElem rightInside,
        Option.some.injEq, Bool.true_and]
      rfl
  · have test : ¬index < alloc.vec.Vec.len left := by scalar_tac
    simp only [test, inside, ite_false]

private theorem memory_loop (left right : alloc.vec.Vec (alloc.vec.Vec Metadata))
    (sameLength : left.val.length = right.val.length) (equal : Bool) (index : Usize) :
    value_flow_dependencies.state_equal_loop1 left right equal index =
      .ok (equal && decide (left.val.drop index.val = right.val.drop index.val)) := by
  unfold value_flow_dependencies.state_equal_loop1
  exact comparison_loop_from_body left.val right.val
    (by have := alloc.vec.Vec.len_ineq left; scalar_tac) sameLength
    _ (memory_body left right sameLength) equal index

theorem source_state_equal (left right : State) :
    value_flow_dependencies.state_equal left right = .ok (decide (left = right)) := by
  unfold value_flow_dependencies.state_equal
  by_cases threadsLength : left.threads.val.length = right.threads.val.length
  · have threadsLen : alloc.vec.Vec.len left.threads = alloc.vec.Vec.len right.threads := by scalar_tac
    simp only [threadsLen, bne_self_eq_false, Bool.false_eq_true, ite_false]
    by_cases memoryLength : left.memory.val.length = right.memory.val.length
    · have memoryLen : alloc.vec.Vec.len left.memory = alloc.vec.Vec.len right.memory := by scalar_tac
      simp only [memoryLen, bne_self_eq_false, Bool.false_eq_true, ite_false]
      rw [threads_loop left.threads right.threads threadsLength]
      simp only [Bool.true_and, bind_ok]
      rw [memory_loop left.memory right.memory memoryLength]
      cases left with
      | mk threads memory =>
        cases right with
        | mk otherThreads otherMemory =>
          by_cases t : threads = otherThreads <;> by_cases m : memory = otherMemory <;>
            simp_all [alloc.vec.Vec.eq_iff]
    · have memoryLen : alloc.vec.Vec.len left.memory ≠ alloc.vec.Vec.len right.memory := by
        intro same
        exact memoryLength (by simpa using congrArg UScalar.val same)
      have different : left ≠ right := by
        intro same
        subst right
        exact memoryLength rfl
      simp [memoryLen, different]
  · have threadsLen : alloc.vec.Vec.len left.threads ≠ alloc.vec.Vec.len right.threads := by
      intro same
      exact threadsLength (by simpa using congrArg UScalar.val same)
    have different : left ≠ right := by
      intro same
      subst right
      exact threadsLength rfl
    simp [threadsLen, different]

end ValueFlowDependencies.Source
