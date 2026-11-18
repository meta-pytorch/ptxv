/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.SourceTypes
import MemorySafety.Code.Funs
import MemorySafety.SourceCertificate

open Aeneas Aeneas.Std Result
open PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace ValueFlow.Source

deriving instance DecidableEq for ptxv_memory_write.value_flow.Pointer,
  ptxv_memory_write.value_flow.ThreadIds, ptxv_memory_write.machine_registers.LaneRegisters,
  ptxv_memory_write.value_flow.Registers

theorem source_pointer_equality (left right : V.Pointer) :
    ptxv_memory_write.value_flow.Pointer.Insts.CoreCmpPartialEqPointer.eq left right =
      .ok (decide (left = right)) := by
  rcases left with ⟨allocation, offset⟩
  rcases right with ⟨otherAllocation, otherOffset⟩
  by_cases a : allocation = otherAllocation <;> by_cases o : offset = otherOffset <;>
    simp_all [ptxv_memory_write.value_flow.Pointer.Insts.CoreCmpPartialEqPointer.eq]

theorem source_pair_equality (left right : U16 × U64) :
    Pair.Insts.CoreCmpPartialEqPair.eq core.cmp.PartialEqU16 core.cmp.PartialEqU64 left right =
      .ok (decide (left = right)) := by
  rcases left with ⟨key, value⟩
  rcases right with ⟨otherKey, otherValue⟩
  by_cases keyEq : key = otherKey <;> by_cases valueEq : value = otherValue <;>
    simp_all [Pair.Insts.CoreCmpPartialEqPair.eq, core.cmp.impls.PartialEqU16.eq,
      core.cmp.impls.PartialEqU64.eq, liftFun2]

theorem source_vector_equality {α : Type} [DecidableEq α]
    (implementation : core.cmp.PartialEq α α)
    (element : ∀ left right, implementation.eq left right = .ok (decide (left = right)))
    (left right : alloc.vec.Vec α) :
    alloc.vec.partial_eq.PartialEqVec.eq implementation left right =
      .ok (decide (left = right)) := by
  have equality := MemorySafety.Source.vector_model_equality implementation id element left right
  simpa only [List.map_id, ←alloc.vec.Vec.eq_iff] using equality

theorem source_lane_equality (left right : ptxv_memory_write.machine_registers.LaneRegisters) :
    ptxv_memory_write.machine_registers.LaneRegisters.Insts.CoreCmpPartialEqLaneRegisters.eq left right =
      .ok (decide (left = right)) := by
  have scalar (left right : U64) : core.cmp.PartialEqU64.eq left right =
      .ok (decide (left = right)) := rfl
  have regs := source_vector_equality core.cmp.PartialEqU64 scalar left.regs right.regs
  have preds := source_vector_equality core.cmp.PartialEqU64 scalar left.preds right.preds
  have pdef := source_vector_equality core.cmp.PartialEqU64 scalar left.pdef right.pdef
  have written := source_vector_equality core.cmp.PartialEqU64 scalar left.written right.written
  have vund := source_vector_equality
    (Pair.Insts.CoreCmpPartialEqPair core.cmp.PartialEqU16 core.cmp.PartialEqU64)
    source_pair_equality left.vund right.vund
  rcases left with ⟨lr, lp, ld, lw, lu⟩
  rcases right with ⟨rr, rp, rd, rw, ru⟩
  by_cases r : lr = rr <;> by_cases p : lp = rp <;> by_cases d : ld = rd <;>
    by_cases w : lw = rw <;> by_cases u : lu = ru <;>
    simp_all [ptxv_memory_write.machine_registers.LaneRegisters.Insts.CoreCmpPartialEqLaneRegisters.eq]

theorem source_array_equality {n : Usize} (left right : Array U32 n) :
    core.array.equality.PartialEqArray.eq core.cmp.PartialEqU32 left right =
      .ok (decide (left = right)) := by
  unfold core.array.equality.PartialEqArray.eq
  have each (pair : U32 × U32) : (do
      let different ← core.cmp.PartialEqU32.ne pair.1 pair.2
      (Result.ok (¬different) : Result Bool)) = .ok (decide (pair.1 = pair.2)) := by
    simp [core.cmp.impls.PartialEqU32.ne, liftFun2,
      MemorySafety.Source.scalar_val_injective]
  simp_rw [each]
  rw [MemorySafety.Source.allM_pure]
  have equalLength : left.val.length = right.val.length := by simp
  have equality := MemorySafety.Source.zipped_model_equality id left.val right.val
  simp only [List.map_id, equalLength, ite_true, id_eq] at equality
  simp only [Array.length, Array.property, ite_true]
  rw [equality]
  simp only [←Array.eq_iff]

theorem source_ids_equality (left right : V.ThreadIds) :
    ptxv_memory_write.value_flow.ThreadIds.Insts.CoreCmpPartialEqThreadIds.eq left right =
      .ok (decide (left = right)) := by
  have tid := source_array_equality left.tid right.tid
  have ctaid := source_array_equality left.ctaid right.ctaid
  have ntid := source_array_equality left.ntid right.ntid
  have nctaid := source_array_equality left.nctaid right.nctaid
  rcases left with ⟨lt, lc, ln, lg, ll⟩
  rcases right with ⟨rt, rc, rn, rg, rl⟩
  by_cases t : lt = rt <;> by_cases c : lc = rc <;> by_cases n : ln = rn <;>
    by_cases g : lg = rg <;> by_cases l : ll = rl <;>
    simp_all [ptxv_memory_write.value_flow.ThreadIds.Insts.CoreCmpPartialEqThreadIds.eq]

theorem source_pointer_tags_loop (left right : Slice (Option V.Pointer))
    (sameLength : left.val.length = right.val.length) (equal : Bool) (index : Usize) :
    ptxv_memory_write.value_flow.pointer_tags_equal_loop left right equal index =
      .ok (equal && decide (left.val.drop index.val = right.val.drop index.val)) := by
  generalize fuelEq : left.val.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing equal index with
  | h fuel ih =>
    have unroll : ptxv_memory_write.value_flow.pointer_tags_equal_loop left right equal index = (do
        let output ← ptxv_memory_write.value_flow.pointer_tags_equal_loop.body left right equal index
        match output with
        | .cont pair => ptxv_memory_write.value_flow.pointer_tags_equal_loop left right pair.1 pair.2
        | .done value => .ok value) := by
      unfold ptxv_memory_write.value_flow.pointer_tags_equal_loop
      rw [loop.eq_def]
      congr 1
      funext output
      cases output with
      | cont pair => cases pair; rfl
      | done value => rfl
    rw [unroll]
    cases equal with
    | false => simp [ptxv_memory_write.value_flow.pointer_tags_equal_loop.body]
    | true =>
      by_cases inside : index.val < left.val.length
      · have rightInside : index.val < right.val.length := by omega
        have bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
          have := Slice.length_ineq left
          scalar_tac
        let next : Usize := Usize.ofNatCore (index.val + 1) bound
        have nextVal : next.val = index.val + 1 := by simp [next]
        have increment := MemorySafety.Source.usize_increment index bound
        have leftRead : Slice.index_usize left index = .ok left.val[index.val]! := by
          have read : Slice.index_usize left index = .ok left.val[index.val] := by
            simp [Slice.index_usize, Slice.getElem?_Usize_eq, List.getElem?_eq_getElem inside]
          exact read.trans (congrArg Result.ok
            (List.Inhabited_getElem_eq_getElem! left.val index.val inside))
        have rightRead : Slice.index_usize right index = .ok right.val[index.val]! := by
          have read : Slice.index_usize right index = .ok right.val[index.val] := by
            simp [Slice.index_usize, Slice.getElem?_Usize_eq, List.getElem?_eq_getElem rightInside]
          exact read.trans (congrArg Result.ok
            (List.Inhabited_getElem_eq_getElem! right.val index.val rightInside))
        have test : index < Slice.len left := by scalar_tac
        have choice (a b : Option V.Pointer) : (match a with
            | none => match b with
              | none => Result.ok true
              | some _ => Result.ok false
            | some value => match b with
              | none => Result.ok false
              | some other => ptxv_memory_write.value_flow.Pointer.Insts.CoreCmpPartialEqPointer.eq value other) =
              .ok (decide (a = b)) := by
          cases a <;> cases b <;> simp [source_pointer_equality]
        have body : ptxv_memory_write.value_flow.pointer_tags_equal_loop.body left right true index =
            .ok (.cont (decide (left.val[index.val]! = right.val[index.val]!), next)) := by
          simp only [ptxv_memory_write.value_flow.pointer_tags_equal_loop.body, test,
            ite_true, leftRead, rightRead, bind_ok, increment]
          generalize leftHead : left.val[index.val]! = a
          generalize rightHead : right.val[index.val]! = b
          cases a <;> cases b <;> simp [source_pointer_equality, next]
        rw [body]
        simp only [bind_ok]
        rw [ih (left.val.length - next.val) (by omega) _ next rfl]
        simp only [nextVal, Bool.true_and, ok.injEq]
        apply Bool.eq_iff_iff.mpr
        simp only [Bool.and_eq_true, decide_eq_true_eq]
        rw [List.drop_eq_getElem_cons inside, List.drop_eq_getElem_cons rightInside]
        simp only [List.cons.injEq,
          List.Inhabited_getElem_eq_getElem! left.val index.val inside,
          List.Inhabited_getElem_eq_getElem! right.val index.val rightInside]
      · have test : ¬index < Slice.len left := by scalar_tac
        have rightOutside : right.val.length ≤ index.val := by omega
        simp [ptxv_memory_write.value_flow.pointer_tags_equal_loop.body, test,
          List.drop_eq_nil_of_le (show left.val.length ≤ index.val by omega),
          List.drop_eq_nil_of_le rightOutside]

theorem source_pointer_tags_equality (left right : Slice (Option V.Pointer)) :
    ptxv_memory_write.value_flow.pointer_tags_equal left right =
      .ok (decide (left.val = right.val)) := by
  unfold ptxv_memory_write.value_flow.pointer_tags_equal
  by_cases sameLength : left.val.length = right.val.length
  · have lengths : Slice.len left = Slice.len right := by scalar_tac
    simp only [lengths, decide_true]
    rw [source_pointer_tags_loop left right sameLength]
    simp
  · have lengths : Slice.len left ≠ Slice.len right := by
      intro equality
      exact sameLength (by simpa using congrArg UScalar.val equality)
    have different : left.val ≠ right.val := by
      intro equality
      exact sameLength (congrArg List.length equality)
    simp only [lengths, decide_false, different]
    unfold ptxv_memory_write.value_flow.pointer_tags_equal_loop
    rw [loop.eq_def]
    simp only [ptxv_memory_write.value_flow.pointer_tags_equal_loop.body,
      Bool.false_eq_true, ite_false, bind_ok]

theorem source_registers_equality (left right : V.Registers) :
    ptxv_memory_write.value_flow.registers_equal left right = .ok (decide (left = right)) := by
  have raw := source_lane_equality left.raw right.raw
  have pointers := source_pointer_tags_equality
    (alloc.vec.Vec.deref left.pointers) (alloc.vec.Vec.deref right.pointers)
  rcases left with ⟨lr, lp⟩
  rcases right with ⟨rr, rp⟩
  by_cases r : lr = rr <;> by_cases p : lp = rp <;>
    simp_all [ptxv_memory_write.value_flow.registers_equal, alloc.vec.Vec.deref,
      alloc.vec.Vec.eq_iff]

theorem source_registers_trait_equality (left right : V.Registers) :
    ptxv_memory_write.value_flow.Registers.Insts.CoreCmpPartialEqRegisters.eq left right =
      .ok (decide (left = right)) :=
  source_registers_equality left right

end ValueFlow.Source
