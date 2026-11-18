/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.IndependentModel
import MemorySafetyExecution.Native.LoadTotal

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
set_option autoImplicit false
set_option maxHeartbeats 8000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option debug.skipKernelTC false
namespace MemorySafetyExecution.Native.Independent

private theorem model_byte (memory : execution_core.AllocationBytes) (index : Nat) :
    (memoryModel memory).bytes[index]?.getD 0=
      (memory.bytes.val[index]?.getD 0#u8).val := by
  simp only [memoryModel,List.getElem?_map]
  cases memory.bytes.val[index]? <;> rfl

private theorem fold_step (memory : MachineModel.Memory) (offset count index value : Nat)
    (inside : index<count) :
    MachineModel.loadFold memory offset (count-index) index value=
      MachineModel.loadFold memory offset (count-(index+1)) (index+1)
        (value ||| ((memory.bytes[offset+index]?.getD 0)*2^(8*index)%2^64)) := by
  have countEq : count-index=(count-(index+1))+1 := by omega
  conv_lhs => rw [countEq]
  rfl

theorem source_load_loop_model (memory : execution_core.AllocationBytes) (offset count : Usize)
    (bounded : offset.val+count.val ≤ memory.bytes.val.length) (small : count.val ≤ 8)
    (value : U64) (index : Usize) (within : index.val ≤ count.val) :
    execution_core.load_value_loop offset count memory value index ⦃ output =>
      output.val=MachineModel.loadFold (memoryModel memory) offset.val
        (count.val-index.val) index.val value.val ⦄ := by
  unfold execution_core.load_value_loop
  apply loop.spec_decr_nat (fun current : U64×Usize => count.val-current.2.val)
    (fun current : U64×Usize => current.2.val ≤ count.val ∧
      MachineModel.loadFold (memoryModel memory) offset.val
        (count.val-current.2.val) current.2.val current.1.val=
      MachineModel.loadFold (memoryModel memory) offset.val
        (count.val-index.val) index.val value.val)
  · rintro ⟨current,cursor⟩ ⟨progress,invariant⟩
    change cursor.val ≤ count.val at progress
    unfold execution_core.load_value_loop.body
    by_cases inside : cursor<count
    · simp only [inside,ite_true]
      step as ⟨position,positionVal⟩
      step as ⟨byte,byteEq⟩
      simp only [lift,bind_ok]
      step as ⟨shift,shiftVal⟩
      step as ⟨part,partVal,partBits⟩
      step as ⟨next,nextVal⟩
      refine ⟨by scalar_tac,?_,by scalar_tac⟩
      have byteBound : position.val<memory.bytes.val.length := by scalar_tac
      have selected : (memoryModel memory).bytes[offset.val+cursor.val]?.getD 0=byte.val := by
        rw [model_byte,←positionVal]
        simpa only [List.getElem?_eq_getElem byteBound,Option.getD_some] using congrArg UScalar.val byteEq.symm
      have shifted : part.val=(byte.val*2^(8*cursor.val))%2^64 := by
        simp [partVal,shiftVal,Nat.shiftLeft_eq,U64.size,U64.numBits,Nat.mul_comm]
      simp only [nextVal,UScalar.val_or,shifted]
      rw [←selected,←fold_step (memoryModel memory) offset.val count.val cursor.val current.val
        (by scalar_tac)]
      exact invariant
    · simp only [inside,ite_false,WP.spec_ok]
      have same : cursor.val=count.val := by scalar_tac
      simpa only [same,Nat.sub_self,MachineModel.loadFold] using invariant
  · exact ⟨within,rfl⟩


theorem source_load_value_model (state : CoreState) (access : execution_core.ResolvedAccess)
    (memory : execution_core.AllocationBytes) (signed : Bool)
    (selected : state.allocations.val[access.allocation.val]?=some memory)
    (bounded : access.offset.val+access.width.val ≤ memory.bytes.val.length)
    (nonzero : 0<access.width.val) (small : access.width.val ≤ 8) :
    execution_core.load_value state access signed ⦃ output =>
      output.val=MachineModel.load (memoryModel memory) access.offset.val access.width.val signed ⦄ := by
  unfold execution_core.load_value
  simp only [alloc.vec.Vec.index_slice_index,alloc.vec.Vec.index_usize,
    alloc.vec.Vec.getElem?_Nat_eq,selected,bind_ok]
  step with source_load_loop_model memory access.offset access.width bounded small
    0#u64 0#usize (by scalar_tac) as ⟨value,rawEq⟩
  simp only [Nat.sub_zero,show (0#usize).val=0 from rfl,show (0#u64).val=0 from rfl] at rawEq
  cases signed with
  | false => simpa only [Bool.false_eq_true,ite_false,WP.spec_ok,MachineModel.load,
      MachineModel.signedLoad,false_and] using rawEq
  | true =>
    simp only [ite_true]
    by_cases narrow : access.width<8#usize
    · simp only [narrow,ite_true]
      step as ⟨bits,bitsVal⟩
      step as ⟨shift,shiftVal⟩
      step as ⟨shifted,shiftedVal,shiftedBits⟩
      simp only [lift,bind_ok]
      step as ⟨result,resultVal,resultBits⟩
      have narrowNat : access.width.val<8 := by scalar_tac
      have shiftNat : shift.val=64-access.width.val*8 := by scalar_tac
      have rawBits : value.bv=BitVec.ofNat 64 (MachineModel.loadFold (memoryModel memory)
          access.offset.val access.width.val 0 0) := by
        rw [←rawEq]
        simp only [UScalar.val,U64.bv,UScalarTy.numBits,BitVec.ofNat_toNat,BitVec.setWidth_eq]
      simp only [MachineModel.load,MachineModel.signedLoad,Bool.true_eq,true_and,narrowNat,ite_true]
      change (IScalar.hcast .U64 result).bv.toNat=_
      rw [IScalar.hcast_bv_eq,resultBits]
      simp only [I64.bv,UScalar.hcast_bv_eq,IScalarTy.numBits,BitVec.setWidth_eq]
      rw [shiftedBits,rawBits]
      simp only [UScalarTy.numBits,BitVec.signExtend_eq,BitVec.setWidth_eq,shiftNat]
    · have notNarrow : ¬access.width.val<8 := by scalar_tac
      simpa only [narrow,ite_false,WP.spec_ok,MachineModel.load,MachineModel.signedLoad,
        Bool.true_eq,true_and,notNarrow] using rawEq

end MemorySafetyExecution.Native.Independent
