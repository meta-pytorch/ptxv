/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.IndependentModel

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false
namespace MemorySafetyExecution.Native.Independent
open ValueFlow.Source

theorem source_render_model (program : CoreProgram) (value : V.Value) (bits : U64)
    (computed : value_flow.render_value value (alloc.vec.Vec.deref program.layouts)=.ok (.Ok bits)) :
    MachineModel.render (programModel program) (valueModel value)=
      some ⟨bits.val,0,(valuePointer value).map pointerModel⟩ := by
  cases value with
  | Scalar scalar =>
    have identity : scalar=bits := by simpa only [value_flow.render_value,ok.injEq,core.result.Result.Ok.injEq] using computed
    subst bits
    rfl
  | Pointer pointer =>
    by_cases outside : pointer.allocation≥Slice.len (alloc.vec.Vec.deref program.layouts)
    · simp only [value_flow.render_value,outside,ite_true,ok.injEq,reduceCtorEq] at computed
    · have insideSlice : pointer.allocation.val<(alloc.vec.Vec.deref program.layouts).val.length := by scalar_tac
      have inside : pointer.allocation.val<program.layouts.val.length := by simpa only [alloc.vec.Vec.deref,Slice.from_val] using insideSlice
      have identity : core.num.U64.wrapping_add program.layouts.val[pointer.allocation.val].base pointer.offset=bits := by
        simpa [value_flow.render_value,outside,Slice.index_usize,alloc.vec.Vec.deref,Slice.from_val,List.getElem?_eq_getElem inside,Nat.not_le_of_gt inside,lift,bind_ok] using computed
      rw [←identity]
      simp [MachineModel.render,programModel,valueModel,pointerModel,valuePointer,
        List.getElem?_eq_getElem inside,layoutModel,core.num.U64.wrapping_add_val_eq,U64.size,U64.numBits]

theorem source_virtual_bank_model (program : CoreProgram) (before after : V.Registers)
    (index : U16) (value : V.Value)
    (canonical : ValueFlow.Storage.MachineRegistersVirtualMap.Canonical before.raw.vund.val)
    (written : value_flow.write_value before index value (alloc.vec.Vec.deref program.layouts)=.ok (.Ok (),after)) :
    ∃word, MachineModel.render (programModel program) (valueModel value)=some word ∧
      bankModel after=MachineModel.writeVirtual (bankModel before) index.val word ∧
      ValueFlow.Storage.MachineRegistersVirtualMap.Canonical after.raw.vund.val := by
  obtain ⟨bits,rendered,selected,others,predicates,proper⟩ :=
    source_write_value_success before after index value (alloc.vec.Vec.deref program.layouts) canonical written
  refine ⟨⟨bits.val,0,(valuePointer value).map pointerModel⟩,
    source_render_model program value bits rendered,?_,proper⟩
  apply congrArg₂ ValueFlow.RegisterBank.mk
  · funext query
    by_cases same : query=index.val
    · subst query
      simpa only [MachineModel.writeVirtual,bankModel,ite_true] using selected
    · simpa only [MachineModel.writeVirtual,bankModel,same,ite_false] using others query same
  · exact funext predicates

theorem source_predicate_bank_model (before after : V.Registers) (index : U16) (value : Bool)
    (written : value_flow.write_predicate before index value=.ok (.Ok (),after)) :
    bankModel after=MachineModel.writePredicate (bankModel before) index.val value := by
  obtain ⟨predicates,words⟩ := source_write_predicate_success before after index value written
  exact congrArg₂ ValueFlow.RegisterBank.mk (funext words) (funext predicates)

end MemorySafetyExecution.Native.Independent
