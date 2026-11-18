/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.Instantiation

open Aeneas Aeneas.Std PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write
set_option autoImplicit false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

theorem ProgramPlacements.trans {original middle alternate : CoreProgram}
    (first : ProgramPlacements original middle) (second : ProgramPlacements middle alternate) :
    ProgramPlacements original alternate := by
  refine ⟨first.1.trans second.1,?_,second.2.2⟩
  intro index old fresh oldSelected freshSelected
  have insideOriginal:=(List.getElem?_eq_some_iff.mp oldSelected).1
  have insideMiddle:index<middle.layouts.val.length:=by rw [←first.1];exact insideOriginal
  let between:=middle.layouts.val[index]
  have betweenSelected : middle.layouts.val[index]?=some between:=List.getElem?_eq_getElem insideMiddle
  have firstFields:=first.2.1 index old between oldSelected betweenSelected
  have secondFields:=second.2.1 index between fresh betweenSelected freshSelected
  exact ⟨secondFields.1.trans firstFields.1,secondFields.2.1.trans firstFields.2.1,secondFields.2.2⟩

theorem InitialBytesVariant.trans {original middle alternate : CoreState}
    (first : InitialBytesVariant original middle) (second : InitialBytesVariant middle alternate) :
    InitialBytesVariant original alternate := by
  refine ⟨first.threads.trans second.threads,first.allocationsLength.trans second.allocationsLength,?_⟩
  intro index old fresh oldSelected freshSelected
  have insideOriginal:=(List.getElem?_eq_some_iff.mp oldSelected).1
  have insideMiddle:index<middle.allocations.val.length:=by rw [←first.allocationsLength];exact insideOriginal
  let between:=middle.allocations.val[index]
  have betweenSelected : middle.allocations.val[index]?=some between:=List.getElem?_eq_getElem insideMiddle
  have firstFields:=first.allocations index old between oldSelected betweenSelected
  have secondFields:=second.allocations index between fresh betweenSelected freshSelected
  exact ⟨firstFields.1.trans secondFields.1,firstFields.2.1.trans secondFields.2.1,
    firstFields.2.2.trans secondFields.2.2⟩

theorem MemoryInstance.trans {original middle alternate : CoreProgram}
    (first : MemoryInstance original middle) (second : MemoryInstance middle alternate) :
    MemoryInstance original alternate :=
  ⟨first.code.trans second.code,first.outputs.trans second.outputs,
    first.placements.trans second.placements,first.initial.trans second.initial⟩

end MemorySafetyExecution.Native
