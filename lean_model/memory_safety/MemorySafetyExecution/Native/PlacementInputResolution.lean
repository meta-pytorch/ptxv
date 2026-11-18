/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.PlacementResolution
import MemorySafetyExecution.Native.InputState

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
set_option autoImplicit false
namespace MemorySafetyExecution.Native
open ValueFlow.Source

theorem input_storage_shapes (shadow : value_flow_dependencies.State) (left right : CoreState)
    (related : StateInputsRelated shadow left right) : StorageShapesRelated left right := by
  refine ⟨related.allocationsLength, ?_⟩
  intro index old fresh oldAt freshAt
  have inside := (List.getElem?_eq_some_iff.mp oldAt).1
  have metadataLength := related.allocationMetadataLength
  have metadataInside : index < shadow.memory.val.length := by omega
  have memory := related.allocations index old fresh shadow.memory.val[index] oldAt freshAt
    (List.getElem?_eq_getElem metadataInside)
  exact ⟨memory.bytesLength, memory.definitions, memory.live⟩

end MemorySafetyExecution.Native
