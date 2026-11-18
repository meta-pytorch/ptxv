/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.SourceModel

open Aeneas Aeneas.Std PtxvMemorySafety

set_option autoImplicit false

namespace MemorySafetyIR

def allocationMemoryModel (allocation : machine.Alloc) : MemorySafety.Allocation :=
  ⟨allocation.base.val, allocation.init.val.length, !allocation.undef⟩

def launchAllocations (launch : machine.Launch) : List MemorySafety.Allocation :=
  launch.allocs.val.map allocationMemoryModel

def dimensionsProduct (dimensions : Array U32 3#usize) : Nat :=
  (dimensions.val.map (fun dimension => dimension.val)).prod

def gridThreadCount (grid : machine.Grid) : Nat :=
  dimensionsProduct grid.grid * dimensionsProduct grid.block

def outputsDefined (launch : machine.Launch) : Prop :=
  ∀ output ∈ launch.outputs.val, ∃ allocation,
    launch.allocs.val[output.val]? = some allocation ∧ allocation.undef = false

end MemorySafetyIR
