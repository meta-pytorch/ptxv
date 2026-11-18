/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyIR.LaunchModel
import MemorySafetyIR.Initialization

open Aeneas Aeneas.Std PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyIR

def launchIRProgram (kernel : ptxv_ir.Kernel) (launch : machine.Launch) : IRProgram :=
  { allocations := launchAllocations launch
    code := kernel.code.val
    threads := match launch.grids.val[0]? with
      | none => 0
      | some grid => gridThreadCount grid
    globals := match launch.grids.val[0]? with
      | none => []
      | some grid => grid.cbank_relocs.val }

def launchOutputs (launch : machine.Launch) : List Nat :=
  launch.outputs.val.map UScalar.val

theorem validated_outputs_initially_defined (kernel : ptxv_ir.Kernel) (launch : machine.Launch)
    (validated : outputsDefined launch) :
    InitiallyDefinedOutputs (launchIRProgram kernel launch) (launchOutputs launch) := by
  intro index member
  obtain ⟨output, outputMember, rfl⟩ := List.mem_map.mp member
  obtain ⟨allocation, found, defined⟩ := validated output outputMember
  refine ⟨allocationMemoryModel allocation, ?_, ?_⟩
  · simp only [launchIRProgram, launchAllocations, List.getElem?_map, found, Option.map_some]
  · simp only [allocationMemoryModel, defined, Bool.not_false]

end MemorySafetyIR
