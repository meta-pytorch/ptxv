/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.IndependentTransport
import MemorySafetyExecution.Native.IndependentStep
import MemorySafetyExecution.Native.Instantiation

open Aeneas Aeneas.Std Result PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write
set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option debug.skipKernelTC false
set_option Elab.async false

namespace MemorySafetyExecution.Native.Independent

theorem source_input_verify_all_independent_instances (source entry : List U8)
    (manifest : PtxFrontendProgram.Manifest) (parsed : strict_memory_frontend_parse.SourceProgram)
    (program : CoreProgram)
    (meaning : PtxFrontendProgram.ConstructionMeaning source entry manifest parsed program)
    (certificate : Slice input_independent_execution.State)
    (accepted : input_independent_execution.verify program certificate=.ok true) :
    ∀ alternate,MemoryInstance program alternate →
      MachineModel.MemorySafeFrom (programModel alternate) (stateModel alternate.initial) := by
  intro alternate instantiation
  obtain ⟨canonical,_,safe⟩:=source_input_verify_core_instances source entry manifest parsed program
    meaning certificate accepted alternate instantiation
  exact core_safety_transports alternate alternate.initial safe canonical (source_execute_independent alternate)

end MemorySafetyExecution.Native.Independent
