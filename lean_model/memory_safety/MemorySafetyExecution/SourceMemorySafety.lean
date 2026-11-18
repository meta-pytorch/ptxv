/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.IndependentInstantiation
import FrontendProgram.SourceConstruction

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution

/-- Raw PTX and its manifest determine the program; payloads and legal bases vary. -/
def SourceMemorySafe (source entry : List U8) (manifest : ptxv_case.Manifest) : Prop :=
  ∃ program,PtxFrontendProgram.SourceProgram source entry manifest program ∧
    Native.FullMachineMemorySafe program ∧
    MachineModel.MemorySafeFrom (Native.Independent.programModel program)
      (Native.Independent.stateModel program.initial) ∧
    ∀ alternate,Native.MemoryInstance program alternate →
      Native.FullMachineMemorySafe alternate ∧
      MachineModel.MemorySafeFrom (Native.Independent.programModel alternate)
        (Native.Independent.stateModel alternate.initial)

end MemorySafetyExecution
