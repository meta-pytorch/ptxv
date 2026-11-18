/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyIR.Projection
import MemorySafetyIR.Instruction

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false

namespace MemorySafetyIR

theorem source_project_code_exact
    (code : Slice ptxv_ir.Inst) (nregs : U16)
    (projected : alloc.vec.Vec CheckedInstruction)
    (accepted : strict_memory_admission.project_code code nregs = .ok (.Ok projected)) :
    List.Forall₂ ExactInstruction code.val projected.val := by
  have related := source_project_code_preserves_every_instruction code nregs projected accepted
  apply List.Forall₂.imp (fun source target admitted => ?_) related
  rcases admitted with ⟨pc, last, computed⟩
  exact source_projected_instruction_is_exact source nregs pc last target computed

end MemorySafetyIR
