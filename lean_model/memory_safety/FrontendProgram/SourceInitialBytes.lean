/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Construction

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace PtxFrontendProgram

@[step]
theorem source_copy_initial_bytes (original bytes : alloc.vec.Vec U8)
    (defined : alloc.vec.Vec Bool) (index : Usize)
    (room : original.val.length < Usize.max) (bound : index.val ≤ original.val.length)
    (bytesEq : bytes.val = original.val.take index.val)
    (definedEq : defined.val = List.replicate index.val true) :
    strict_memory_frontend_launch.initial_allocation_loop0 original bytes defined index
      ⦃ output => output.1.val = original.val ∧
        output.2.val = List.replicate original.val.length true ⦄ := by
  unfold strict_memory_frontend_launch.initial_allocation_loop0
  apply loop.spec_decr_nat
    (fun current => original.val.length - current.2.2.val)
    (fun current => current.2.2.val ≤ original.val.length ∧
      current.1.val = original.val.take current.2.2.val ∧
      current.2.1.val = List.replicate current.2.2.val true)
  · rintro ⟨bytes, defined, index⟩ ⟨bound, bytesEq, definedEq⟩
    unfold strict_memory_frontend_launch.initial_allocation_loop0.body
    by_cases inside : index < alloc.vec.Vec.len original
    · simp only [inside, ite_true]
      step as ⟨byte, byteEq⟩
      step as ⟨bytesNext, bytesNextEq⟩
      step as ⟨definedNext, definedNextEq⟩
      step as ⟨next, nextVal⟩
      refine ⟨?_, ?_, ?_, ?_⟩
      · scalar_tac
      · have insideList : index.val < original.val.length := by scalar_tac
        simpa only [nextVal, List.take_succ_eq_append_getElem insideList, bytesEq, ←byteEq] using bytesNextEq
      · simpa only [nextVal, List.replicate_add, List.replicate_one, definedEq] using definedNextEq
      · scalar_tac
    · simp only [inside, ite_false, WP.spec_ok]
      have atEnd : index.val = original.val.length := by scalar_tac
      exact ⟨by simpa [atEnd] using bytesEq, by simpa only [atEnd] using definedEq⟩
  · exact ⟨bound, bytesEq, definedEq⟩

@[step]
theorem source_zero_initial_bytes (original bytes : alloc.vec.Vec U8) (initialized : Bool)
    (defined : alloc.vec.Vec Bool) (index : Usize)
    (room : original.val.length < Usize.max) (bound : index.val ≤ original.val.length)
    (bytesEq : bytes.val = List.replicate index.val 0#u8)
    (definedEq : defined.val = List.replicate index.val initialized) :
    strict_memory_frontend_launch.initial_allocation_loop1 original initialized bytes defined index
      ⦃ output => output.1.val = List.replicate original.val.length 0#u8 ∧
        output.2.val = List.replicate original.val.length initialized ⦄ := by
  unfold strict_memory_frontend_launch.initial_allocation_loop1
  apply loop.spec_decr_nat
    (fun current => original.val.length - current.2.2.val)
    (fun current => current.2.2.val ≤ original.val.length ∧
      current.1.val = List.replicate current.2.2.val 0#u8 ∧
      current.2.1.val = List.replicate current.2.2.val initialized)
  · rintro ⟨bytes, defined, index⟩ ⟨bound, bytesEq, definedEq⟩
    unfold strict_memory_frontend_launch.initial_allocation_loop1.body
    by_cases inside : index < alloc.vec.Vec.len original
    · simp only [inside, ite_true]
      step as ⟨bytesNext, bytesNextEq⟩
      step as ⟨definedNext, definedNextEq⟩
      step as ⟨next, nextVal⟩
      refine ⟨?_, ?_, ?_, ?_⟩
      · scalar_tac
      · simpa only [nextVal, List.replicate_add, List.replicate_one, bytesEq] using bytesNextEq
      · simpa only [nextVal, List.replicate_add, List.replicate_one, definedEq] using definedNextEq
      · scalar_tac
    · simp only [inside, ite_false, WP.spec_ok]
      have atEnd : index.val = original.val.length := by scalar_tac
      exact ⟨by simpa only [atEnd] using bytesEq, by simpa only [atEnd] using definedEq⟩
  · exact ⟨bound, bytesEq, definedEq⟩

end PtxFrontendProgram
