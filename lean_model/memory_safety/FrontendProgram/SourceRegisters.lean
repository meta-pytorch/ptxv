/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Construction
import MemoryWrite.TotalPrimitives

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace PtxFrontendProgram

@[step]
theorem source_bitmap_words (count : U16) :
    strict_memory_frontend_launch.bitmap_words count
      ⦃ words => words.val = (count.val + 63) / 64 ⦄ := by
  unfold strict_memory_frontend_launch.bitmap_words
  repeat' first | step | split
  all_goals scalar_tac

@[step]
private theorem registers_loop (virtuals : U16) (regs : alloc.vec.Vec U64)
    (pointers : alloc.vec.Vec (Option ptxv_memory_write.value_flow.Pointer))
    (index : Usize) (bound : index.val ≤ virtuals.val)
    (initialRegs : regs.val = List.replicate index.val 0#u64)
    (initialPointers : pointers.val = List.replicate index.val none) :
    strict_memory_frontend_launch.empty_registers_loop0 virtuals regs pointers index
      ⦃ output => output.1.val = List.replicate virtuals.val 0#u64 ∧
        output.2.val = List.replicate virtuals.val none ⦄ := by
  unfold strict_memory_frontend_launch.empty_registers_loop0
  apply loop.spec_decr_nat
    (fun current => virtuals.val - current.2.2.val)
    (fun current => current.2.2.val ≤ virtuals.val ∧
      current.1.val = List.replicate current.2.2.val 0#u64 ∧
      current.2.1.val = List.replicate current.2.2.val none)
  · rintro ⟨regs, pointers, index⟩ ⟨bound, regsEq, pointersEq⟩
    unfold strict_memory_frontend_launch.empty_registers_loop0.body
    simp only [lift, bind_ok]
    by_cases inside : index < UScalar.cast .Usize virtuals
    · simp only [inside, ite_true]
      step as ⟨regsNext, regsNextEq⟩
      step as ⟨pointersNext, pointersNextEq⟩
      step as ⟨next, nextVal⟩
      refine ⟨?_, ?_, ?_, ?_⟩
      · scalar_tac
      · simpa only [nextVal, regsEq, List.replicate_add, List.replicate_one] using regsNextEq
      · simpa only [nextVal, pointersEq, List.replicate_add, List.replicate_one] using pointersNextEq
      · scalar_tac
    · simp only [inside, ite_false, WP.spec_ok]
      have equal : index.val = virtuals.val := by scalar_tac
      exact ⟨by simpa only [equal] using regsEq, by simpa only [equal] using pointersEq⟩
  · exact ⟨bound, initialRegs, initialPointers⟩

@[step]
private theorem written_loop (virtuals : U16) (index : Usize) (written : alloc.vec.Vec U64)
    (bound : index.val ≤ (virtuals.val + 63) / 64)
    (initial : written.val = List.replicate index.val 0#u64) :
    strict_memory_frontend_launch.empty_registers_loop1 virtuals index written
      ⦃ output => output.val = List.replicate ((virtuals.val + 63) / 64) 0#u64 ⦄ := by
  unfold strict_memory_frontend_launch.empty_registers_loop1
  apply loop.spec_decr_nat
    (fun current => (virtuals.val + 63) / 64 - current.1.val)
    (fun current => current.1.val ≤ (virtuals.val + 63) / 64 ∧
      current.2.val = List.replicate current.1.val 0#u64)
  · rintro ⟨index, written⟩ ⟨bound, writtenEq⟩
    unfold strict_memory_frontend_launch.empty_registers_loop1.body
    step as ⟨words, wordsVal⟩
    by_cases inside : index < words
    · simp only [inside, ite_true]
      step as ⟨writtenNext, writtenNextEq⟩
      step as ⟨next, nextVal⟩
      refine ⟨?_, ?_, ?_⟩
      · scalar_tac
      · simpa only [nextVal, writtenEq, List.replicate_add, List.replicate_one] using writtenNextEq
      · scalar_tac
    · simp only [inside, ite_false, WP.spec_ok]
      have equal : index.val = (virtuals.val + 63) / 64 := by scalar_tac
      simpa only [equal] using writtenEq
  · exact ⟨bound, initial⟩

@[step]
private theorem predicates_loop (predicates : U16) (index : Usize)
    (preds pdef : alloc.vec.Vec U64)
    (bound : index.val ≤ (predicates.val + 63) / 64)
    (initialPreds : preds.val = List.replicate index.val 0#u64)
    (initialPdef : pdef.val = List.replicate index.val 0#u64) :
    strict_memory_frontend_launch.empty_registers_loop2 predicates index preds pdef
      ⦃ output => output.1.val = List.replicate ((predicates.val + 63) / 64) 0#u64 ∧
        output.2.val = List.replicate ((predicates.val + 63) / 64) 0#u64 ⦄ := by
  unfold strict_memory_frontend_launch.empty_registers_loop2
  apply loop.spec_decr_nat
    (fun current => (predicates.val + 63) / 64 - current.1.val)
    (fun current => current.1.val ≤ (predicates.val + 63) / 64 ∧
      current.2.1.val = List.replicate current.1.val 0#u64 ∧
      current.2.2.val = List.replicate current.1.val 0#u64)
  · rintro ⟨index, preds, pdef⟩ ⟨bound, predsEq, pdefEq⟩
    unfold strict_memory_frontend_launch.empty_registers_loop2.body
    step as ⟨words, wordsVal⟩
    by_cases inside : index < words
    · simp only [inside, ite_true]
      step as ⟨predsNext, predsNextEq⟩
      step as ⟨pdefNext, pdefNextEq⟩
      step as ⟨next, nextVal⟩
      refine ⟨?_, ?_, ?_, ?_⟩
      · scalar_tac
      · simpa only [nextVal, predsEq, List.replicate_add, List.replicate_one] using predsNextEq
      · simpa only [nextVal, pdefEq, List.replicate_add, List.replicate_one] using pdefNextEq
      · scalar_tac
    · simp only [inside, ite_false, WP.spec_ok]
      have equal : index.val = (predicates.val + 63) / 64 := by scalar_tac
      exact ⟨by simpa only [equal] using predsEq, by simpa only [equal] using pdefEq⟩
  · exact ⟨bound, initialPreds, initialPdef⟩

@[step]
theorem source_empty_registers (virtuals predicates : U16) :
    strict_memory_frontend_launch.empty_registers virtuals predicates
      ⦃ output => InitialRegisters virtuals.val predicates.val output ⦄ := by
  unfold strict_memory_frontend_launch.empty_registers
  step with registers_loop virtuals (alloc.vec.Vec.new U64)
    (alloc.vec.Vec.new (Option ptxv_memory_write.value_flow.Pointer)) 0#usize
    (by simp) (by simp) (by simp) as ⟨regs, pointers, regsEq, pointersEq⟩
  step with written_loop virtuals 0#usize (alloc.vec.Vec.new U64)
    (by simp) (by simp) as ⟨written, writtenEq⟩
  step with predicates_loop predicates 0#usize (alloc.vec.Vec.new U64)
    (alloc.vec.Vec.new U64) (by simp) (by simp) (by simp) as ⟨preds, pdef, predsEq, pdefEq⟩
  simp only [InitialRegisters]
  exact ⟨regsEq, pointersEq, writtenEq, predsEq, pdefEq, by simp⟩

end PtxFrontendProgram
