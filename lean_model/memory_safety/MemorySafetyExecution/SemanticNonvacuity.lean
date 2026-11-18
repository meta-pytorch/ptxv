/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.SourceMemorySafety

open Aeneas Aeneas.Std PtxvMemorySafety
set_option autoImplicit false
set_option debug.skipKernelTC false
namespace SemanticAudit

open MemorySafetyExecution

/-- Progress also excludes running actors whose PC has no instruction. -/
theorem running_actor_has_instruction (program : MachineModel.Program)
    (initial state : MachineModel.State) (actor : Nat) (thread : MachineModel.Thread)
    (safe : MachineModel.MemorySafeFrom program initial)
    (reachable : MachineModel.Reachable program initial state)
    (selected : state.threads[actor]?=some thread) (running : thread.running=true) :
    ∃ instruction,program.code[thread.pc]?=some instruction := by
  obtain ⟨event,after,computed⟩:=(safe.1 state reachable).1 actor thread selected running
  unfold MachineModel.step at computed
  simp only [selected] at computed
  cases fetched : program.code[thread.pc]? with
  | none => simp [fetched] at computed
  | some instruction => exact ⟨instruction,rfl⟩

theorem invalid_pc_actor_not_running (program : MachineModel.Program)
    (initial state : MachineModel.State) (actor : Nat) (thread : MachineModel.Thread)
    (safe : MachineModel.MemorySafeFrom program initial)
    (reachable : MachineModel.Reachable program initial state)
    (selected : state.threads[actor]?=some thread) (missing : program.code[thread.pc]?=none) :
    thread.running=false := by
  cases running:thread.running with
  | false => rfl
  | true =>
    obtain ⟨instruction,fetched⟩:=running_actor_has_instruction program initial state actor thread safe reachable selected running
    rw [missing] at fetched
    cases fetched

theorem source_program_has_complete_positive_actor_inventory
    (source entry : List U8) (manifest : ptxv_case.Manifest)
    (program : Native.CoreProgram)
    (meaning : PtxFrontendProgram.SourceProgram source entry manifest program) :
    program.initial.threads.val.length=
      (PtxFrontendProgram.dimensionsModel manifest.grid).size*
        (PtxFrontendProgram.dimensionsModel manifest.block).size ∧
      0<program.initial.threads.val.length ∧
      ∀ (index : Nat) (thread : ptxv_memory_write.execution_core.ThreadState),program.initial.threads.val[index]?=some thread → thread.running=true := by
  obtain ⟨parsed,parsedMeaning,construction⟩:=meaning
  rcases construction with ⟨entryMeaning,supported,inventory,code,layouts,allocationsLength,allocations,threads,outputs⟩
  dsimp only [PtxFrontendProgram.ThreadsMeaning] at threads
  obtain ⟨gridPositive,blockPositive,length,threadMeaning⟩:=threads
  have gridSize : 0<(PtxFrontendProgram.dimensionsModel manifest.grid).size :=
    Nat.mul_pos (Nat.mul_pos gridPositive.1 gridPositive.2.1) gridPositive.2.2
  have blockSize : 0<(PtxFrontendProgram.dimensionsModel manifest.block).size :=
    Nat.mul_pos (Nat.mul_pos blockPositive.1 blockPositive.2.1) blockPositive.2.2
  refine ⟨length,by rw [length];exact Nat.mul_pos gridSize blockSize,?_⟩
  intro index thread selected
  exact (threadMeaning index thread selected).2.1

theorem source_safe_has_nonempty_entry (source entry : List U8) (manifest : ptxv_case.Manifest)
    (safe : SourceMemorySafe source entry manifest) :
    ∃ program,PtxFrontendProgram.SourceProgram source entry manifest program ∧
      0<program.initial.threads.val.length ∧ program.code.val≠[] := by
  obtain ⟨program,meaning,native,independent,instances⟩:=safe
  obtain ⟨length,positive,running⟩:=source_program_has_complete_positive_actor_inventory source entry manifest program meaning
  let thread:=program.initial.threads.val[0]
  have selected : program.initial.threads.val[0]?=some thread:=List.getElem?_eq_getElem positive
  have threadRunning:=running 0 thread selected
  have modeledSelected : (Native.Independent.stateModel program.initial).threads[0]?=
      some (Native.Independent.threadModel thread) := by
    simp only [Native.Independent.stateModel,List.getElem?_map,selected,Option.map_some]
  have fetched:=running_actor_has_instruction (Native.Independent.programModel program)
    (Native.Independent.stateModel program.initial) (Native.Independent.stateModel program.initial)
    0 (Native.Independent.threadModel thread) independent .initial modeledSelected threadRunning
  refine ⟨program,meaning,positive,?_⟩
  intro empty
  obtain ⟨instruction,fetched⟩:=fetched
  simp only [Native.Independent.programModel,empty,List.map_nil,List.getElem?_nil] at fetched
  cases fetched

#print axioms SemanticAudit.running_actor_has_instruction
#print axioms SemanticAudit.invalid_pc_actor_not_running
#print axioms SemanticAudit.source_program_has_complete_positive_actor_inventory
#print axioms SemanticAudit.source_safe_has_nonempty_entry
end SemanticAudit
