/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.ActionAccess
import ValueFlow.SourceEvaluate

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write
set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false
namespace MemorySafetyExecution.Native

/-- Selected source instruction, independent evaluation, and its complete memory event. -/
def ExecutionInstructionAccessComplete (program : CoreProgram) (before : CoreState)
    (actor : Usize) (event : memory_safety.Event) : Prop :=
  ∃ thread instruction commit,
    before.threads.val[actor.val]?=some thread ∧ thread.running=true ∧
    program.code.val[thread.pc.val]?=some instruction ∧
    execution_core.prepare program before actor=.ok (.Ok commit) ∧
    commit.actor=actor ∧ commit.next_pc.val=thread.pc.val+1 ∧
    ValueFlow.evaluate (ValueFlow.Source.bankModel thread.registers)
      (ValueFlow.Source.idsModel thread.ids) (ValueFlow.Source.guardedModel instruction)=
      some (ValueFlow.Source.actionModel commit.action) ∧
    ActionAccess program before actor commit.action event

theorem source_execute_instruction_access_complete (program : CoreProgram)
    (before after : CoreState) (actor : Usize) (event : memory_safety.Event)
    (executed : execution_core.execute program before actor=.ok (.Ok event,after)) :
    ExecutionInstructionAccessComplete program before actor event := by
  obtain ⟨commit,prepared,committed⟩ := source_execute_prepared program before after actor event executed
  obtain ⟨thread,instruction,selected,running,fetched,commitActor,nextPc,evaluated⟩ :=
    source_prepare_evaluation program before actor commit prepared
  have independent := ValueFlow.Source.source_evaluate thread.registers thread.ids instruction
  simp only [evaluated,WP.spec_ok] at independent
  refine ⟨thread,instruction,commit,selected,running,fetched,prepared,commitActor,nextPc,?_,
    source_prepared_action_access program before after actor commit event prepared committed⟩
  simpa only [ValueFlow.Source.outcomeModel] using independent.symm

end MemorySafetyExecution.Native
