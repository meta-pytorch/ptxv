/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.MachineModelSafety
import MemorySafetyExecution.SourceSoundness

set_option autoImplicit false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.MachineModel

/-- Independent obligations for every byte of an instruction's memory span. -/
def SpanSafe (program : Program) (state : State) (pointer : ValueFlow.Pointer)
    (width : Nat) (write : Bool) : Prop :=
  ∃ layout memory,
    program.layouts[pointer.allocation]?=some layout ∧
    state.allocations[pointer.allocation]?=some memory ∧
    ValueFlow.supportedWidth width ∧
    layout.alignment≠0 ∧ layout.alignment%width=0 ∧
    pointer.offset%width=0 ∧ layout.base%width=0 ∧
    pointer.offset+width≤layout.bytes ∧
    layout.base+pointer.offset+width≤2^64 ∧
    memory.live=true ∧ pointer.offset+width≤memory.bytes.length ∧
    memory.defined.length=memory.bytes.length ∧
    (write=true ∨ ∀ index∈List.range width,
      memory.defined[pointer.offset+index]?=some true)

def ActionSafe (program : Program) (state : State) : ValueFlow.Action → Prop
  | .load _ _ width _ pointer => SpanSafe program state pointer width false
  | .store width pointer _ => SpanSafe program state pointer width true
  | _ => True

theorem resolve_span_safe (program : Program) (state : State) (pointer : ValueFlow.Pointer)
    (width : Nat) (write : Bool) (request : MemorySafety.Request) (memory : Memory)
    (computed : resolve program state pointer width write=some (request,memory)) :
    SpanSafe program state pointer width write := by
  unfold resolve at computed
  cases layoutAt : program.layouts[pointer.allocation]? with
  | none => simp [layoutAt] at computed
  | some layout =>
    cases memoryAt : state.allocations[pointer.allocation]? with
    | none => simp [layoutAt,memoryAt] at computed
    | some actual =>
      simp [layoutAt,memoryAt] at computed
      obtain ⟨supported,positive,alignment,offsetAlignment,baseAlignment,bounds,
        noWrap,live,storage,flags,initialized,_⟩:=computed
      exact ⟨layout,actual,layoutAt,memoryAt,supported,positive,alignment,
        offsetAlignment,baseAlignment,bounds,noWrap,live,storage,flags,
        by simpa only [List.mem_range] using initialized⟩

theorem apply_action_safe (program : Program) (state : State) (actor : Nat)
    (thread : Thread) (action : ValueFlow.Action) (event : MemorySafety.Event) (after : State)
    (computed : applyAction program state actor thread action=some (event,after)) :
    ActionSafe program state action := by
  cases action with
  | load destination bits width signed pointer =>
    unfold applyAction at computed
    cases resolved : resolve program state pointer width false with
    | none => simp [resolved] at computed
    | some pair => exact resolve_span_safe program state pointer width false pair.1 pair.2 resolved
  | store width pointer value =>
    unfold applyAction at computed
    cases resolved : resolve program state pointer width true with
    | none => simp [resolved] at computed
    | some pair => exact resolve_span_safe program state pointer width true pair.1 pair.2 resolved
  | _ => trivial

theorem step_instruction_safe (program : Program) (state : State) (actor : Nat)
    (thread : Thread) (instruction : ValueFlow.GuardedInstruction) (action : ValueFlow.Action)
    (event : MemorySafety.Event) (after : State)
    (selected : state.threads[actor]?=some thread) (running : thread.running=true)
    (fetched : program.code[thread.pc]?=some instruction)
    (evaluated : ValueFlow.evaluate thread.registers thread.ids instruction=some action)
    (computed : step program state actor=some (event,after)) :
    ActionSafe program state action := by
  have applied : applyAction program state actor thread action=some (event,after) := by
    simpa [step,selected,running,fetched,evaluated] using computed
  exact apply_action_safe program state actor thread action event after applied

/-- Safety rules out an unsafe memory instruction at every reachable state. -/
theorem memory_safe_instruction_spans (program : Program) (initial state : State)
    (safe : MemorySafeFrom program initial) (reachable : Reachable program initial state)
    (actor : Nat) (thread : Thread) (instruction : ValueFlow.GuardedInstruction)
    (action : ValueFlow.Action)
    (selected : state.threads[actor]?=some thread) (running : thread.running=true)
    (fetched : program.code[thread.pc]?=some instruction)
    (evaluated : ValueFlow.evaluate thread.registers thread.ids instruction=some action) :
    ActionSafe program state action := by
  obtain ⟨event,after,computed⟩:=(safe.1 state reachable).1 actor thread selected running
  exact step_instruction_safe program state actor thread instruction action event after
    selected running fetched evaluated computed

/-- Every reachable running actor fetches and evaluates an instruction with a safe span. -/
theorem memory_safe_running_instruction (program : Program) (initial state : State)
    (safe : MemorySafeFrom program initial) (reachable : Reachable program initial state)
    (actor : Nat) (thread : Thread)
    (selected : state.threads[actor]?=some thread) (running : thread.running=true) :
    ∃ instruction action,program.code[thread.pc]?=some instruction ∧
      ValueFlow.evaluate thread.registers thread.ids instruction=some action ∧
      ActionSafe program state action := by
  obtain ⟨event,after,computed⟩:=(safe.1 state reachable).1 actor thread selected running
  cases fetched : program.code[thread.pc]? with
  | none => simp [step,selected,running,fetched] at computed
  | some instruction =>
    cases evaluated : ValueFlow.evaluate thread.registers thread.ids instruction with
    | none => simp [step,selected,running,fetched,evaluated] at computed
    | some action =>
      exact ⟨instruction,action,rfl,evaluated,
        step_instruction_safe program state actor thread instruction action event after
          selected running fetched evaluated computed⟩

end MemorySafetyExecution.MachineModel

namespace MemorySafetyExecution.Source

open Aeneas Aeneas.Std Result PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

theorem source_check_is_safe_instruction_spans {F : Type _}
    (producerInterface : core.ops.function.FnOnce F
      (execution_core.Program × Usize) strict_memory_source.Candidate)
    (source entry : Slice U8) (manifest : ptxv_case.Manifest) (cap : Usize) (producer : F)
    (report : strict_memory_source.Report)
    (computed : strict_memory_source.check producerInterface source entry manifest cap producer=.ok report)
    (accepted : strict_memory_source.Report.is_safe report=.ok true) :
    ∃ program,PtxFrontendProgram.SourceProgram source.val entry.val manifest program ∧
      ∀ alternate,Native.MemoryInstance program alternate →
      ∀ state,MachineModel.Reachable (Native.Independent.programModel alternate)
        (Native.Independent.stateModel alternate.initial) state →
      ∀ (actor : Nat) (thread : MachineModel.Thread),
        state.threads[actor]?=some thread → thread.running=true →
        ∃ instruction action,
          (Native.Independent.programModel alternate).code[thread.pc]?=some instruction ∧
          ValueFlow.evaluate thread.registers thread.ids instruction=some action ∧
          MachineModel.ActionSafe (Native.Independent.programModel alternate) state action := by
  obtain ⟨program,sourceMeaning,_,_,allInstances⟩:=source_check_is_safe_full_memory_safety
    producerInterface source entry manifest cap producer report computed accepted
  refine ⟨program,sourceMeaning,?_⟩
  intro alternate instantiation state reachable actor thread selected running
  exact MachineModel.memory_safe_running_instruction _ _ state (allInstances alternate instantiation).2
    reachable actor thread selected running

end MemorySafetyExecution.Source
