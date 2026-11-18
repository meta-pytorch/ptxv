/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyIR.Semantics
import MemorySafety.Soundness

open Aeneas Aeneas.Std PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyIR

def requestMemoryModel (request : Request) : MemorySafety.Request :=
  ⟨request.address.val, request.width.val, request.alignment.val,
    match request.kind with
    | .Load => .load
    | .Store defined => .store defined⟩

def scalarMemoryWidth (bytes : U8) : Prop :=
  bytes = 1#u8 ∨ bytes = 2#u8 ∨ bytes = 4#u8 ∨ bytes = 8#u8

instance (bytes : U8) : Decidable (scalarMemoryWidth bytes) := by
  unfold scalarMemoryWidth
  infer_instance

def interpretInstruction (valuation : Valuation) (instruction : ptxv_ir.Inst) : MemorySafety.Instruction :=
  match instruction.op with
  | .Ld _ count .Global bytes operand none _ false =>
    if count = 1#u8 ∧ scalarMemoryWidth bytes then
      if predicate valuation instruction.guard then
        .access (requestMemoryModel ⟨address valuation operand, width bytes, width bytes, .Load⟩)
      else .skip
    else .unsupported
  | .St .Global bytes operand (.Imm bits) none =>
    if scalarMemoryWidth bytes then
      if predicate valuation instruction.guard then
        .access (requestMemoryModel ⟨address valuation operand, width bytes, width bytes, .Store true⟩)
      else .skip
    else .unsupported
  | .Nop => .skip
  | .Exit => if predicate valuation instruction.guard then .halt else .skip
  | _ => .unsupported

abbrev GlobalBinding := U8 × U32 × U64

def globalAddress (bindings : List GlobalBinding) (index : U32) : Option U64 :=
  if index.val * 8 ≤ U32.max then
    (bindings.find? (fun binding => decide (binding.1.val = 31 ∧ binding.2.1.val = index.val * 8))).map
      (fun binding => binding.2.2)
  else none

def bindGlobals (bindings : List GlobalBinding) (valuation : Valuation) : Valuation :=
  { valuation with special := fun register =>
    match register with
    | .GlobalAddr index => (globalAddress bindings index).getD (valuation.special register)
    | _ => valuation.special register }

structure IRProgram where
  allocations : List MemorySafety.Allocation
  code : List ptxv_ir.Inst
  threads : Nat
  globals : List GlobalBinding

def memoryFrame (program : IRProgram) : MemorySafety.Program :=
  ⟨program.allocations, List.replicate program.threads []⟩

def atValuations (program : IRProgram) (valuations : Nat → Valuation) : MemorySafety.Program :=
  ⟨program.allocations,
    (List.range program.threads).map (fun thread => program.code.map (interpretInstruction (bindGlobals program.globals (valuations thread))))⟩

def projectedProgram (program : IRProgram) (code : List MemorySafety.Instruction) : MemorySafety.Program :=
  ⟨program.allocations, List.replicate program.threads code⟩

def IRStep (program : IRProgram) (before : MemorySafety.State) (event : MemorySafety.Event)
    (after : MemorySafety.State) : Prop :=
  ∃ valuations, MemorySafety.Step (atValuations program valuations) before event after

inductive IRReachable (program : IRProgram) : MemorySafety.State → Prop
  | initial : IRReachable program (MemorySafety.initialState (memoryFrame program))
  | step {before event after} : IRReachable program before → IRStep program before event after →
      IRReachable program after

def IRMemorySafe (program : IRProgram) : Prop :=
  MemorySafety.LayoutSafe (memoryFrame program) ∧
  (∀ state, IRReachable program state → state.bad = false) ∧
  ∀ before event after, IRReachable program before → IRStep program before event after →
    MemorySafety.EventSafe (memoryFrame program) before event

theorem projected_code_equal (source : List ptxv_ir.Inst) (code : List MemorySafety.Instruction)
    (globals : List GlobalBinding)
    (projected : List.Forall₂ (fun instruction target =>
      ∀ valuation, interpretInstruction (bindGlobals globals valuation) instruction = target) source code)
    (valuation : Valuation) : source.map (interpretInstruction (bindGlobals globals valuation)) = code := by
  induction projected with
  | nil => rfl
  | cons first rest ih => simp only [List.map_cons, first valuation, ih]

theorem at_valuations_equal (program : IRProgram) (code : List MemorySafety.Instruction)
    (projected : List.Forall₂ (fun instruction target =>
      ∀ valuation, interpretInstruction (bindGlobals program.globals valuation) instruction = target) program.code code)
    (valuations : Nat → Valuation) : atValuations program valuations = projectedProgram program code := by
  unfold atValuations projectedProgram
  simp_rw [projected_code_equal program.code code program.globals projected]
  simp

theorem projected_initial_equal (program : IRProgram) (code : List MemorySafety.Instruction) :
    MemorySafety.initialState (memoryFrame program) =
      MemorySafety.initialState (projectedProgram program code) := by
  simp [MemorySafety.initialState, memoryFrame, projectedProgram]

theorem ir_reachable_projected (program : IRProgram) (code : List MemorySafety.Instruction)
    (projected : List.Forall₂ (fun instruction target =>
      ∀ valuation, interpretInstruction (bindGlobals program.globals valuation) instruction = target) program.code code)
    (state : MemorySafety.State) (reachable : IRReachable program state) :
    MemorySafety.Reachable (projectedProgram program code) state := by
  induction reachable with
  | initial =>
    rw [projected_initial_equal]
    exact .initial
  | @step before event after reachable transition ih =>
    obtain ⟨valuations, transition⟩ := transition
    rw [at_valuations_equal program code projected valuations] at transition
    exact .step ih transition

theorem projected_memory_safe (program : IRProgram) (code : List MemorySafety.Instruction)
    (projected : List.Forall₂ (fun instruction target =>
      ∀ valuation, interpretInstruction (bindGlobals program.globals valuation) instruction = target) program.code code)
    (safe : MemorySafety.MemorySafe (projectedProgram program code)) : IRMemorySafe program := by
  refine ⟨safe.1, ?_, ?_⟩
  · intro state reachable
    exact safe.2.1 state (ir_reachable_projected program code projected state reachable)
  · intro before event after reachable transition
    obtain ⟨valuations, transition⟩ := transition
    rw [at_valuations_equal program code projected valuations] at transition
    exact safe.2.2 before event after
      (ir_reachable_projected program code projected before reachable) transition

end MemorySafetyIR

namespace MemorySafetyIR

theorem ir_memory_safe_range (program : IRProgram) (safe : IRMemorySafe program)
    (before after : MemorySafety.State) (thread : Nat) (request : MemorySafety.Request)
    (reachable : IRReachable program before)
    (transition : IRStep program before (.access thread request) after) :
    0 < request.width ∧ 0 < request.alignment ∧ request.address % request.alignment = 0 ∧
      request.address + request.width ≤ MemorySafety.addressLimit ∧
      MemorySafety.LiveSpan (memoryFrame program) before request := by
  have components := MemorySafety.access_safe_components (memoryFrame program) before thread request
    (safe.2.2 before (.access thread request) after reachable transition)
  exact ⟨components.1, components.2.1, components.2.2.1,
    components.2.2.2.1, components.2.2.2.2.1⟩

theorem ir_memory_safe_read (program : IRProgram) (safe : IRMemorySafe program)
    (before after : MemorySafety.State) (thread : Nat) (request : MemorySafety.Request)
    (reachable : IRReachable program before)
    (transition : IRStep program before (.access thread request) after)
    (load : request.kind = .load) (offset : Nat) (inside : offset < request.width) :
    MemorySafety.byteDefined (memoryFrame program) before (request.address + offset) = true :=
  MemorySafety.safe_read_every_byte (memoryFrame program) before thread request
    (safe.2.2 before (.access thread request) after reachable transition) load offset inside

theorem ir_memory_safe_races (program : IRProgram) (safe : IRMemorySafe program)
    (before after : MemorySafety.State) (thread : Nat) (request : MemorySafety.Request)
    (reachable : IRReachable program before)
    (transition : IRStep program before (.access thread request) after)
    (earlier : MemorySafety.AccessRecord) (recorded : earlier ∈ before.history) :
    ¬ MemorySafety.Conflict earlier (MemorySafety.accessRecord thread request) :=
  MemorySafety.safe_access_conflicts_with_no_previous_access (memoryFrame program) before thread request
    (safe.2.2 before (.access thread request) after reachable transition) earlier recorded

end MemorySafetyIR
