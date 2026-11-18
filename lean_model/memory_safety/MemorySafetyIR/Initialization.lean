/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyIR.GlobalSpecialization
import MemorySafety.Initialization

open Aeneas Aeneas.Std PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyIR

theorem interpreted_preserves_definedness (valuation : Valuation) (instruction : ptxv_ir.Inst) :
    MemorySafety.PreservesDefinedness (interpretInstruction valuation instruction) := by
  unfold interpretInstruction
  split
  all_goals try split_ifs
  all_goals simp [MemorySafety.PreservesDefinedness, requestMemoryModel]

theorem projected_code_preserves_definedness (source : List ptxv_ir.Inst)
    (code : List MemorySafety.Instruction) (globals : List GlobalBinding)
    (projected : List.Forall₂ (fun instruction target =>
      ∀ valuation, interpretInstruction (bindGlobals globals valuation) instruction = target) source code)
    : ∀ instruction ∈ code, MemorySafety.PreservesDefinedness instruction := by
  let valuation : Valuation := ⟨fun _ _ => 0#u64, fun _ _ => 0#u64, fun _ => 0#u64,
    fun _ => false, fun _ _ => true, fun _ _ => true, fun _ => true⟩
  rw [← projected_code_equal source code globals projected valuation]
  intro instruction member
  obtain ⟨original, _, rfl⟩ := List.mem_map.mp member
  exact interpreted_preserves_definedness _ original

theorem ir_reachable_definedness (program : IRProgram) (code : List MemorySafety.Instruction)
    (projected : List.Forall₂ (fun instruction target =>
      ∀ valuation, interpretInstruction (bindGlobals program.globals valuation) instruction = target) program.code code)
    (safe : MemorySafety.MemorySafe (projectedProgram program code))
    (state : MemorySafety.State) (reachable : IRReachable program state) :
    MemorySafety.DefinedWrites state ∧ state.live = List.replicate program.allocations.length true := by
  apply MemorySafety.reachable_definedness (projectedProgram program code) safe
  · intro thread member instruction present
    have threadEq : thread = code := List.eq_of_mem_replicate member
    subst thread
    exact projected_code_preserves_definedness _ _ _ projected instruction present
  · exact ir_reachable_projected program code projected state reachable

theorem ir_initially_defined_byte_remains_defined (program : IRProgram)
    (code : List MemorySafety.Instruction)
    (projected : List.Forall₂ (fun instruction target =>
      ∀ valuation, interpretInstruction (bindGlobals program.globals valuation) instruction = target) program.code code)
    (safe : MemorySafety.MemorySafe (projectedProgram program code))
    (state : MemorySafety.State) (reachable : IRReachable program state)
    (address : Nat)
    (initial : MemorySafety.byteDefined (memoryFrame program)
      (MemorySafety.initialState (memoryFrame program)) address = true) :
    MemorySafety.byteDefined (memoryFrame program) state address = true := by
  exact MemorySafety.initially_defined_byte_remains_defined (memoryFrame program) state address
    (ir_reachable_definedness program code projected safe state reachable).1 initial

theorem ir_initially_defined_allocation_survives (program : IRProgram)
    (code : List MemorySafety.Instruction)
    (projected : List.Forall₂ (fun instruction target =>
      ∀ valuation, interpretInstruction (bindGlobals program.globals valuation) instruction = target) program.code code)
    (safe : MemorySafety.MemorySafe (projectedProgram program code))
    (state : MemorySafety.State) (reachable : IRReachable program state)
    (index : Nat) (allocation : MemorySafety.Allocation)
    (found : program.allocations[index]? = some allocation)
    (defined : allocation.initiallyDefined = true) (offset : Nat) (inside : offset < allocation.bytes) :
    state.live[index]? = some true ∧
      MemorySafety.byteDefined (memoryFrame program) state (allocation.base + offset) = true := by
  have invariant := ir_reachable_definedness program code projected safe state reachable
  have indexInside := (List.getElem?_eq_some_iff.mp found).1
  constructor
  · rw [invariant.2]
    simp [indexInside]
  · have initial := MemorySafety.allocation_initially_defined_byte (memoryFrame program) safe.1
      allocation (List.mem_of_getElem? found) defined (allocation.base + offset)
      (by unfold MemorySafety.Contains; omega)
    exact ir_initially_defined_byte_remains_defined program code projected safe state reachable _ initial

def InitiallyDefinedOutputs (program : IRProgram) (outputs : List Nat) : Prop :=
  ∀ index ∈ outputs, ∃ allocation,
    program.allocations[index]? = some allocation ∧ allocation.initiallyDefined = true

def OutputsInitialized (program : IRProgram) (outputs : List Nat) : Prop :=
  ∀ state, IRReachable program state → ∀ index ∈ outputs, ∃ allocation,
    program.allocations[index]? = some allocation ∧ state.live[index]? = some true ∧
    ∀ offset < allocation.bytes,
      MemorySafety.byteDefined (memoryFrame program) state (allocation.base + offset) = true

def LaunchMemorySafe (program : IRProgram) (outputs : List Nat) : Prop :=
  IRMemorySafe program ∧ OutputsInitialized program outputs

theorem projected_launch_memory_safe (program : IRProgram) (outputs : List Nat)
    (code : List MemorySafety.Instruction)
    (projected : List.Forall₂ (fun instruction target =>
      ∀ valuation, interpretInstruction (bindGlobals program.globals valuation) instruction = target) program.code code)
    (safe : MemorySafety.MemorySafe (projectedProgram program code))
    (outputsDefined : InitiallyDefinedOutputs program outputs) : LaunchMemorySafe program outputs := by
  refine ⟨projected_memory_safe program code projected safe, ?_⟩
  intro state reachable index output
  obtain ⟨allocation, found, defined⟩ := outputsDefined index output
  have invariant := ir_reachable_definedness program code projected safe state reachable
  have indexInside := (List.getElem?_eq_some_iff.mp found).1
  refine ⟨allocation, found, ?_, ?_⟩
  · rw [invariant.2]
    simp [indexInside]
  · intro offset inside
    exact (ir_initially_defined_allocation_survives program code projected safe state reachable
      index allocation found defined offset inside).2

end MemorySafetyIR
