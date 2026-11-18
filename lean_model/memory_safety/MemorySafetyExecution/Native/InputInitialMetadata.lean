/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.SourceLoop
import MemorySafetyExecution.Native.InputState

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety.ptxv_memory_write
open MemorySafety.Source

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

open ValueFlow.Source

def initialInputMetadata (defined : Bool) : Metadata :=
  { captured:=0#u32, history:=none, tainted:=true, defined }

def initialInputRegistersModel (registers : value_flow.Registers) : List Metadata × List Metadata :=
  (List.replicate registers.raw.regs.val.length (initialInputMetadata false),
    List.replicate (registers.raw.preds.val.length*64) (initialInputMetadata false))

def inputRegistersModel (registers : DependencyRegisters) : List Metadata × List Metadata :=
  (registers.virtuals.val,registers.predicates.val)

private theorem initial_virtuals_loop (values : alloc.vec.Vec U64)
    (virtuals : alloc.vec.Vec Metadata) (index : Usize)
    (bound : index.val ≤ values.val.length)
    (initial : virtuals.val=List.replicate index.val (initialInputMetadata false)) :
    input_independent_execution.initial_registers_loop0 values virtuals index
      ⦃ output => output.val=List.replicate values.val.length (initialInputMetadata false) ⦄ := by
  unfold input_independent_execution.initial_registers_loop0
  apply loop.spec_decr_nat (fun current => values.val.length-current.2.val)
    (fun current => current.2.val≤values.val.length ∧
      current.1.val=List.replicate current.2.val (initialInputMetadata false))
  · rintro ⟨virtuals,index⟩ ⟨bound,initial⟩
    unfold input_independent_execution.initial_registers_loop0.body
    by_cases inside : index < alloc.vec.Vec.len values
    · simp only [inside,ite_true,value_flow_dependencies.metadata,symbolic.prepare,bind_ok]
      have room := alloc.vec.Vec.len_ineq values
      step as ⟨nextVirtuals,appended⟩
      step as ⟨next,nextVal⟩
      refine ⟨by scalar_tac,?_,by scalar_tac⟩
      simpa only [nextVal,List.replicate_add,List.replicate_one,initial,initialInputMetadata] using appended
    · simp only [inside,ite_false,WP.spec_ok]
      have atEnd : index.val=values.val.length := by scalar_tac
      simpa only [atEnd] using initial
  · exact ⟨bound,initial⟩

private theorem initial_predicates_loop (count index : Usize) (predicates : alloc.vec.Vec Metadata)
    (bound : index.val≤count.val)
    (initial : predicates.val=List.replicate index.val (initialInputMetadata false)) :
    input_independent_execution.initial_registers_loop1 count index predicates
      ⦃ output => output.val=List.replicate count.val (initialInputMetadata false) ⦄ := by
  unfold input_independent_execution.initial_registers_loop1
  apply loop.spec_decr_nat (fun current => count.val-current.1.val)
    (fun current => current.1.val≤count.val ∧
      current.2.val=List.replicate current.1.val (initialInputMetadata false))
  · rintro ⟨index,predicates⟩ ⟨bound,initial⟩
    unfold input_independent_execution.initial_registers_loop1.body
    by_cases inside : index < count
    · simp only [inside,ite_true,value_flow_dependencies.metadata,symbolic.prepare,bind_ok]
      step as ⟨nextPredicates,appended⟩
      step as ⟨next,nextVal⟩
      refine ⟨by scalar_tac,?_,by scalar_tac⟩
      simpa only [nextVal,List.replicate_add,List.replicate_one,initial,initialInputMetadata] using appended
    · simp only [inside,ite_false,WP.spec_ok]
      have atEnd : index.val=count.val := by scalar_tac
      simpa only [atEnd] using initial
  · exact ⟨bound,initial⟩

theorem source_initial_input_registers (registers : value_flow.Registers)
    (shadow : DependencyRegisters)
    (computed : input_independent_execution.initial_registers registers = .ok (some shadow)) :
    inputRegistersModel shadow=initialInputRegistersModel registers := by
  unfold input_independent_execution.initial_registers at computed
  simp only [lift,bind_ok] at computed
  cases checked : Usize.checked_mul (alloc.vec.Vec.len registers.raw.preds) 64#usize with
  | none => simp [checked] at computed
  | some count =>
    simp only [checked] at computed
    have multiplication := Usize.checked_mul_bv_spec (alloc.vec.Vec.len registers.raw.preds) 64#usize
    rw [checked] at multiplication
    have countVal : count.val=registers.raw.preds.val.length*64 := by
      simpa only [alloc.vec.Vec.len_val,UScalar.ofNatCore_val_eq] using multiplication.2.1
    obtain ⟨virtuals,virtualsComputed,computed⟩ := source_bind_success _ _ _ computed
    obtain ⟨predicates,predicatesComputed,computed⟩ := source_bind_success _ _ _ computed
    have virtualsLaw := initial_virtuals_loop registers.raw.regs (alloc.vec.Vec.new Metadata) 0#usize (by simp) (by simp)
    have predicatesLaw := initial_predicates_loop count 0#usize (alloc.vec.Vec.new Metadata) (by simp) (by simp)
    rw [virtualsComputed] at virtualsLaw
    rw [predicatesComputed] at predicatesLaw
    simp only [WP.spec_ok] at virtualsLaw predicatesLaw
    have shadowEq : ({virtuals,predicates} : DependencyRegisters)=shadow := by simpa using computed
    subst shadow
    dsimp only [inputRegistersModel,initialInputRegistersModel]
    exact Prod.ext virtualsLaw (by simpa only [countVal] using predicatesLaw)

private theorem initial_memory_loop (memory : execution_core.AllocationBytes)
    (shadow : alloc.vec.Vec Metadata) (index : Usize)
    (bound : index.val≤memory.defined.val.length)
    (initial : shadow.val=(memory.defined.val.take index.val).map initialInputMetadata) :
    input_independent_execution.initial_memory_loop memory shadow index
      ⦃ output => output.val=memory.defined.val.map initialInputMetadata ⦄ := by
  unfold input_independent_execution.initial_memory_loop
  apply loop.spec_decr_nat (fun current => memory.defined.val.length-current.2.val)
    (fun current => current.2.val≤memory.defined.val.length ∧
      current.1.val=(memory.defined.val.take current.2.val).map initialInputMetadata)
  · rintro ⟨shadow,index⟩ ⟨bound,initial⟩
    unfold input_independent_execution.initial_memory_loop.body
    by_cases inside : index < alloc.vec.Vec.len memory.defined
    · simp only [inside,ite_true]
      have room := alloc.vec.Vec.len_ineq memory.defined
      have shadowLength : shadow.val.length=index.val := by
        rw [initial,List.length_map,List.length_take,Nat.min_eq_left bound]
      step as ⟨flag,flagEq⟩
      simp only [value_flow_dependencies.metadata,symbolic.prepare,bind_ok]
      step as ⟨nextShadow,appended⟩
      step as ⟨next,nextVal⟩
      refine ⟨by scalar_tac,?_,by scalar_tac⟩
      have indexBound : index.val<memory.defined.val.length := by scalar_tac
      simpa only [nextVal,List.take_succ_eq_append_getElem indexBound,List.map_append,
        List.map_cons,List.map_nil,initial,initialInputMetadata,←flagEq] using appended
    · simp only [inside,ite_false,WP.spec_ok]
      have atEnd : index.val=memory.defined.val.length := by scalar_tac
      simpa only [atEnd,List.take_length] using initial
  · exact ⟨bound,initial⟩

@[step]
theorem source_initial_input_memory (memory : execution_core.AllocationBytes) :
    input_independent_execution.initial_memory memory
      ⦃ output => output.val=memory.defined.val.map initialInputMetadata ⦄ := by
  exact initial_memory_loop memory (alloc.vec.Vec.new Metadata) 0#usize (by simp) (by simp)

private theorem initial_memory_inventory_loop (allocations : alloc.vec.Vec execution_core.AllocationBytes)
    (index : Usize) (memory : alloc.vec.Vec (alloc.vec.Vec Metadata))
    (bound : index.val≤allocations.val.length)
    (initial : memory.val.map (fun entry=>entry.val)=
      (allocations.val.take index.val).map (fun allocation=>allocation.defined.val.map initialInputMetadata)) :
    input_independent_execution.initial_dependencies_loop0_loop0 allocations index memory
      ⦃ output => output.val.map (fun entry=>entry.val)=
        allocations.val.map (fun allocation=>allocation.defined.val.map initialInputMetadata) ⦄ := by
  unfold input_independent_execution.initial_dependencies_loop0_loop0
  apply loop.spec_decr_nat (fun current => allocations.val.length-current.1.val)
    (fun current => current.1.val≤allocations.val.length ∧
      current.2.val.map (fun entry=>entry.val)=
        (allocations.val.take current.1.val).map (fun allocation=>allocation.defined.val.map initialInputMetadata))
  · rintro ⟨index,memory⟩ ⟨bound,initial⟩
    unfold input_independent_execution.initial_dependencies_loop0_loop0.body
    by_cases inside : index < alloc.vec.Vec.len allocations
    · simp only [inside,ite_true]
      have room := alloc.vec.Vec.len_ineq allocations
      have memoryLength : memory.val.length=index.val := by
        have length:=congrArg List.length initial
        simpa only [List.length_map,List.length_take,Nat.min_eq_left bound] using length
      step as ⟨allocation,allocationEq⟩
      step with source_initial_input_memory allocation as ⟨shadow,shadowEq⟩
      step as ⟨nextMemory,appended⟩
      step as ⟨next,nextVal⟩
      refine ⟨by scalar_tac,?_,by scalar_tac⟩
      have indexBound : index.val<allocations.val.length := by scalar_tac
      rw [appended,List.map_append,List.map_cons,List.map_nil,initial]
      simp only [nextVal,List.take_succ_eq_append_getElem indexBound,List.map_append,
        List.map_cons,List.map_nil,shadowEq,←allocationEq]
    · simp only [inside,ite_false,WP.spec_ok]
      have atEnd : index.val=allocations.val.length := by scalar_tac
      simpa only [atEnd,List.take_length] using initial
  · exact ⟨bound,initial⟩

theorem source_initial_input_memory_inventory (allocations : alloc.vec.Vec execution_core.AllocationBytes)
    (memory : alloc.vec.Vec (alloc.vec.Vec Metadata))
    (computed : input_independent_execution.initial_dependencies_loop0_loop0 allocations 0#usize
      (alloc.vec.Vec.new (alloc.vec.Vec Metadata))=.ok memory) :
    memory.val.map (fun entry=>entry.val)=
      allocations.val.map (fun allocation=>allocation.defined.val.map initialInputMetadata) := by
  have law:=initial_memory_inventory_loop allocations 0#usize (alloc.vec.Vec.new (alloc.vec.Vec Metadata))
    (by simp) (by simp)
  rw [computed] at law
  simpa only [WP.spec_ok] using law

private theorem initial_dependencies_loop (state : CoreState)
    (threads : alloc.vec.Vec DependencyRegisters) (index : Usize)
    (bound : index.val≤state.threads.val.length)
    (initial : threads.val.map inputRegistersModel=
      (state.threads.val.take index.val).map (fun thread=>initialInputRegistersModel thread.registers))
    (output : Option value_flow_dependencies.State)
    (computed : input_independent_execution.initial_dependencies_loop0 state threads index=.ok output) :
    ∀ dependencies, output=some dependencies →
      dependencies.threads.val.map inputRegistersModel=
        state.threads.val.map (fun thread=>initialInputRegistersModel thread.registers) ∧
      dependencies.memory.val.map (fun entry=>entry.val)=
        state.allocations.val.map (fun allocation=>allocation.defined.val.map initialInputMetadata) := by
  unfold input_independent_execution.initial_dependencies_loop0 at computed
  apply PtxFrontendProgram.source_loop_success _ (fun current=>state.threads.val.length-current.2.val)
    (fun current=>current.2.val≤state.threads.val.length ∧
      current.1.val.map inputRegistersModel=
        (state.threads.val.take current.2.val).map (fun thread=>initialInputRegistersModel thread.registers))
    (fun (output : Option value_flow_dependencies.State)=>∀ dependencies,output=some dependencies →
      dependencies.threads.val.map inputRegistersModel=
        state.threads.val.map (fun thread=>initialInputRegistersModel thread.registers) ∧
      dependencies.memory.val.map (fun entry=>entry.val)=
        state.allocations.val.map (fun allocation=>allocation.defined.val.map initialInputMetadata))
    ?_ (threads,index) output ⟨bound,initial⟩ computed
  rintro ⟨threads,index⟩ flow ⟨bound,initial⟩ computed
  unfold input_independent_execution.initial_dependencies_loop0.body at computed
  by_cases inside : index < alloc.vec.Vec.len state.threads
  · simp only [inside,ite_true] at computed
    obtain ⟨thread,read,computed⟩:=source_bind_success _ _ _ computed
    have threadAt:=vec_index_success state.threads index thread read
    obtain ⟨registers,registersComputed,computed⟩:=source_bind_success _ _ _ computed
    cases registers with
    | none =>
      simp only [ok.injEq] at computed
      subst flow
      intro dependencies impossible
      cases impossible
    | some registers =>
      have registersEq:=source_initial_input_registers thread.registers registers registersComputed
      obtain ⟨nextThreads,pushed,computed⟩:=source_bind_success _ _ _ computed
      have threadsEq:=vec_push_success threads nextThreads registers pushed
      obtain ⟨next,increment,computed⟩:=source_bind_success _ _ _ computed
      have nextVal : next.val=index.val+1 := by
        have room:=alloc.vec.Vec.len_ineq state.threads
        have law:=Usize.add_spec (x:=index) (y:=1#usize) (by scalar_tac)
        rw [increment] at law
        simpa only [WP.spec_ok,UScalar.ofNatCore_val_eq] using law
      simp only [ok.injEq] at computed
      subst flow
      refine ⟨⟨by scalar_tac,?_⟩,by scalar_tac⟩
      have indexBound : index.val<state.threads.val.length := by scalar_tac
      have threadEq : state.threads.val[index.val]=thread := by
        simpa only [List.getElem?_eq_getElem indexBound,Option.some.injEq] using threadAt
      rw [threadsEq,List.map_append,List.map_cons,List.map_nil,initial]
      simp only [nextVal,List.take_succ_eq_append_getElem indexBound,List.map_append,
        List.map_cons,List.map_nil,threadEq,registersEq]
  · simp only [inside,ite_false] at computed
    obtain ⟨memory,memoryComputed,computed⟩:=source_bind_success _ _ _ computed
    have memoryEq:=source_initial_input_memory_inventory state.allocations memory memoryComputed
    simp only [ok.injEq] at computed
    subst flow
    intro dependencies same
    cases same
    have atEnd : index.val=state.threads.val.length := by scalar_tac
    exact ⟨by simpa only [atEnd,List.take_length] using initial,memoryEq⟩

theorem source_initial_input_dependencies (state : CoreState) (dependencies : value_flow_dependencies.State)
    (computed : input_independent_execution.initial_dependencies state=.ok (some dependencies)) :
    dependencies.threads.val.map inputRegistersModel=
      state.threads.val.map (fun thread=>initialInputRegistersModel thread.registers) ∧
    dependencies.memory.val.map (fun entry=>entry.val)=
      state.allocations.val.map (fun allocation=>allocation.defined.val.map initialInputMetadata) := by
  exact initial_dependencies_loop state (alloc.vec.Vec.new DependencyRegisters) 0#usize
    (by simp) (by simp) (some dependencies) computed dependencies rfl

end MemorySafetyExecution.Native
