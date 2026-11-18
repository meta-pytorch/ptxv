/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.IndependentModel
import MemorySafetyExecution.MachineModelSafety
import MemorySafetyExecution.Native.Enabledness

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 6000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native.Independent

theorem source_terminal_iff (state : CoreState) :
    execution_core.terminal state = .ok true ↔ MachineModel.Terminal (stateModel state) := by
  rw [source_terminal]
  simp only [ok.injEq]
  constructor
  · intro terminal thread member
    change thread ∈ state.threads.val.map threadModel at member
    obtain ⟨raw, rawMember, rfl⟩ := List.mem_map.mp member
    have flag := (List.all_eq_true.mp terminal) raw rawMember
    cases value : raw.running <;> simp_all only [threadModel, Bool.not_false, Bool.not_true,
      Bool.false_eq_true]
  · intro terminal
    apply List.all_eq_true.mpr
    intro raw member
    have flag := terminal (threadModel raw) (List.mem_map.mpr ⟨raw, member, rfl⟩)
    change raw.running = false at flag
    simp only [flag, Bool.not_false]

theorem source_running_of_model (state : CoreState) (actor : Nat)
    (thread : MachineModel.Thread)
    (selected : (stateModel state).threads[actor]? = some thread)
    (running : thread.running = true) :
    ∃ nativeActor : Usize, nativeActor.val = actor ∧
      execution_core.running state nativeActor = .ok true := by
  simp only [stateModel, List.getElem?_map] at selected
  cases found : state.threads.val[actor]? with
  | none => simp only [found, Option.map_none, reduceCtorEq] at selected
  | some raw =>
    have same : threadModel raw = thread := by
      simpa only [found, Option.map_some, Option.some.injEq] using selected
    have rawRunning : raw.running = true := by
      simpa only [←same, threadModel] using running
    have inside := (List.getElem?_eq_some_iff.mp found).1
    have bound : actor < 2 ^ UScalarTy.Usize.numBits := by
      have maximum := state.threads.property
      scalar_tac
    let nativeActor := Usize.ofNatCore actor bound
    have exactActor : nativeActor.val = actor := Usize.ofNatCore_val_eq bound
    refine ⟨nativeActor, exactActor, ?_⟩
    have test : nativeActor < alloc.vec.Vec.len state.threads := by scalar_tac
    simp only [execution_core.running, test, ite_true, alloc.vec.Vec.index_slice_index,
      alloc.vec.Vec.index_usize, alloc.vec.Vec.getElem?_Nat_eq, exactActor, found,
      bind_ok, rawRunning]

private theorem flags_loop_spec (values : alloc.vec.Vec Bool) (index : Usize) (defined : Bool) :
    execution_core.append_output_loop0 values index defined
      ⦃ result => result = (defined && (values.val.drop index.val).all id) ⦄ := by
  unfold execution_core.append_output_loop0
  apply loop.spec_decr_nat
    (fun current : Usize × Bool => values.val.length - current.1.val)
    (fun current : Usize × Bool =>
      (current.2 && (values.val.drop current.1.val).all id) =
        (defined && (values.val.drop index.val).all id))
  · rintro ⟨cursor, flag⟩ invariant
    unfold execution_core.append_output_loop0.body
    by_cases inside : cursor.val < values.val.length
    · have test : cursor < alloc.vec.Vec.len values := by scalar_tac
      simp only [test, ite_true]
      cases flag with
      | false => simpa only [Bool.false_eq_true, ite_false, WP.spec_ok, Bool.false_and] using invariant
      | true =>
        simp only [ite_true]
        step as ⟨entry, entryEq⟩
        step as ⟨next, nextVal⟩
        refine ⟨?_, by scalar_tac⟩
        rw [nextVal]
        rw [List.drop_eq_getElem_cons inside] at invariant
        simpa only [List.all_cons, Bool.true_and, entryEq, id_eq] using invariant
    · have test : ¬cursor < alloc.vec.Vec.len values := by scalar_tac
      simp only [test, ite_false, WP.spec_ok]
      simpa only [List.drop_eq_nil_of_le (show values.val.length ≤ cursor.val by omega),
        List.all_nil, Bool.and_true] using invariant
  · rfl

private theorem flags_all (values : alloc.vec.Vec Bool)
    (computed : execution_core.append_output_loop0 values 0#usize true = .ok true) :
    ∀ flag ∈ values.val, flag = true := by
  have specification := flags_loop_spec values 0#usize true
  simp only [computed, WP.spec_ok] at specification
  have complete : values.val.all id = true := by
    simpa using specification.symm
  exact List.all_eq_true.mp complete

private def SelectedOutputInitialized (state : CoreState) (object : Usize) : Prop :=
  ∃ memory, state.allocations.val[object.val]? = some memory ∧ memory.live = true ∧
    memory.defined.val.length = memory.bytes.val.length ∧ ∀ flag ∈ memory.defined.val, flag = true

private theorem append_output_initialized (state : CoreState) (object : Usize)
    (before after : alloc.vec.Vec U8)
    (computed : execution_core.append_output state object before = .ok (.Ok (), after)) :
    SelectedOutputInitialized state object := by
  unfold execution_core.append_output at computed
  by_cases available : object >= alloc.vec.Vec.len state.allocations
  · simp only [available, ite_true, ok.injEq, Prod.mk.injEq, reduceCtorEq, false_and] at computed
  · simp only [available, ite_false] at computed
    obtain ⟨memory, read, computed⟩ := MemorySafety.Source.source_bind_success _ _ _ computed
    by_cases live : memory.live = true
    · simp only [live, ite_true] at computed
      by_cases shape : alloc.vec.Vec.len memory.defined != alloc.vec.Vec.len memory.bytes
      · simp only [shape, ite_true, ok.injEq, Prod.mk.injEq, reduceCtorEq, false_and] at computed
      · simp only [shape] at computed
        obtain ⟨defined, scanned, computed⟩ := MemorySafety.Source.source_bind_success _ _ _ computed
        cases defined with
        | false => simp only [Bool.false_eq_true, ite_false, ok.injEq, Prod.mk.injEq,
            reduceCtorEq, false_and] at computed
        | true =>
          have selected := MemorySafety.Source.vec_index_success state.allocations object memory
            (by simpa only [alloc.vec.Vec.index_slice_index] using read)
          refine ⟨memory, selected, live, ?_, flags_all memory.defined scanned⟩
          scalar_tac
    · simp only [live, ite_false, ok.injEq, Prod.mk.injEq, reduceCtorEq, false_and] at computed

private theorem outputs_unroll (state : CoreState) (allocations : Slice Usize)
    (bytes : alloc.vec.Vec U8) (fault : Option Fault) (index : Usize) :
    execution_core.outputs_loop state allocations bytes fault index = (do
      let control ← execution_core.outputs_loop.body state allocations bytes fault index
      match control with
      | .cont triple => execution_core.outputs_loop state allocations triple.1 triple.2.1 triple.2.2
      | .done result => .ok result) := by
  unfold execution_core.outputs_loop
  rw [loop.eq_def]
  congr 1
  funext control
  cases control with
  | cont triple => rcases triple with ⟨bytes, fault, index⟩; rfl
  | done result => rfl

private theorem outputs_fault (state : CoreState) (allocations : Slice Usize)
    (bytes : alloc.vec.Vec U8) (fault : Fault) (index : Usize) :
    execution_core.outputs_loop state allocations bytes (some fault) index = .ok (.Err fault) := by
  rw [outputs_unroll]
  by_cases inside : index < Slice.len allocations
  · simp [execution_core.outputs_loop.body, inside, core.option.Option.is_none, Option.isNone, bind_ok]
  · simp [execution_core.outputs_loop.body, inside, bind_ok]

private theorem outputs_loop_initialized (state : CoreState) (allocations : Slice Usize)
    (before after : alloc.vec.Vec U8) (index : Usize)
    (computed : execution_core.outputs_loop state allocations before none index = .ok (.Ok after)) :
    ∀ object ∈ allocations.val.drop index.val, SelectedOutputInitialized state object := by
  generalize fuelEq : allocations.val.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing before after index with
  | h fuel ih =>
    rw [outputs_unroll] at computed
    by_cases inside : index.val < allocations.val.length
    · have test : index < Slice.len allocations := by scalar_tac
      have incrementBound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := allocations.property
        scalar_tac
      let following := Usize.ofNatCore (index.val + 1) incrementBound
      have followingVal : following.val = index.val + 1 := Usize.ofNatCore_val_eq incrementBound
      have increment := MemorySafety.Source.source_usize_increment index incrementBound
      simp [execution_core.outputs_loop.body, test, core.option.Option.is_none, Option.isNone,
        Std.bind_assoc] at computed
      obtain ⟨object, read, computed⟩ := MemorySafety.Source.source_bind_success _ _ _ computed
      obtain ⟨pair, appended, computed⟩ := MemorySafety.Source.source_bind_success _ _ _ computed
      rcases pair with ⟨result, nextBytes⟩
      cases result with
      | Err fault =>
        simp [bind_ok, increment, outputs_fault, ok.injEq, reduceCtorEq] at computed
      | Ok unitValue =>
        cases unitValue
        have indexed := Slice.index_usize_spec allocations index inside
        simp only [read, WP.spec_ok] at indexed
        have same : allocations.val[index.val] = object := indexed.symm
        have current := append_output_initialized state object before nextBytes appended
        simp [bind_ok, increment] at computed
        have remaining := ih (allocations.val.length - following.val) (by omega)
          nextBytes after following computed rfl
        intro query member
        rw [List.drop_eq_getElem_cons inside] at member
        rcases List.mem_cons.mp member with head | tail
        · exact (head.trans same).symm ▸ current
        · apply remaining query
          simpa only [followingVal] using tail
    · intro object member
      have empty : allocations.val.drop index.val = [] := List.drop_eq_nil_of_le (by omega)
      simp only [empty, List.not_mem_nil] at member

theorem source_outputs_initialized (program : CoreProgram) (state : CoreState)
    (bytes : alloc.vec.Vec U8)
    (computed : execution_core.outputs state (alloc.vec.Vec.deref program.outputs) = .ok (.Ok bytes)) :
    MachineModel.OutputsInitialized (programModel program) (stateModel state) := by
  have initialized := outputs_loop_initialized state (alloc.vec.Vec.deref program.outputs)
    (alloc.vec.Vec.new U8) bytes 0#usize computed
  intro object member
  change object ∈ program.outputs.val.map UScalar.val at member
  obtain ⟨nativeObject, nativeMember, rfl⟩ := List.mem_map.mp member
  have valid := initialized nativeObject (by
    simpa [alloc.vec.Vec.deref, Slice.from_val] using nativeMember)
  obtain ⟨memory, selected, live, shape, defined⟩ := valid
  refine ⟨memoryModel memory, ?_, live, ?_, defined⟩
  · simp only [stateModel, List.getElem?_map, selected, Option.map_some]
  · simpa only [memoryModel, List.length_map] using shape

end MemorySafetyExecution.Native.Independent
