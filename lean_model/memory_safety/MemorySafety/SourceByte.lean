/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.SourcePrimitives

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 3000000
set_option Elab.async false

namespace MemorySafety.Source

def writtenValue (rows : List memory_safety.WrittenSpan) (address : Nat) : Option Bool :=
  (rows.reverse.find? (fun row => decide (Contains row.address.val row.width.val address))).map
    (fun row => row.defined)

def allocationValue (rows : List memory_safety.Allocation) (address : Nat) : Bool :=
  ((rows.find? (fun row => decide (Contains row.base.val row.bytes.val address))).map
    (fun row => row.initially_defined)).getD false

@[step]
theorem source_written_byte (rows : alloc.vec.Vec memory_safety.WrittenSpan) (address : U64) :
    memory_safety.byte_defined_loop0 rows address false false (alloc.vec.Vec.len rows)
      ⦃ result => (if result.1 then some result.2 else none) = writtenValue rows.val address.val ∧
        (result.1 = false → result.2 = false) ⦄ := by
  unfold memory_safety.byte_defined_loop0
  apply loop.spec_decr_nat
    (fun current : Bool × Bool × Usize => current.2.2.val)
    (fun current : Bool × Bool × Usize => current.2.2.val ≤ rows.val.length ∧
      (if current.1 then some current.2.1 else writtenValue (rows.val.take current.2.2.val) address.val) =
        writtenValue rows.val address.val ∧ (current.1 = false → current.2.1 = false))
  · intro current invariant
    rcases current with ⟨found, defined, cursor⟩
    obtain ⟨cursorBound, selected, undefined⟩ := invariant
    dsimp only at cursorBound selected undefined
    unfold memory_safety.byte_defined_loop0.body
    by_cases positive : 0 < cursor.val
    · have positiveSource : cursor > 0#usize := by scalar_tac
      simp only [positiveSource, ite_true]
      cases found with
      | true => simpa using And.intro selected undefined
      | false =>
        simp only [Bool.false_eq_true, ite_false] at selected ⊢
        step as ⟨previous, previousEq⟩
        have previousVal : previous.val + 1 = cursor.val := by scalar_tac
        have inside : previous.val < rows.val.length := by omega
        step as ⟨row, rowEq⟩
        have rowIndex : row = rows.val[previous.val]'inside := rowEq
        step with source_contains row.address row.width address as ⟨contains, containsEq⟩
        have takeEq : rows.val.take cursor.val = rows.val.take previous.val ++ [row] := by
          rw [←previousVal, List.take_succ_eq_append_getElem inside, rowIndex]
        have selectedStep : writtenValue (rows.val.take cursor.val) address.val =
            if contains then some row.defined else writtenValue (rows.val.take previous.val) address.val := by
          have decision : decide (Contains row.address.val row.width.val address.val) = contains := by
            cases contains with
            | false =>
              apply decide_eq_false
              intro h
              have impossible := containsEq.mpr h
              cases impossible
            | true => exact decide_eq_true (containsEq.mp rfl)
          simp only [writtenValue, takeEq, List.reverse_append, List.reverse_cons,
            List.reverse_nil, List.nil_append, List.singleton_append, List.find?_cons, decision]
          cases contains <;> rfl
        cases contains with
        | false =>
          simp only [Bool.false_eq_true, ite_false, WP.spec_ok]
          refine ⟨by omega, ?_, by simpa using undefined, by omega⟩
          simpa [selectedStep] using selected
        | true =>
          simp only [ite_true, WP.spec_ok]
          refine ⟨by omega, ?_, by simp, by omega⟩
          simpa [selectedStep] using selected
    · have zeroSource : ¬cursor > 0#usize := by scalar_tac
      simp only [zeroSource, ite_false, WP.spec_ok]
      have zero : cursor.val = 0 := by omega
      simpa [zero, writtenValue] using And.intro selected undefined
  · simp [List.take_length]

@[step]
theorem source_allocation_byte (program : memory_safety.Program) (address : U64)
    (found defined : Bool) (notFound : found = false → defined = false) :
    memory_safety.byte_defined_loop1 program address found defined 0#usize
      ⦃ result => result = if found then defined else allocationValue program.allocations.val address.val ⦄ := by
  unfold memory_safety.byte_defined_loop1
  apply loop.spec_decr_nat
    (fun current : Bool × Bool × Usize => program.allocations.val.length - current.2.2.val)
    (fun current : Bool × Bool × Usize => current.2.2.val ≤ program.allocations.val.length ∧
      (if current.1 then current.2.1 else allocationValue (program.allocations.val.drop current.2.2.val) address.val) =
        (if found then defined else allocationValue program.allocations.val address.val) ∧
      (current.1 = false → current.2.1 = false))
  · intro current invariant
    rcases current with ⟨currentFound, currentDefined, cursor⟩
    obtain ⟨cursorBound, selected, undefined⟩ := invariant
    dsimp only at cursorBound selected undefined
    unfold memory_safety.byte_defined_loop1.body
    by_cases inside : cursor.val < program.allocations.val.length
    · have withinSource : cursor < alloc.vec.Vec.len program.allocations := by scalar_tac
      simp only [withinSource, ite_true]
      cases currentFound with
      | true => simpa using selected
      | false =>
        simp only [Bool.false_eq_true, ite_false] at selected ⊢
        step as ⟨row, rowEq⟩
        have rowIndex : row = program.allocations.val[cursor.val]'inside := rowEq
        step with source_contains row.base row.bytes address as ⟨contains, containsEq⟩
        have dropEq : program.allocations.val.drop cursor.val = row :: program.allocations.val.drop (cursor.val + 1) := by
          rw [List.drop_eq_getElem_cons inside, rowIndex]
        have selectedStep : allocationValue (program.allocations.val.drop cursor.val) address.val =
            if contains then row.initially_defined else allocationValue (program.allocations.val.drop (cursor.val + 1)) address.val := by
          have decision : decide (Contains row.base.val row.bytes.val address.val) = contains := by
            cases contains with
            | false =>
              apply decide_eq_false
              intro h
              have impossible := containsEq.mpr h
              cases impossible
            | true => exact decide_eq_true (containsEq.mp rfl)
          simp only [allocationValue, dropEq, List.find?_cons, decision]
          cases contains <;> rfl
        cases contains with
        | false =>
          simp only [Bool.false_eq_true, ite_false, bind_ok]
          step as ⟨next, nextEq⟩
          refine ⟨by scalar_tac, ?_, by simpa using undefined, by scalar_tac⟩
          have nextVal : next.val = cursor.val + 1 := by scalar_tac
          simpa [nextVal, selectedStep] using selected
        | true =>
          simp only [ite_true, bind_ok]
          step as ⟨next, nextEq⟩
          refine ⟨by scalar_tac, ?_, by simp, by scalar_tac⟩
          simpa [selectedStep] using selected
    · have outsideSource : ¬cursor < alloc.vec.Vec.len program.allocations := by scalar_tac
      simp only [outsideSource, ite_false, WP.spec_ok]
      have endEq : cursor.val = program.allocations.val.length := by omega
      cases currentFound <;> simp_all [allocationValue]
  · exact ⟨by simp, by simp, notFound⟩

@[step]
theorem source_byte_defined (program : memory_safety.Program) (state : memory_safety.State)
    (address : U64) :
    memory_safety.byte_defined program state address
      ⦃ result => result = byteDefined (programModel program) (stateModel state) address.val ⦄ := by
  unfold memory_safety.byte_defined
  step with source_written_byte state.writes address as ⟨found, defined, selectedEq, missing⟩
  step with source_allocation_byte program address found defined missing as ⟨result, resultEq⟩
  have modelEq : byteDefined (programModel program) (stateModel state) address.val =
      (writtenValue state.writes.val address.val).getD (allocationValue program.allocations.val address.val) := by
    simp only [byteDefined, programModel, stateModel, writtenValue, allocationValue]
    rw [←List.map_reverse]
    simp only [List.find?_map]
    simp only [Function.comp_def, writtenModel, allocationModel]
    have projection (w : Option memory_safety.WrittenSpan) (a : Option memory_safety.Allocation) :
        (match w.map writtenModel with
        | some row => row.defined
        | none => match a.map allocationModel with
          | some block => block.initiallyDefined
          | none => false) =
        (w.map (fun row => row.defined)).getD ((a.map (fun block => block.initially_defined)).getD false) := by
      cases w <;> cases a <;> rfl
    exact projection _ _
  rw [modelEq, ←selectedEq]
  cases found <;> simpa using resultEq

end MemorySafety.Source
