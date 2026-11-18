/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Effects

open Aeneas Aeneas.Std PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.Consumers

def sameKnownOrigin (left right : Cell) : Bool :=
  match left.origin, right.origin with
  | some left, some right => decide (left = right)
  | _, _ => false

theorem sameKnownOrigin_iff (left right : Cell) :
    sameKnownOrigin left right = true ↔
      ∃ origin, left.origin = some origin ∧ right.origin = some origin := by
  cases hl : left.origin <;> cases hr : right.origin <;>
    simp [sameKnownOrigin, hl, hr, eq_comm]

theorem sameKnownOrigin_sound {Input : Type} (sem : Semantics Input)
    (leftMemory rightMemory : Memory Input) (left right : Cell)
    (hleft : CellRepresents sem leftMemory left) (hright : CellRepresents sem rightMemory right)
    (hcheck : sameKnownOrigin left right = true) (input : Input) (hinput : sem.domain input) :
    (leftMemory input (location left.key)).defined = true ∧
      (rightMemory input (location right.key)).defined = true ∧
      (leftMemory input (location left.key)).bits = (rightMemory input (location right.key)).bits := by
  obtain ⟨origin, hl, hr⟩ := (sameKnownOrigin_iff left right).mp hcheck
  have hleftByte := hleft.byte.origin_sound origin hl input hinput
  have hrightByte := hright.byte.origin_sound origin hr input hinput
  exact ⟨hleftByte.1, hrightByte.1, hleftByte.2.trans hrightByte.2.symm⟩

def sameKnownWord (left right : Cell) : Bool :=
  match left.word, right.word with
  | some (.Known left), some (.Known right) => decide (left = right)
  | _, _ => false

theorem sameKnownWord_iff (left right : Cell) :
    sameKnownWord left right = true ↔
      ∃ id, left.word = some (.Known id) ∧ right.word = some (.Known id) := by
  cases hl : left.word with
  | none => simp [sameKnownWord, hl]
  | some leftHistory =>
    cases leftHistory <;> cases hr : right.word with
    | none => simp [sameKnownWord, hl, hr]
    | some rightHistory =>
      cases rightHistory <;> simp [sameKnownWord, hl, hr, eq_comm]

theorem sameKnownWord_sound {Input : Type} (sem : Semantics Input)
    (leftMemory rightMemory : Memory Input) (left right : Cell)
    (hleft : CellRepresents sem leftMemory left) (hright : CellRepresents sem rightMemory right)
    (hcheck : sameKnownWord left right = true) (input : Input) (hinput : sem.domain input)
    (lane : Fin 4) :
    (leftMemory input (location left.key lane.val)).defined = true ∧
      (rightMemory input (location right.key lane.val)).defined = true ∧
      (leftMemory input (location left.key lane.val)).bits =
        (rightMemory input (location right.key lane.val)).bits := by
  obtain ⟨id, hl, hr⟩ := (sameKnownWord_iff left right).mp hcheck
  have hleftWord := hleft.word id hl input hinput lane
  have hrightWord := hright.word id hr input hinput lane
  exact ⟨hleftWord.1, hrightWord.1, hleftWord.2.trans hrightWord.2.symm⟩

theorem sameKnownWord_rejects_opaque (left right : Cell) (id : U32)
    (hleft : left.word = some (.Opaque id)) : sameKnownWord left right = false := by
  simp [sameKnownWord, hleft]

theorem untainted_independent {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (cell : Cell) (hcell : CellRepresents sem memory cell)
    (huntainted : cell.tainted = false) (left right : Input)
    (hleft : sem.domain left) (hright : sem.domain right) :
    (memory left (location cell.key)).bits = (memory right (location cell.key)).bits := by
  exact (hcell.byte.constant huntainted left hleft).trans
    (hcell.byte.constant huntainted right hright).symm

def sameUntaintedCapture (left right : Cell) : Bool :=
  decide (left.tainted = false ∧ right.tainted = false ∧ left.defined = true ∧
    right.defined = true ∧ left.value = right.value)

theorem sameUntaintedCapture_sound {Input : Type} (sem : Semantics Input)
    (leftMemory rightMemory : Memory Input) (left right : Cell)
    (hleft : CellRepresents sem leftMemory left) (hright : CellRepresents sem rightMemory right)
    (hcheck : sameUntaintedCapture left right = true) (leftInput rightInput : Input)
    (hleftInput : sem.domain leftInput) (hrightInput : sem.domain rightInput) :
    (leftMemory leftInput (location left.key)).defined = true ∧
      (rightMemory rightInput (location right.key)).defined = true ∧
      (leftMemory leftInput (location left.key)).bits =
        (rightMemory rightInput (location right.key)).bits := by
  have hconditions : left.tainted = false ∧ right.tainted = false ∧ left.defined = true ∧
      right.defined = true ∧ left.value = right.value := of_decide_eq_true hcheck
  obtain ⟨hlt, hrt, hld, hrd, hvalue⟩ := hconditions
  refine ⟨hleft.byte.defined_sound hld leftInput hleftInput,
    hright.byte.defined_sound hrd rightInput hrightInput, ?_⟩
  calc
    _ = left.value := (hleft.byte.constant hlt leftInput hleftInput).trans hleft.byte.captured.symm
    _ = right.value := hvalue
    _ = _ := hright.byte.captured.trans (hright.byte.constant hrt rightInput hrightInput).symm

def originWithFallback (cell : Cell) (initialOrigin : Key → Option Origin) : Option Origin :=
  match cell.origin with
  | some origin => some origin
  | none => if cell.written then none else initialOrigin cell.key

def InitialOriginsSound {Input : Type} (sem : Semantics Input)
    (initialOrigin : Key → Option Origin) : Prop :=
  ∀ key origin, initialOrigin key = some origin → ∀ input, sem.domain input →
    (sem.initial input (location key)).defined = true ∧
      (sem.initial input (location key)).bits = sem.origin origin input

theorem originWithFallback_sound {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (cell : Cell) (initialOrigin : Key → Option Origin)
    (hcell : CellRepresents sem memory cell) (hinitial : InitialOriginsSound sem initialOrigin)
    (origin : Origin) (hlookup : originWithFallback cell initialOrigin = some origin)
    (input : Input) (hinput : sem.domain input) :
    (memory input (location cell.key)).defined = true ∧
      (memory input (location cell.key)).bits = sem.origin origin input := by
  cases horigin : cell.origin with
  | some existing =>
    have heq : existing = origin := by simpa [originWithFallback, horigin] using hlookup
    subst existing
    exact hcell.byte.origin_sound origin horigin input hinput
  | none =>
    cases hwritten : cell.written with
    | true => simp [originWithFallback, horigin, hwritten] at hlookup
    | false =>
      have hsource : initialOrigin cell.key = some origin := by
        simpa [originWithFallback, horigin, hwritten] using hlookup
      rw [hcell.initial_fallback hwritten input hinput]
      exact hinitial cell.key origin hsource input hinput

theorem originWithFallback_does_not_resurrect (cell : Cell) (initialOrigin : Key → Option Origin)
    (hunknown : cell.origin = none) (hwritten : cell.written = true) :
    originWithFallback cell initialOrigin = none := by
  simp [originWithFallback, hunknown, hwritten]

def initialWordFallback (cells : Fin 4 → Cell) (initialHistory : Option WordHistory) :
    Option WordHistory :=
  if ∀ lane, (cells lane).written = false then initialHistory else none

theorem initialWordFallback_sound {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (key : Key) (cells : Fin 4 → Cell)
    (initialHistory : Option WordHistory)
    (hcell : ∀ lane, CellRepresents sem memory (cells lane))
    (hpositions : ∀ lane, location (cells lane).key = location key lane.val)
    (hinitial : WordSound sem sem.initial key initialHistory) :
    WordSound sem memory key (initialWordFallback cells initialHistory) := by
  intro id hid input hinput lane
  by_cases hunwritten : ∀ lane, (cells lane).written = false
  · have hhistory : initialHistory = some (.Known id) := by
      simpa [initialWordFallback, hunwritten] using hid
    have hcurrent := (hcell lane).initial_fallback (hunwritten lane) input hinput
    rw [hpositions lane] at hcurrent
    rw [hcurrent]
    exact hinitial id hhistory input hinput lane
  · simp [initialWordFallback, hunwritten] at hid

theorem initialWordFallback_rejects_partial_write (cells : Fin 4 → Cell)
    (initialHistory : Option WordHistory) (lane : Fin 4) (hwritten : (cells lane).written = true) :
    initialWordFallback cells initialHistory = none := by
  have hnot : ¬∀ lane, (cells lane).written = false := by
    intro h
    have hf := h lane
    rw [hwritten] at hf
    cases hf
  simp [initialWordFallback, hnot]

theorem storage_knownOrigin_consumer {Input : Type} (sem : Semantics Input)
    (leftMemory rightMemory : Memory Input) (leftState rightState : Storage) (left right : Key)
    (hleft : Storage.Represents sem leftMemory leftState)
    (hright : Storage.Represents sem rightMemory rightState)
    (hleftKey : left ∈ leftState.keys.val) (hrightKey : right ∈ rightState.keys.val)
    (hcheck : sameKnownOrigin (leftState.getCell left) (rightState.getCell right) = true)
    (input : Input) (hinput : sem.domain input) :
    (leftMemory input (location left)).defined = true ∧
      (rightMemory input (location right)).defined = true ∧
      (leftMemory input (location left)).bits = (rightMemory input (location right)).bits := by
  exact sameKnownOrigin_sound sem leftMemory rightMemory _ _
    (hleft _ (Storage.getCell_mem leftState left hleftKey))
    (hright _ (Storage.getCell_mem rightState right hrightKey)) hcheck input hinput

end MemoryWrite.Consumers
