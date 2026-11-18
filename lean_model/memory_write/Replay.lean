/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Production
import Lean

open Aeneas Aeneas.Std PtxvMemoryWrite MemoryWrite Lean

namespace MemoryWrite.Replay

def jsonField (json : Json) (caseName : String) : Except String Json := json.getObjVal? caseName

def jsonOptional (json : Json) (caseName : String) (fallback : Json) : Json :=
  (jsonField json caseName).toOption.getD fallback

def nat (json : Json) : Except String Nat := json.getNat?

def u8 (json : Json) : Except String U8 := do
  let value ← nat json
  if h : value ≤ U8.max then return U8.ofNatCore value (by scalar_tac)
  else throw "byte exceeds u8"

def u32 (json : Json) : Except String U32 := do
  let value ← nat json
  if h : value ≤ U32.max then return U32.ofNatCore value (by scalar_tac)
  else throw "node exceeds u32"

def u64 (json : Json) : Except String U64 := do
  let value ← nat json
  if h : value ≤ U64.max then return U64.ofNatCore value (by scalar_tac)
  else throw "address exceeds u64"

def jsonAt (json : Json) (index : Nat) : Except String Json := do
  let values ← json.getArr?
  match values[index]? with
  | none => throw s!"missing array element {index}"
  | some value => return value

def key (json : Json) : Except String Key := do
  return ⟨← u8 (← jsonAt json 0), ← u64 (← jsonAt json 1), ← u64 (← jsonAt json 2)⟩

def origin (json : Json) : Except String (Option Origin) := do
  if json == Json.null then return none
  match ← nat (← jsonAt json 0) with
  | 0 => return some .Zero
  | 1 => return some (.E (← u64 (← jsonAt json 1)) (← u64 (← jsonAt json 2)) (← u8 (← jsonAt json 3)))
  | 2 => return some (.N (← u32 (← jsonAt json 1)) (← u8 (← jsonAt json 3)))
  | 3 => return some (.E7 (← u64 (← jsonAt json 1)) (← u64 (← jsonAt json 2)))
  | tag => throw s!"unknown origin tag {tag}"

def word (json : Json) : Except String (Option WordHistory) := do
  if json == Json.null then return none
  let id ← u32 (← jsonAt json 1)
  match ← nat (← jsonAt json 0) with
  | 0 => return some (.Known id)
  | 1 => return some (.Opaque id)
  | tag => throw s!"unknown word tag {tag}"

def byte (json : Json) : Except String PayloadByte := do
  return ⟨← u8 (← jsonField json "value"), ← origin (← jsonField json "origin"),
    ← (← jsonField json "tainted").getBool?, ← (← jsonField json "defined").getBool?,
    ← word (← jsonField json "word")⟩

def cell (json : Json) : Except String Cell := do
  let data ← byte json
  return ⟨← key (← jsonField json "key"), data.value, data.origin, data.tainted,
    data.defined, ← (← jsonField json "written").getBool?, data.word⟩

def slice {α : Type} (values : List α) : Except String (Slice α) :=
  if h : values.length ≤ Usize.max then .ok (Slice.from values h)
  else .error "slice exceeds usize"

def cells (json : Json) : Except String (Slice Cell) := do
  let rows ← json.getArr?
  slice (← rows.toList.mapM cell)

def family (destination : Key) : WriteFamily :=
  if destination.space = 3#u8 then .tmemStore
  else if destination.space = 4#u8 then .localStore else .store

def wordUpdate (json : Json) : Except String WordUpdate := do
  return ⟨← key (← jsonField json "key"), ← word (← jsonField json "history")⟩

def marker (json : Json) : Except String Key := do
  return ⟨1#u8, ← u64 (← jsonAt json 0), ← u64 (← jsonAt json 1)⟩

def markers (json : Json) : Except String WrittenMarkers := do
  return ⟨← (← (← jsonField json "numeric_written").getArr?).toList.mapM marker,
    ← (← (← jsonField json "half_written").getArr?).toList.mapM marker⟩

structure State where
  machine : MemoryMachine
  markers : WrittenMarkers

def event (tracking : WrittenTracking) (before : State) (json : Json) : Except String State := do
  let kind ← (← jsonField json "kind").getStr?
  let destination ← key (← jsonField json "key")
  if kind == "numeric_forget" then
    let storage := before.machine.storage.setWord ⟨destination, none⟩
    return ⟨{ before.machine with storage }, before.markers.recordNumeric tracking destination⟩
  if kind != "write" then throw s!"unsupported production event {kind}"
  let payload ← slice (← (← (← jsonField json "payload").getArr?).toList.mapM byte)
  let words ← (← (jsonOptional json "word_updates" (toJson ([] : List Json))).getArr?).toList.mapM wordUpdate
  let effect : WriteEffect := { family := family destination, destination, payload, words }
  match applyEffect before.machine.storage effect with
  | .error error => throw s!"write rejected: {repr error}"
  | .ok storage => return ⟨{ before.machine with storage }, before.markers.record tracking effect⟩

def sameMarkers (actual expected : List Key) : Bool :=
  actual.all (fun marker => marker ∈ expected) && expected.all (fun marker => marker ∈ actual)

def checkCase (json : Json) : Except String String := do
  let caseName ← (← jsonField json "name").getStr?
  let initial ← cells (← jsonField json "cells")
  let storage ← match Storage.fromCells initial with
    | none => throw s!"{caseName}: duplicate source keys"
    | some storage => pure storage
  let beforeMarkers ← markers (← jsonField json "markers_before")
  let initialState : State := ⟨{ storage }, beforeMarkers⟩
  let tracking : WrittenTracking := {
    numericEnabled := ← (jsonOptional json "numeric_tracking" (toJson true)).getBool?
    halfEnabled := ← (jsonOptional json "half_tracking" (toJson true)).getBool? }
  let final ← (← (← jsonField json "events").getArr?).toList.foldlM (event tracking) initialState
  let expected ← cells (← jsonField json "expected")
  unless final.machine.storage.toCells.val = expected.val do
    let mismatch := (List.zip final.machine.storage.toCells.val expected.val).findIdx?
      (fun pair => pair.1 != pair.2)
    throw s!"{caseName}: byte/history mismatch jsonAt row {mismatch}"
  let expectedMarkers ← markers (← jsonField json "markers_after")
  unless sameMarkers final.markers.numericWritten expectedMarkers.numericWritten do
    throw s!"{caseName}: numeric written markers differ"
  unless sameMarkers final.markers.halfWritten expectedMarkers.halfWritten do
    throw s!"{caseName}: half written markers differ"
  return caseName

end MemoryWrite.Replay

def main (args : List String) : IO Unit := do
  let [path] := args | throw (IO.userError "usage: lake env lean --run Replay.lean CORPUS.json")
  let input ← IO.FS.readFile path
  let result := do
    let json ← Json.parse input
    let cases ← (← json.getObjVal? "cases").getArr?
    cases.toList.mapM MemoryWrite.Replay.checkCase
  match result with
  | .error error => throw (IO.userError error)
  | .ok cases => IO.println s!"Lean production replay: {cases.length} passed"
