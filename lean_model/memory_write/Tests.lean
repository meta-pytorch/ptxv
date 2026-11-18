/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite

open Aeneas Aeneas.Std PtxvMemoryWrite MemoryWrite

def oldCell (offset : U64) : Cell :=
  ⟨⟨1#u8, 0#u64, offset⟩, 7#u8, some (.N 1#u32 0#u8), true, true, false, some (WordHistory.Known 2#u32)⟩

def unknownByte : PayloadByte := ⟨0#u8, none, true, true, none⟩

def payload (bytes : List PayloadByte) (h : bytes.length ≤ Usize.max) : Slice PayloadByte :=
  Slice.from bytes h

def expectCell (caseName : String) (before : Cell) (write : Key) (bytes : Slice PayloadByte)
    (expected : Cell) : IO Unit := do
  match (cell_after_write before write bytes).match with
  | .ok actual =>
    unless actual = expected do throw (IO.userError s!"{caseName}: unexpected byte/history effects")
  | _ => throw (IO.userError s!"{caseName}: Rust extraction did not return successfully")

#eval do
  let two := payload [unknownByte, unknownByte] (by simp; scalar_tac)
  let eight := payload (List.replicate 8 unknownByte) (by simp; scalar_tac)
  let empty := payload [] (by simp)
  let base := oldCell 0#u64
  expectCell "offset-two half write" base ⟨1#u8, 0#u64, 2#u64⟩ two
    { base with word := none }
  let upper := oldCell 4#u64
  expectCell "upper word of wide write" upper base.key eight
    { upper with
      value := 0#u8
      origin := none
      written := true
      word := none }
  expectCell "other allocation" base ⟨1#u8, 1#u64, 0#u64⟩ eight base
  expectCell "empty write" base base.key empty base
  IO.println "Lean extraction regressions: 4 passed"

def runExtracted {α : Type} (caseName : String) (program : Result α) : IO α := do
  match program.match with
  | .ok value => pure value
  | _ => throw (IO.userError s!"{caseName}: Rust extraction did not return successfully")

def expectCopiedBytes (caseName : String) (source destination : List Cell) : IO Unit := do
  let bytes := fun cells : List Cell => cells.map fun cell =>
    (cell.value, cell.origin, cell.tainted, cell.defined)
  unless bytes source = bytes destination do
    throw (IO.userError s!"{caseName}: copied bytes or metadata changed")

#eval do
  let state := Slice.from (List.ofFn fun index : Fin 12 =>
    { oldCell (U64.ofNatCore index.val (by scalar_tac)) with
      value := U8.ofNatCore index.val (by scalar_tac)
      origin := some (.E 0#u64 (U64.ofNatCore index.val (by scalar_tac)) 0#u8)
      defined := decide (index.val ≠ 5)
      word := some (WordHistory.Known (U32.ofNatCore index.val (by scalar_tac))) }) (by simp; scalar_tac)
  let source := Slice.from (state.val.take 8) (by simp [state]; scalar_tac)
  let frozen ← runExtracted "snapshot" (snapshot_cells source)
  unless frozen.val = source.val.map snapshotByte do
    throw (IO.userError "snapshot: metadata was not preserved")
  let write : Key := ⟨1#u8, 0#u64, 2#u64⟩
  let overlap ← runExtracted "overlap" (apply_copy_effect state write source)
  expectCopiedBytes "overlap" source.val ((overlap.val.drop 2).take 8)
  unless overlap.val[2]?.map Cell.word = some (some (WordHistory.Known 0#u32)) ∧
      overlap.val[6]?.map Cell.word = some (some (WordHistory.Known 4#u32)) ∧
      overlap.val[7]?.map Cell.word = some none do
    throw (IO.userError "overlap: incorrect complete or partial word histories")
  let unknown := payload (List.replicate 8 unknownByte) (by simp; scalar_tac)
  let changed ← runExtracted "source rewrite"
    (apply_write_effect state ⟨1#u8, 0#u64, 0#u64⟩ unknown)
  let completed ← runExtracted "delayed copy" (apply_write_effect changed write frozen.slice)
  expectCopiedBytes "delayed copy" source.val ((completed.val.drop 2).take 8)
  let empty := Slice.from ([] : List Cell) (by simp)
  let unchanged ← runExtracted "empty copy" (apply_copy_effect state write empty)
  unless unchanged.val = state.val do
    throw (IO.userError "empty copy changed memory")
  IO.println "Lean snapshot regressions: 4 passed"
