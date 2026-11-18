/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite

open Aeneas Aeneas.Std PtxvMemoryWrite MemoryWrite

namespace MemoryWrite.RustTests

def expect (caseName : String) (condition : Bool) : IO Unit :=
  unless condition do throw (IO.userError caseName)

def returned {α : Type} (caseName : String) (result : Result α) : IO α :=
  match result.match with
  | .ok value => pure value
  | _ => throw (IO.userError s!"{caseName}: extracted Rust failed or diverged")

def key (offset : U64) : Key := ⟨1#u8, 5#u64, offset⟩

def cells : alloc.vec.Vec Cell :=
  alloc.vec.Vec.from (List.ofFn fun index : Fin 12 =>
    { key := key (U64.ofNatCore index.val (by scalar_tac))
      value := U8.ofNatCore index.val (by scalar_tac)
      origin := some (.N 7#u32 (U8.ofNatCore (index.val % 4) (by change index.val % 4 < 256; omega)))
      tainted := true, defined := index.val != 3, written := false
      word := some (.Known 7#u32) }) (by simp; scalar_tac)

def checkEngine : IO Unit := do
  let captured ← returned "capture" (checked.capture_span cells.slice (key 0#u64) 6#usize)
  let some payload := captured | throw (IO.userError "capture rejected live source")
  let write : checked.PreparedWrite :=
    ⟨.Store, key 3#u64, payload, alloc.vec.Vec.new _, true, false, false⟩
  for numeric in [false, true] do
    for half in [false, true] do
      let before : engine.Engine :=
        ⟨cells, alloc.vec.Vec.new _, alloc.vec.Vec.new _, numeric, half⟩
      let (status, after) ← returned "engine apply" (engine.apply before write)
      match status with
      | .Err _ => throw (IO.userError "valid effect rejected")
      | .Ok _ => pure ()
      expect "numeric tracker independence"
        (after.numeric_written.val.map Key.offset ==
          if numeric then [0#u64, 4#u64, 4#u64, 4#u64, 4#u64, 8#u64] else [])
      expect "half tracker independence"
        (after.half_written.val.map Key.offset ==
          if half then [3#u64, 4#u64, 5#u64, 6#u64, 7#u64, 8#u64] else [])
      let copied ← returned "read destination"
        (checked.capture_span after.cells.slice (key 3#u64) 6#usize)
      let some bytes := copied | throw (IO.userError "destination disappeared")
      expect "captured bits" (bytes.val.map PayloadByte.value == [0#u8, 1#u8, 2#u8, 3#u8, 4#u8, 5#u8])
      expect "captured definedness" (bytes.val.map PayloadByte.defined == [true, true, true, false, true, true])
      expect "captured taint" (bytes.val.all PayloadByte.tainted)
      expect "partial history invalidation" ((bytes.val[3]?.map PayloadByte.word) == some none)
      let (rejected, unchanged) ← returned "dependent guard"
        (engine.apply after { write with guard_dependent := true })
      match rejected with
      | .Err .InputDependentGuard => pure ()
      | _ => throw (IO.userError "dependent guard accepted")
      expect "rejection cell frame" (unchanged.cells.val == after.cells.val)
      expect "rejection numeric frame" (unchanged.numeric_written.val == after.numeric_written.val)
      expect "rejection half frame" (unchanged.half_written.val == after.half_written.val)

def main : IO Unit := do
  checkEngine
  let wrapping ← returned "wrapping source"
    (checked.capture_span cells.slice (key 18446744073709551615#u64) 2#usize)
  expect "wrapping source accepted" wrapping.isNone
  IO.println "Rust extraction tests: capture, checked effects, four tracker configurations, rejection frames passed"

#eval main

end MemoryWrite.RustTests
