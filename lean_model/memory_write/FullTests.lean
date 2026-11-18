/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite

open Aeneas Aeneas.Std PtxvMemoryWrite MemoryWrite

namespace MemoryWrite.FullTests

def expect (caseName : String) (condition : Bool) : IO Unit :=
  unless condition do throw (IO.userError caseName)

def get {α : Type} (caseName : String) (result : Except WriteError α) : IO α :=
  match result with
  | .ok value => pure value
  | .error error => throw (IO.userError s!"{caseName}: {repr error}")

def globalKey (offset : U64) : Key := ⟨1#u8, 0#u64, offset⟩

def initial : Storage :=
  Storage.allocation (globalKey 0#u64)
    (Slice.from (List.ofFn fun index : Fin 32 =>
      U8.ofNatCore index.val (by scalar_tac)) (by simp; scalar_tac)) true true

def bytes (value : U8) (width : Usize) : Slice PayloadByte :=
  Slice.from (List.replicate width.val ⟨value, none, false, true, none⟩) (by simp; scalar_tac)

def effect (destination : Key) (value : U8) (width : Usize) : WriteEffect :=
  { family := .store, destination, payload := bytes value width }

def layout : MemoryLayout :=
  { sharedBytes := fun _ => 32
    globalBytes := fun _ => 32
    paramBytes := fun _ => 32
    localBytes := fun _ => 32
    tmemAllocated := fun _ _ => false }

def atByte (state : Storage) (offset : U64) : U8 := (state.getCell (globalKey offset)).value

def checkCopies : IO Unit := do
  let captured ← get "capture cp.async"
    (Transfers.captureCpAsync initial (globalKey 8#u64) (some (globalKey 0#u64)) 8#usize)
  let changed ← get "overwrite source" (applyEffect initial (effect (globalKey 0#u64) 77#u8 8#usize))
  let completed ← get "complete captured" (Transfers.completeCaptured changed captured)
  expect "captured copy read completion memory" (atByte completed 8#u64 == 0#u8)
  expect "capture lost taint" (completed.getCell (globalKey 8#u64)).tainted
  let deferred ← get "deferred copy"
    (Transfers.bulkCopy changed (globalKey 8#u64) (globalKey 0#u64) 8#usize)
  expect "deferred copy read issue memory" (atByte deferred 8#u64 == 77#u8)
  let overlap ← get "overlap" (Transfers.bulkCopy initial (globalKey 2#u64) (globalKey 0#u64) 8#usize)
  expect "overlap did not snapshot source" (atByte overlap 8#u64 == 6#u8)
  let chunks : List Transfers.Chunk :=
    [⟨globalKey 4#u64, some (globalKey 0#u64), 4#usize⟩,
     ⟨globalKey 8#u64, some (globalKey 4#u64), 4#usize⟩]
  let scatter ← get "capture scatter" (Transfers.prepareScatter initial .bulkCopy chunks)
  let scattered ← get "scatter" (Transfers.completeCaptured initial scatter)
  expect "scatter read an earlier destination" (atByte scattered 8#u64 == 4#u8)
  let zero ← get "zero capture" (Transfers.captureCpAsync initial (globalKey 8#u64) none 4#usize)
  let zeroed ← get "zero completion" (Transfers.completeCaptured initial zero)
  expect "zero fill value" (atByte zeroed 9#u64 == 0#u8)
  expect "zero fill taint" (!(zeroed.getCell (globalKey 9#u64)).tainted)
  expect "zero fill origin" ((zeroed.getCell (globalKey 9#u64)).origin == some .Zero)

def checkAtomics : IO Unit := do
  let replacement := bytes 55#u8 4#usize
  let exchange ← get "exchange" (Transfers.exchange initial (globalKey 0#u64) replacement)
  expect "exchange lost old bytes" (exchange.old.val.map PayloadByte.value == [0#u8, 1#u8, 2#u8, 3#u8])
  expect "exchange replacement" (atByte exchange.storage 1#u64 == 55#u8)
  let equal := Slice.from [0#u8, 1#u8, 2#u8, 3#u8] (by simp; scalar_tac)
  let unequal := Slice.from [9#u8, 9#u8, 9#u8, 9#u8] (by simp; scalar_tac)
  let success ← get "CAS success" (Transfers.compareExchange initial (globalKey 0#u64) equal replacement)
  expect "CAS success branch" success.exchanged
  expect "CAS success bytes" (atByte success.storage 0#u64 == 55#u8)
  let failure ← get "CAS failure" (Transfers.compareExchange initial (globalKey 0#u64) unequal replacement)
  expect "CAS failure branch" (!failure.exchanged)
  expect "CAS failure changed bytes" (atByte failure.storage 1#u64 == 1#u8)
  let withHistory := initial.setWord ⟨globalKey 0#u64, some (.Known 7#u32)⟩
  let failedHistory ← get "CAS failure history"
    (Transfers.compareExchange withHistory (globalKey 0#u64) unequal replacement)
  expect "failed CAS retained numeric history"
    ((failedHistory.storage.getCell (globalKey 0#u64)).word == none)
  expect "failed CAS changed byte written marker"
    (!(failedHistory.storage.getCell (globalKey 0#u64)).written)
  match Transfers.compareExchange initial (globalKey 0#u64) equal replacement true with
  | .error .inputDependentGuard => pure ()
  | _ => throw (IO.userError "dependent CAS guard accepted")

def checkBounds : IO Unit := do
  let write := effect (globalKey 0#u64) 3#u8 4#usize
  let checked : CheckedWrite := ⟨.global 0#u64 4#u64, write⟩
  let normalized ← get "normalized destination" (applyChecked layout initial checked)
  expect "raw key escaped location normalization" (atByte normalized 0#u64 == 0#u8)
  expect "normalized destination" (atByte normalized 4#u64 == 3#u8)
  match applyChecked layout initial ⟨.param 0#u64 0#u64, write⟩ with
  | .error _ => pure ()
  | _ => throw (IO.userError "parameter write accepted")
  match applyChecked layout initial ⟨.global 0#u64 31#u64, write⟩ with
  | .error .outOfBounds => pure ()
  | _ => throw (IO.userError "out of bounds accepted")
  match applyChecked layout initial
      ⟨.tmem 0#u64 ⟨0, by omega⟩ ⟨0, by omega⟩ ⟨0, by omega⟩,
       { write with family := .tmemStore }⟩ with
  | .error .outOfBounds => pure ()
  | _ => throw (IO.userError "unallocated TMEM accepted")
  let localWrite : CheckedWrite :=
    ⟨.local 7#u64 4#u64, { write with family := .localStore }⟩
  let localState ← get "local sparse extension" (applyChecked layout Storage.empty localWrite)
  expect "local write did not allocate bytes"
    ((localState.getCell ⟨4#u8, 7#u64, 4#u64⟩).value == 3#u8)
  match applyEffect initial { write with addressDependent := true } with
  | .error .inputDependentAddress => pure ()
  | _ => throw (IO.userError "input dependent address accepted")

def checkHistoriesAndLifecycle : IO Unit := do
  let known := initial.setWord ⟨globalKey 0#u64, some (.Known 9#u32)⟩
  let narrow ← get "narrow clobber" (applyEffect known (effect (globalKey 2#u64) 44#u8 2#usize))
  expect "partial write kept known history" ((narrow.getCell (globalKey 0#u64)).word == none)
  expect "partial write lost written marker" (narrow.getCell (globalKey 2#u64)).written
  let opaqueState ← get "opaque partial history"
    (applyEffect known { (effect (globalKey 2#u64) 44#u8 2#usize) with
      words := [⟨globalKey 0#u64, some (.Opaque 11#u32)⟩] })
  expect "opaque history was treated as absence"
    ((opaqueState.getCell (globalKey 0#u64)).word == some (.Opaque 11#u32))
  let markers := WrittenMarkers.record {} {} (effect (globalKey 2#u64) 44#u8 4#usize)
  expect "numeric marker missed upper word" (globalKey 4#u64 ∈ markers.numericWritten)
  expect "half marker missed byte" (globalKey 5#u64 ∈ markers.halfWritten)
  let disabled := WrittenMarkers.record {} ⟨false, false⟩ (effect (globalKey 2#u64) 44#u8 4#usize)
  expect "disabled tracking wrote markers" (disabled.numericWritten.isEmpty && disabled.halfWritten.isEmpty)
  let freed := known.removeSpan (globalKey 0#u64) 4#usize
  expect "free kept source readable" ((freed.readSpan (globalKey 0#u64) 1#usize).isNone)
  expect "free kept origin/history" ((freed.getCell (globalKey 0#u64)).word == none)
  let some reallocated := freed.registerWrite (globalKey 0#u64) (bytes 66#u8 4#usize)
    | throw (IO.userError "reallocation failed")
  expect "reallocation reused bytes" (atByte reallocated 0#u64 == 66#u8)
  expect "reallocation reused history" ((reallocated.getCell (globalKey 0#u64)).word == none)

def checkTmem : IO Unit := do
  let lane : Fin 128 := ⟨1, by omega⟩
  let column : Fin 512 := ⟨2, by omega⟩
  let loc := MemoryLocation.tmem 0#u64 lane column 0
  let (allocated, state) ← get "allocate TMEM"
    (allocateTmem layout Storage.empty 0#u64 lane column 4#usize)
  let cells ← get "read allocated TMEM" (readChecked allocated state loc 4#usize)
  expect "allocated TMEM was defined" (cells.val[0]?.map Cell.defined == some false)
  let written ← get "write TMEM" (applyChecked allocated state ⟨loc,
    { family := .tmemStore, destination := globalKey 0#u64, payload := bytes 21#u8 4#usize }⟩)
  expect "TMEM word-to-byte mapping" ((written.getCell loc.key).value == 21#u8)
  let poisoned ← get "poison TMEM" (poisonChecked allocated written loc 4#usize)
  expect "poison retained definedness" (!(poisoned.getCell loc.key).defined)
  expect "poison changed physical TMEM bytes" ((poisoned.getCell loc.key).value == 21#u8)
  let freed := freeTmemSpan allocated poisoned 0#u64 lane column 4#usize
  expect "free erased physical TMEM backing" ((freed.2.getCell loc.key).value == 21#u8)
  match readChecked freed.1 freed.2 loc 4#usize with
  | .error .unreadableSource => pure ()
  | _ => throw (IO.userError "freed TMEM remained readable")
  let otherLane : Fin 128 := ⟨2, by omega⟩
  let (otherLayout, otherState) ← get "allocate another TMEM span"
    (allocateTmem freed.1 freed.2 0#u64 otherLane column 4#usize)
  let (reallocatedLayout, reallocated) ← get "reallocate TMEM backing"
    (allocateTmem otherLayout otherState 0#u64 lane column 4#usize)
  let restored ← get "read reallocated TMEM"
    (readChecked reallocatedLayout reallocated loc 4#usize)
  expect "reallocation lost retained physical bytes" (restored.val.all fun cell => cell.value == 21#u8)
  expect "reallocation revived definedness" (restored.val.all fun cell => !cell.defined)
  expect "reallocation revived provenance/history"
    (restored.val.all fun cell => cell.origin == none && cell.word == none)

def checkCentralState : IO Unit := do
  let before : CentralState := { storage := initial }
  let after ← get "central write"
    (MemoryWrite.write layout before ⟨.global 0#u64 2#u64, effect (globalKey 0#u64) 8#u8 4#usize⟩)
  expect "central byte write" (atByte after.storage 2#u64 == 8#u8)
  expect "central numeric marker" (globalKey 4#u64 ∈ after.markers.numericWritten)
  expect "central half marker" (globalKey 5#u64 ∈ after.markers.halfWritten)
  let expected := Slice.from [9#u8, 9#u8, 9#u8, 9#u8] (by simp; scalar_tac)
  let failed ← get "central failed CAS"
    (MemoryWrite.compareExchange layout before (.global 0#u64 0#u64) expected (bytes 22#u8 4#usize))
  expect "central failed CAS branch" (!failed.exchanged)
  expect "failed CAS missing numeric marker" (globalKey 0#u64 ∈ failed.state.markers.numericWritten)
  expect "failed CAS wrote half markers" failed.state.markers.halfWritten.isEmpty

def checkAdapter : IO Unit := do
  let raw : Adapter.RawPlanes := {
    inventory := initial.keys, bytes := initial.bytes, initialBytes := initial.bytes
    halfInputKinds := [(0#u64, .bf16)], f32Inputs := [0#u64]
    initialWordNodes := [(globalKey 0#u64, 9#u32)] }
  expect "half initial fallback"
    (raw.effectiveOrigin (globalKey 3#u64) == some (.E 0#u64 1#u64 1#u8))
  let halfWritten := { raw with halfWritten := [globalKey 3#u64] }
  expect "half written marker failed to block fallback"
    (halfWritten.effectiveOrigin (globalKey 3#u64) == none)
  let changed := { raw with bytes := (globalKey 3#u64, 99#u8) :: raw.bytes }
  expect "half fallback ignored capture mismatch" (changed.effectiveOrigin (globalKey 3#u64) == none)
  expect "numeric initial producer fallback"
    (raw.effectiveWord (globalKey 0#u64) == some (.Known 9#u32))
  let numericWritten := { raw with numericWritten := [globalKey 0#u64] }
  expect "numeric marker failed to block fallback" (numericWritten.effectiveWord (globalKey 0#u64) == none)
  expect "numeric marker incorrectly blocked half fallback"
    (numericWritten.effectiveOrigin (globalKey 0#u64) == some (.E 0#u64 0#u64 0#u8))
  expect "adapter lost effective origin"
    ((raw.toStorage.getCell (globalKey 3#u64)).origin == raw.effectiveOrigin (globalKey 3#u64))
  let start : Key := ⟨3#u8, 0#u64, 8#u64⟩
  let wordKey : Key := ⟨3#u8, 0#u64, 2#u64⟩
  let tmem : Adapter.RawPlanes := {
    inventory := Slice.from (Storage.spanKeys start 4#usize) (by simp; scalar_tac)
    tmemWords := [(wordKey, 0x04030201#u32)], tmemInit := [(wordKey, 1#u8)]
    numeric := [(wordKey, .Known 7#u32)], taint := [(wordKey, true)] }
  expect "TMEM word byte normalization" (tmem.effectiveByte start == 1#u8)
  expect "TMEM numeric word normalization" (tmem.effectiveWord start == some (.Known 7#u32))
  expect "TMEM interior byte received word history"
    (tmem.effectiveWord (Storage.offsetKey start 1) == none)
  expect "TMEM word taint did not expand" (tmem.effectiveTaint (Storage.offsetKey start 3))
  let undefinedTmem := { tmem with tmemInit := [(wordKey, 0#u8)] }
  expect "TMEM poison exposed history" (undefinedTmem.effectiveWord start == none)
  expect "TMEM poison erased physical bytes" (undefinedTmem.effectiveByte start == 1#u8)
  let freed := { tmem with tmemInit := [(wordKey, 2#u8)] }
  expect "unallocated TMEM remained live" freed.liveKeys.isEmpty
  expect "unallocated TMEM lost backing" (Storage.lookup freed.toStorage.bytes start 0#u8 == 1#u8)
  let localState : Adapter.RawPlanes := {
    inventory := Slice.from [] (by simp)
    localPackedWords := [(⟨4#u8, 0#u64, 0#u64⟩, 0x104030201#u64)] }
  expect "local packed byte" (localState.effectiveByte ⟨4#u8, 0#u64, 1#u64⟩ == 2#u8)
  expect "local undefined mask byte" (!(localState.effectiveDefined ⟨4#u8, 0#u64, 0#u64⟩))
  expect "local undefined mask spread" (localState.effectiveDefined ⟨4#u8, 0#u64, 1#u64⟩)

def checkPending : IO Unit := do
  let machine : MemoryMachine := { storage := initial }
  let (issued, ticket) ← get "issue copy"
    (issueCopy machine .bulkCopy (globalKey 8#u64) (globalKey 0#u64) 4#usize)
  let changed ← get "pending source overwrite" (applyEffect issued.storage (effect (globalKey 0#u64) 99#u8 4#usize))
  let completed ← get "complete ticket" (complete { issued with storage := changed } ticket)
  expect "pending ticket used live source" (atByte completed.storage 8#u64 == 0#u8)
  expect "completed ticket retained" completed.pending.isEmpty
  match complete completed ticket with
  | .error .missingPending => pure ()
  | _ => throw (IO.userError "ticket completed twice")

def checkCentralAsync : IO Unit := do
  let before : CentralState := { storage := initial }
  let transfers : List CheckedTransfer :=
    [⟨.bulkCopy, .global 0#u64 8#u64, some (.global 0#u64 0#u64), 4#usize⟩]
  let (issued, ticket) ← get "central capture" (issueTransfer layout before transfers)
  expect "central issue wrote markers"
    (issued.markers.numericWritten.isEmpty && issued.markers.halfWritten.isEmpty)
  let changed ← get "central overwrite captured source"
    (MemoryWrite.write layout issued
      ⟨.global 0#u64 0#u64, effect (globalKey 0#u64) 99#u8 4#usize⟩)
  let completed ← get "central complete" (finishChecked layout changed ticket)
  expect "central completion reread source" (atByte completed.storage 8#u64 == 0#u8)
  expect "central completion missed numeric marker" (globalKey 8#u64 ∈ completed.markers.numericWritten)
  expect "central completion missed half marker" (globalKey 11#u64 ∈ completed.markers.halfWritten)
  expect "central completion kept ticket" completed.pending.isEmpty
  let restricted := { layout with globalBytes := fun _ => 10 }
  match finishChecked restricted changed ticket with
  | .error .outOfBounds => pure ()
  | _ => throw (IO.userError "central completion ignored current destination capacity")
  let (deferred, deferredTicket) := issueDeferredTransfer before transfers
  let changedDeferred ← get "central overwrite deferred source"
    (MemoryWrite.write layout deferred
      ⟨.global 0#u64 0#u64, effect (globalKey 0#u64) 77#u8 4#usize⟩)
  let completedDeferred ← get "central complete deferred"
    (finishChecked layout changedDeferred deferredTicket)
  expect "deferred completion used issue source" (atByte completedDeferred.storage 8#u64 == 77#u8)
  expect "deferred completion missed numeric marker"
    (globalKey 8#u64 ∈ completedDeferred.markers.numericWritten)
  expect "deferred completion missed half marker"
    (globalKey 11#u64 ∈ completedDeferred.markers.halfWritten)

#eval do
  checkCopies
  checkAtomics
  checkBounds
  checkHistoriesAndLifecycle
  checkPending
  checkTmem
  checkCentralState
  checkCentralAsync
  checkAdapter
  IO.println "Lean full-module regressions: 9 groups passed"

end MemoryWrite.FullTests
