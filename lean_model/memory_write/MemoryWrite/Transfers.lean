/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Effects

open Aeneas Aeneas.Std PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite

namespace Transfers

def zeroValues {Input : Type} : Input → Nat → Byte := fun _ _ => ⟨0#u8, true⟩

def zeroPayload (width : Usize) : Slice PayloadByte :=
  Slice.from (List.replicate width.val ⟨0#u8, some .Zero, false, true, none⟩) (by simp; scalar_tac)

@[simp] theorem zeroPayload_length (width : Usize) :
    (zeroPayload width).val.length = width.val := by simp [zeroPayload]

@[simp] theorem zeroPayload_at (width : Usize) (index : Nat) (hi : index < width.val) :
    payloadAt (zeroPayload width) index = ⟨0#u8, some .Zero, false, true, none⟩ := by
  simp [payloadAt, zeroPayload, hi]

theorem zeroPayload_sound {Input : Type} (sem : Semantics Input) (width : Usize) :
    PayloadSound sem (zeroPayload width) zeroValues := by
  have hbyte := zeroPayload_at width
  constructor
  · intro index hi
    rw [hbyte index (by simpa using hi)]
    constructor
    · rfl
    · simp [zeroValues]
    · intro id hid input hinput
      have hid' : id = .Zero := Option.some.inj hid.symm
      subst id
      exact ⟨rfl, (sem.zero_origin input hinput).symm⟩
    · simp [zeroValues]
  · intro index hi id hid
    rw [hbyte index (by have := zeroPayload_length width; omega)] at hid
    cases hid

structure Chunk where
  destination : Key
  source : Option Key
  width : Usize

structure CapturedChunk where
  effect : WriteEffect
  source : Option Key

def CapturedChunk.values {Input : Type} (memory : Memory Input) (chunk : CapturedChunk) :
    Input → Nat → Byte :=
  match chunk.source with
  | some source => snapshotValues memory source
  | none => zeroValues

def CapturedChunk.Sound {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (chunk : CapturedChunk) : Prop :=
  PayloadSound sem chunk.effect.payload (chunk.values memory) ∧ chunk.effect.words = []

def prepareChunk (state : Storage) (family : WriteFamily) (chunk : Chunk) :
    Except WriteError CapturedChunk :=
  match chunk.source with
  | none => .ok ⟨{ family := family, destination := chunk.destination, payload := zeroPayload chunk.width }, none⟩
  | some source => do
      let effect ← prepareCopy state family chunk.destination source chunk.width
      .ok ⟨effect, some source⟩

theorem prepareChunk_sound {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (state : Storage) (family : WriteFamily) (chunk : Chunk) (captured : CapturedChunk)
    (hstate : Storage.Represents sem memory state)
    (hrun : prepareChunk state family chunk = .ok captured) :
    captured.Sound sem memory ∧ captured.effect.destination = chunk.destination ∧
      captured.effect.payload.val.length = chunk.width.val ∧ captured.source = chunk.source := by
  cases hsource : chunk.source with
  | none =>
    simp only [prepareChunk, hsource, Except.ok.injEq] at hrun
    subst captured
    exact ⟨⟨zeroPayload_sound sem chunk.width, rfl⟩, rfl, zeroPayload_length _, rfl⟩
  | some source =>
    cases hcopy : prepareCopy state family chunk.destination source chunk.width with
    | error error => simp [Bind.bind, Except.bind, prepareChunk, hsource, hcopy] at hrun
    | ok effect =>
      have heq : captured = ⟨effect, some source⟩ := by
        simpa [Bind.bind, Except.bind, prepareChunk, hsource, hcopy] using hrun.symm
      subst captured
      have hs := prepareCopy_sound sem memory state family chunk.destination source chunk.width
        effect hstate hcopy
      have hwords : effect.words = [] := by
        unfold prepareCopy at hcopy
        cases hread : state.readSpan source chunk.width with
        | none => simp [hread] at hcopy
        | some cells => simp [hread] at hcopy; subst effect; rfl
      exact ⟨⟨hs.1.1, hwords⟩, hs.2.1, hs.2.2, rfl⟩

/-- Every source is read from the same storage, including overlapping scatter destinations. -/
def prepareScatter (state : Storage) (family : WriteFamily) :
    List Chunk → Except WriteError (List CapturedChunk)
  | [] => .ok []
  | chunk :: chunks => do
      let captured ← prepareChunk state family chunk
      let rest ← prepareScatter state family chunks
      .ok (captured :: rest)

theorem prepareScatter_sound {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (state : Storage) (family : WriteFamily) (chunks : List Chunk) (captured : List CapturedChunk)
    (hstate : Storage.Represents sem memory state)
    (hrun : prepareScatter state family chunks = .ok captured) :
    ∀ chunk ∈ captured, chunk.Sound sem memory := by
  induction chunks generalizing captured with
  | nil => simp [prepareScatter] at hrun; subst captured; simp
  | cons chunk chunks ih =>
    cases hfirst : prepareChunk state family chunk with
    | error error => simp [Bind.bind, Except.bind, prepareScatter, hfirst] at hrun
    | ok first =>
      cases hrest : prepareScatter state family chunks with
      | error error => simp [Bind.bind, Except.bind, prepareScatter, hfirst, hrest] at hrun
      | ok rest =>
        have heq : captured = first :: rest := by
          simpa [Bind.bind, Except.bind, prepareScatter, hfirst, hrest] using hrun.symm
        subst captured
        intro item hitem
        rcases List.mem_cons.mp hitem with heq | hrestmem
        · subst item
          exact (prepareChunk_sound sem memory state family chunk first hstate hfirst).1
        · exact ih rest hrest item hrestmem

def descriptions {Input : Type} (memory : Memory Input) (chunks : List CapturedChunk) :
    List (DescribedEffect Input) :=
  chunks.map fun chunk => ⟨chunk.effect, chunk.values memory⟩

theorem descriptions_sound {Input : Type} (sem : Semantics Input)
    (sourceMemory destinationMemory : Memory Input) (chunks : List CapturedChunk)
    (hchunks : ∀ chunk ∈ chunks, chunk.Sound sem sourceMemory) :
    EffectsSound sem destinationMemory (descriptions sourceMemory chunks) := by
  induction chunks generalizing destinationMemory with
  | nil => trivial
  | cons chunk chunks ih =>
    have hfirst := hchunks chunk (by simp)
    refine ⟨⟨hfirst.1, ?_⟩, ?_⟩
    · simp [hfirst.2]
    · exact ih _ (fun other hother => hchunks other (by simp [hother]))

def completeCaptured (state : Storage) (chunks : List CapturedChunk) : Except WriteError Storage :=
  runEffects state (chunks.map CapturedChunk.effect)

theorem completeCaptured_preserves_representation {Input : Type} (sem : Semantics Input)
    (sourceMemory destinationMemory : Memory Input) (sourceState before after : Storage)
    (family : WriteFamily) (chunks : List Chunk) (captured : List CapturedChunk)
    (hsource : Storage.Represents sem sourceMemory sourceState)
    (hprepare : prepareScatter sourceState family chunks = .ok captured)
    (hstate : Storage.Represents sem destinationMemory before)
    (hrun : completeCaptured before captured = .ok after) :
    Storage.Represents sem (effectsMemory destinationMemory (descriptions sourceMemory captured)) after ∧
      after.keys = before.keys := by
  apply runEffects_preserves_representation sem destinationMemory before after
    (descriptions sourceMemory captured) hstate
    (descriptions_sound sem sourceMemory destinationMemory captured
      (prepareScatter_sound sem sourceMemory sourceState family chunks captured hsource hprepare))
  simpa [completeCaptured, descriptions, List.map_map, Function.comp_def] using hrun

structure DeferredTransfer where
  family : WriteFamily
  chunks : List Chunk

def completeDeferred (state : Storage) (transfer : DeferredTransfer) : Except WriteError Storage := do
  let captured ← prepareScatter state transfer.family transfer.chunks
  completeCaptured state captured

theorem completeDeferred_preserves_representation {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (before after : Storage) (transfer : DeferredTransfer)
    (hstate : Storage.Represents sem memory before)
    (hrun : completeDeferred before transfer = .ok after) :
    ∃ captured, prepareScatter before transfer.family transfer.chunks = .ok captured ∧
      Storage.Represents sem (effectsMemory memory (descriptions memory captured)) after ∧
      after.keys = before.keys := by
  cases hprepare : prepareScatter before transfer.family transfer.chunks with
  | error error => simp [Bind.bind, Except.bind, completeDeferred, hprepare] at hrun
  | ok captured =>
    have hcomplete : completeCaptured before captured = .ok after := by
      simpa [Bind.bind, Except.bind, completeDeferred, hprepare] using hrun
    exact ⟨captured, rfl,
      completeCaptured_preserves_representation sem memory memory before before after
        transfer.family transfer.chunks captured hstate hprepare hstate hcomplete⟩

def bulkCopy (state : Storage) (destination source : Key) (width : Usize) :
    Except WriteError Storage :=
  completeDeferred state ⟨.bulkCopy, [⟨destination, some source, width⟩]⟩

def tmaLoad (state : Storage) (elements : List Chunk) : Except WriteError Storage :=
  completeDeferred state ⟨.tmaLoad, elements⟩

def tmaStore (state : Storage) (elements : List Chunk) : Except WriteError Storage :=
  completeDeferred state ⟨.tmaStore, elements⟩

def captureCpAsync (state : Storage) (destination : Key) (source : Option Key) (width : Usize) :
    Except WriteError (List CapturedChunk) :=
  prepareScatter state .bulkCopy [⟨destination, source, width⟩]


def captureCpAsyncSized (state : Storage) (destination source : Key)
    (sourceWidth totalWidth : Usize) : Except WriteError (List CapturedChunk) :=
  if sourceWidth.val ≤ totalWidth.val ∧ destination.offset.val + totalWidth.val ≤ U64.max + 1 then
    prepareScatter state .bulkCopy [
      ⟨destination, some source, sourceWidth⟩,
      ⟨Storage.offsetKey destination sourceWidth.val, none,
        Usize.ofNatCore (totalWidth.val - sourceWidth.val) (by
          have hmax := totalWidth.hmax
          omega)⟩]
  else .error .outOfBounds

def readPayload (state : Storage) (source : Key) (width : Usize) :
    Except WriteError (Slice PayloadByte) := do
  let effect ← prepareCopy state .bulkCopy source source width
  .ok effect.payload

theorem readPayload_sound {Input : Type} (sem : Semantics Input) (memory : Memory Input)
    (state : Storage) (source : Key) (width : Usize) (payload : Slice PayloadByte)
    (hstate : Storage.Represents sem memory state)
    (hrun : readPayload state source width = .ok payload) :
    PayloadSound sem payload (snapshotValues memory source) ∧ payload.val.length = width.val := by
  cases hcopy : prepareCopy state .bulkCopy source source width with
  | error error => simp [Bind.bind, Except.bind, readPayload, hcopy] at hrun
  | ok effect =>
    have heq : payload = effect.payload := by simpa [Bind.bind, Except.bind, readPayload, hcopy] using hrun.symm
    subst payload
    have hs := prepareCopy_sound sem memory state .bulkCopy source source width effect hstate hcopy
    exact ⟨hs.1.1, hs.2.2⟩

structure AtomicResult where
  storage : Storage
  old : Slice PayloadByte
  exchanged : Bool

def exchange (state : Storage) (destination : Key) (payload : Slice PayloadByte) :
    Except WriteError AtomicResult := do
  let old ← readPayload state destination (Slice.len payload)
  let next ← applyEffect state { family := .atomicExchange, destination := destination, payload := payload }
  .ok ⟨next, old, true⟩

theorem exchange_preserves_representation {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (state : Storage) (destination : Key) (payload : Slice PayloadByte)
    (values : Input → Nat → Byte) (result : AtomicResult)
    (hstate : Storage.Represents sem memory state) (hpayload : PayloadSound sem payload values)
    (hrun : exchange state destination payload = .ok result) :
    Storage.Represents sem (storeBytes memory destination payload.val.length values) result.storage ∧
      PayloadSound sem result.old (snapshotValues memory destination) ∧
      result.old.val.length = payload.val.length ∧ result.exchanged = true ∧
      result.storage.keys = state.keys := by
  cases hread : readPayload state destination (Slice.len payload) with
  | error error => simp [Bind.bind, Except.bind, exchange, hread] at hrun
  | ok old =>
    cases hwrite : applyEffect state { family := .atomicExchange, destination := destination, payload := payload } with
    | error error => simp [Bind.bind, Except.bind, exchange, hread, hwrite] at hrun
    | ok next =>
      have heq : result = ⟨next, old, true⟩ := by
        simpa [Bind.bind, Except.bind, exchange, hread, hwrite] using hrun.symm
      subst result
      have hr := readPayload_sound sem memory state destination (Slice.len payload) old hstate hread
      have hw := applyEffect_preserves_representation sem memory state next
        { family := .atomicExchange, destination := destination, payload := payload } values hstate ⟨hpayload, by simp⟩ hwrite
      exact ⟨by simpa [effectMemory] using hw.1, hr.1, by simpa using hr.2, rfl, hw.2⟩

def bytesEqual (old : Slice PayloadByte) (expected : Slice U8) : Bool :=
  old.val.map PayloadByte.value == expected.val

def failedCompareExchangeState (state : Storage) (destination : Key) : Storage :=
  state.setWord ⟨destination, none⟩


@[simp] theorem failedCompareExchangeState_bytes (state : Storage) (destination : Key) :
    (failedCompareExchangeState state destination).bytes = state.bytes := rfl

@[simp] theorem failedCompareExchangeState_origins (state : Storage) (destination : Key) :
    (failedCompareExchangeState state destination).origins = state.origins := rfl

@[simp] theorem failedCompareExchangeState_taint (state : Storage) (destination : Key) :
    (failedCompareExchangeState state destination).taint = state.taint := rfl

@[simp] theorem failedCompareExchangeState_defined (state : Storage) (destination : Key) :
    (failedCompareExchangeState state destination).defined = state.defined := rfl

@[simp] theorem failedCompareExchangeState_written (state : Storage) (destination : Key) :
    (failedCompareExchangeState state destination).written = state.written := rfl

def compareExchange (state : Storage) (destination : Key) (expected : Slice U8)
    (replacement : Slice PayloadByte) (compareDependent : Bool := false) :
    Except WriteError AtomicResult := do
  if compareDependent then .error .inputDependentGuard
  else
    let old ← readPayload state destination (Slice.len replacement)
    if bytesEqual old expected then
      let next ← applyEffect state { family := .atomicCompareExchange, destination := destination, payload := replacement }
      .ok ⟨next, old, true⟩
    else .ok ⟨failedCompareExchangeState state destination, old, false⟩

@[simp] theorem compareExchange_rejects_dependent (state : Storage) (destination : Key)
    (expected : Slice U8) (replacement : Slice PayloadByte) :
    compareExchange state destination expected replacement true = .error .inputDependentGuard := by
  simp [compareExchange]

theorem compareExchange_failed_effect (state : Storage) (destination : Key)
    (expected : Slice U8) (replacement : Slice PayloadByte) (old : Slice PayloadByte)
    (hread : readPayload state destination (Slice.len replacement) = .ok old)
    (hmiss : bytesEqual old expected = false) :
    compareExchange state destination expected replacement false = .ok ⟨failedCompareExchangeState state destination, old, false⟩ := by
  simp [Bind.bind, Except.bind, compareExchange, hread, hmiss]

def compareExchangeResultMemory {Input : Type} (memory : Memory Input) (destination : Key)
    (replacement : Slice PayloadByte) (values : Input → Nat → Byte) (exchanged : Bool) :
    Memory Input :=
  if exchanged then storeBytes memory destination replacement.val.length values else memory

theorem compareExchange_preserves_representation {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (state : Storage) (destination : Key) (expected : Slice U8)
    (replacement : Slice PayloadByte) (values : Input → Nat → Byte)
    (result : AtomicResult) (compareDependent : Bool)
    (hstate : Storage.Represents sem memory state)
    (hpayload : PayloadSound sem replacement values)
    (hrun : compareExchange state destination expected replacement compareDependent = .ok result) :
    Storage.Represents sem
      (compareExchangeResultMemory memory destination replacement values result.exchanged)
      result.storage ∧ PayloadSound sem result.old (snapshotValues memory destination) ∧
      result.old.val.length = replacement.val.length ∧ result.storage.keys = state.keys := by
  unfold compareExchange at hrun
  split at hrun
  · simp at hrun
  · cases hread : readPayload state destination (Slice.len replacement) with
    | error error => simp [Bind.bind, Except.bind, hread] at hrun
    | ok old =>
      have hr := readPayload_sound sem memory state destination (Slice.len replacement) old hstate hread
      by_cases hmatch : bytesEqual old expected = true
      · cases hwrite : applyEffect state { family := .atomicCompareExchange, destination := destination, payload := replacement } with
        | error error => simp [Bind.bind, Except.bind, hread, hmatch, hwrite] at hrun
        | ok next =>
          have heq : result = ⟨next, old, true⟩ := by
            simpa [Bind.bind, Except.bind, hread, hmatch, hwrite] using hrun.symm
          subst result
          have hw := applyEffect_preserves_representation sem memory state next
            { family := .atomicCompareExchange, destination := destination, payload := replacement } values hstate ⟨hpayload, by simp⟩ hwrite
          exact ⟨by simpa [compareExchangeResultMemory, effectMemory] using hw.1,
            hr.1, by simpa using hr.2, hw.2⟩
      · have heq : result = ⟨failedCompareExchangeState state destination, old, false⟩ := by
          simpa [Bind.bind, Except.bind, hread, hmatch] using hrun.symm
        subst result
        have hfailed : Storage.Represents sem memory (failedCompareExchangeState state destination) := by
          apply Storage.setWord_preserves_representation sem memory state ⟨destination, none⟩ hstate
          intro id hid
          cases hid
        exact ⟨hfailed, hr.1, by simpa using hr.2, rfl⟩

def memoryMatches {Input : Type} (memory : Memory Input) (input : Input)
    (destination : Key) (width : Nat) (expected : Slice U8) : Bool :=
  (List.ofFn fun index : Fin width => (memory input (location destination index.val)).bits) == expected.val

def compareExchangeMemory {Input : Type} (memory : Memory Input) (destination : Key)
    (replacement : Slice PayloadByte) (expected : Slice U8) (values : Input → Nat → Byte) :
    Memory Input := fun input address =>
  if memoryMatches memory input destination replacement.val.length expected
  then storeBytes memory destination replacement.val.length values input address
  else memory input address

/-- A fixed branch may represent CAS on all inputs only when its comparison is stable. -/
theorem compareExchange_memory_on_stable_inputs {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (destination : Key) (replacement : Slice PayloadByte)
    (expected : Slice U8) (values : Input → Nat → Byte) (result : AtomicResult)
    (hstable : ∀ input, sem.domain input →
      memoryMatches memory input destination replacement.val.length expected = result.exchanged)
    (input : Input) (hinput : sem.domain input) (address : Address) :
    compareExchangeMemory memory destination replacement expected values input address =
      compareExchangeResultMemory memory destination replacement values result.exchanged input address := by
  cases hbranch : result.exchanged <;> simp [compareExchangeMemory, compareExchangeResultMemory, hstable input hinput, hbranch]


theorem represents_congr {Input : Type} (sem : Semantics Input)
    (before after : Memory Input) (state : Storage)
    (hstate : Storage.Represents sem before state)
    (heq : ∀ input, sem.domain input → ∀ address, after input address = before input address) :
    Storage.Represents sem after state := by
  intro cell hcell
  have hbefore := hstate cell hcell
  constructor
  · constructor
    · simpa only [heq sem.capture sem.capture_valid] using hbefore.byte.captured
    · intro hdefined input hinput
      simpa only [heq input hinput] using hbefore.byte.defined_sound hdefined input hinput
    · intro id hid input hinput
      simpa only [heq input hinput] using hbefore.byte.origin_sound id hid input hinput
    · intro hclean input hinput
      simpa only [heq input hinput, heq sem.capture sem.capture_valid] using
        hbefore.byte.constant hclean input hinput
  · intro id hid input hinput lane
    simpa only [heq input hinput] using hbefore.word id hid input hinput lane
  · intro hnot input hinput
    simpa only [heq input hinput] using hbefore.initial_fallback hnot input hinput

theorem compareExchange_semantic_preservation {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (state : Storage) (destination : Key) (expected : Slice U8)
    (replacement : Slice PayloadByte) (values : Input → Nat → Byte)
    (result : AtomicResult) (compareDependent : Bool)
    (hstate : Storage.Represents sem memory state)
    (hpayload : PayloadSound sem replacement values)
    (hrun : compareExchange state destination expected replacement compareDependent = .ok result)
    (hstable : ∀ input, sem.domain input →
      memoryMatches memory input destination replacement.val.length expected = result.exchanged) :
    Storage.Represents sem (compareExchangeMemory memory destination replacement expected values)
      result.storage ∧ PayloadSound sem result.old (snapshotValues memory destination) ∧
      result.old.val.length = replacement.val.length ∧ result.storage.keys = state.keys := by
  have hresult := compareExchange_preserves_representation sem memory state destination expected
    replacement values result compareDependent hstate hpayload hrun
  refine ⟨represents_congr sem _ _ result.storage hresult.1 ?_, hresult.2⟩
  intro input hinput address
  exact compareExchange_memory_on_stable_inputs sem memory destination replacement expected values
    result hstable input hinput address

end Transfers

end MemoryWrite
