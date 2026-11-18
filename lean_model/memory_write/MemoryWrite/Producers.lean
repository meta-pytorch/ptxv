/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Theorem

open Aeneas Aeneas.Std PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.Producers

def literalValues {Input : Type} (data : Slice U8) : Input → Nat → Byte :=
  fun _ index => ⟨data.val[index]?.getD 0#u8, true⟩

def literalPayload (data : Slice U8) : Slice PayloadByte :=
  Slice.from (data.val.map fun value =>
    ⟨value, if value = 0#u8 then some .Zero else none, false, true, none⟩) (by simp)

@[simp]
theorem literalPayload_length (data : Slice U8) : (literalPayload data).val.length = data.val.length := by
  simp [literalPayload]

theorem literalPayload_sound {Input : Type} (sem : Semantics Input) (data : Slice U8) :
    PayloadSound sem (literalPayload data) (literalValues data) := by
  constructor
  · intro index hindex
    have hi : index < data.val.length := by simpa [literalPayload] using hindex
    simp only [payloadAt, literalPayload, Slice.from_val, List.getElem?_map,
      List.getElem?_eq_getElem hi, Option.map_some, Option.getD_some]
    constructor
    · simp [literalValues, List.getElem?_eq_getElem hi]
    · simp [literalValues]
    · intro id hid input hinput
      split at hid
      · cases hid
        simp [literalValues, sem.zero_origin input hinput, *]
      · contradiction
    · simp [literalValues]
  · intro index hindex id hid
    have hi : index < data.val.length := by simp [literalPayload] at hindex; omega
    simp [payloadAt, literalPayload, List.getElem?_eq_getElem hi] at hid

def unknownDefinedPayload {Input : Type} (sem : Semantics Input) (width : Usize)
    (values : Input → Nat → Byte) (defined : Nat → Bool) : Slice PayloadByte :=
  Slice.from (List.ofFn fun index : Fin width.val =>
    ⟨(values sem.capture index.val).bits, none, true, defined index.val, none⟩)
    (by simp; scalar_tac)

theorem unknownDefinedPayload_sound {Input : Type} (sem : Semantics Input) (width : Usize)
    (values : Input → Nat → Byte) (defined : Nat → Bool)
    (hdefined : ∀ index, index < width.val → defined index = true →
      ∀ input, sem.domain input → (values input index).defined = true) :
    PayloadSound sem (unknownDefinedPayload sem width values defined) values := by
  constructor
  · intro index hindex
    have hi : index < width.val := by simpa [unknownDefinedPayload] using hindex
    simp only [payloadAt, unknownDefinedPayload, Slice.from_val, List.getElem?_ofFn,
      dite_eq_left hi, Option.getD_some]
    exact ⟨rfl, hdefined index hi, by simp, by simp⟩
  · intro index hindex id hid
    have hi : index < width.val := by simp [unknownDefinedPayload] at hindex; omega
    simp [payloadAt, unknownDefinedPayload, hi] at hid

def unknownPayload {Input : Type} (sem : Semantics Input) (width : Usize)
    (values : Input → Nat → Byte) : Slice PayloadByte :=
  unknownDefinedPayload sem width values (fun _ => false)

theorem unknownPayload_sound {Input : Type} (sem : Semantics Input) (width : Usize)
    (values : Input → Nat → Byte) : PayloadSound sem (unknownPayload sem width values) values := by
  exact unknownDefinedPayload_sound sem width values (fun _ => false) (by simp)

def takePayload (payload : Slice PayloadByte) (width : Usize) : Slice PayloadByte :=
  Slice.from (payload.val.take width.val) (by
    have := payload.property
    simp only [List.length_take]
    omega)

theorem payloadAt_takePayload (payload : Slice PayloadByte) (width : Usize) (index : Nat)
    (hindex : index < width.val) :
    payloadAt (takePayload payload width) index = payloadAt payload index := by
  simp [payloadAt, takePayload, hindex]

theorem takePayload_sound {Input : Type} (sem : Semantics Input) (payload : Slice PayloadByte)
    (values : Input → Nat → Byte) (width : Usize) (hpayload : PayloadSound sem payload values) :
    PayloadSound sem (takePayload payload width) values := by
  constructor
  · intro index hindex
    have hi : index < payload.val.length := by simp [takePayload] at hindex; omega
    have hw : index < width.val := by simp [takePayload] at hindex; omega
    rw [payloadAt_takePayload payload width index hw]
    exact hpayload.bytes index hi
  · intro index hindex id hid input hinput lane
    have hi : index + 4 ≤ payload.val.length := by simp [takePayload] at hindex; omega
    have hw : index < width.val := by simp [takePayload] at hindex; omega
    rw [payloadAt_takePayload payload width index hw] at hid
    exact hpayload.words index hi id hid input hinput lane

structure RegisterByte where
  value : U8
  origin : Option Origin
  tainted : Bool
  defined : Bool

def RegisterByte.toPayload (byte : RegisterByte) (history : Option WordHistory) : PayloadByte :=
  ⟨byte.value, byte.origin, byte.tainted, byte.defined, history⟩

structure RegisterWord where
  bytes : Fin 4 → RegisterByte
  history : Option WordHistory

structure RegisterWordSound {Input : Type} (sem : Semantics Input) (register : RegisterWord)
    (values : Input → Fin 4 → Byte) : Prop where
  bytes : ∀ lane, DatumSound sem (fun input => values input lane)
    (register.bytes lane).value (register.bytes lane).origin
    (register.bytes lane).tainted (register.bytes lane).defined
  history : ∀ id, register.history = some (.Known id) → ∀ input, sem.domain input →
    ∀ lane, (values input lane).defined = true ∧
      (values input lane).bits = sem.word id input lane

def registerPayload (file : Nat → RegisterWord) (registers : List Nat)
    (hsize : 4 * registers.length ≤ Usize.max) : Slice PayloadByte :=
  Slice.from (List.ofFn fun index : Fin (4 * registers.length) =>
    let register := file (registers[index.val / 4]?.getD 0)
    let lane : Fin 4 := ⟨index.val % 4, by omega⟩
    (register.bytes lane).toPayload (if index.val % 4 = 0 then register.history else none))
    (by simpa using hsize)

def registerValues {Input : Type} (values : Nat → Input → Fin 4 → Byte)
    (registers : List Nat) : Input → Nat → Byte :=
  fun input index => values (registers[index / 4]?.getD 0) input ⟨index % 4, by omega⟩

@[simp]
theorem registerPayload_length (file : Nat → RegisterWord) (registers : List Nat)
    (hsize : 4 * registers.length ≤ Usize.max) :
    (registerPayload file registers hsize).val.length = 4 * registers.length := by
  simp [registerPayload]

theorem registerPayload_at (file : Nat → RegisterWord) (registers : List Nat)
    (hsize : 4 * registers.length ≤ Usize.max) (index : Nat)
    (hindex : index < 4 * registers.length) :
    payloadAt (registerPayload file registers hsize) index =
      ((file (registers[index / 4]?.getD 0)).bytes ⟨index % 4, by omega⟩).toPayload
        (if index % 4 = 0 then (file (registers[index / 4]?.getD 0)).history else none) := by
  simp [payloadAt, registerPayload, hindex]

theorem registerPayload_sound {Input : Type} (sem : Semantics Input)
    (file : Nat → RegisterWord) (values : Nat → Input → Fin 4 → Byte) (registers : List Nat)
    (hsize : 4 * registers.length ≤ Usize.max)
    (hfile : ∀ register ∈ registers, RegisterWordSound sem (file register) (values register)) :
    PayloadSound sem (registerPayload file registers hsize) (registerValues values registers) := by
  constructor
  · intro index hindex
    have hi : index < 4 * registers.length := by simpa [registerPayload] using hindex
    have hr : index / 4 < registers.length := by omega
    rw [registerPayload_at file registers hsize index hi]
    have hregister := hfile registers[index / 4] (List.getElem_mem hr)
    simpa [registerValues, List.getElem?_eq_getElem hr, RegisterByte.toPayload] using
      hregister.bytes ⟨index % 4, by omega⟩
  · intro index hindex id hid input hinput lane
    have hi : index < 4 * registers.length := by simp [registerPayload] at hindex; omega
    have hr : index / 4 < registers.length := by omega
    rw [registerPayload_at file registers hsize index hi] at hid
    have haligned : index % 4 = 0 := by
      by_contra h
      simp [RegisterByte.toPayload, h] at hid
    have hhistory : (file registers[index / 4]).history = some (.Known id) := by
      simpa [RegisterByte.toPayload, haligned, List.getElem?_eq_getElem hr] using hid
    have hregister := hfile registers[index / 4] (List.getElem_mem hr)
    have hword := hregister.history id hhistory input hinput lane
    have hdiv : (index + lane.val) / 4 = index / 4 := by omega
    have hmod : (index + lane.val) % 4 = lane.val := by omega
    simpa [registerValues, hdiv, hmod, List.getElem?_eq_getElem hr] using hword

theorem narrowRegisterPayload_sound {Input : Type} (sem : Semantics Input)
    (file : Nat → RegisterWord) (values : Nat → Input → Fin 4 → Byte) (registers : List Nat)
    (hsize : 4 * registers.length ≤ Usize.max) (width : Usize)
    (hfile : ∀ register ∈ registers, RegisterWordSound sem (file register) (values register)) :
    PayloadSound sem (takePayload (registerPayload file registers hsize) width)
      (registerValues values registers) :=
  takePayload_sound sem _ _ width (registerPayload_sound sem file values registers hsize hfile)

def captureAndClobber (file : Nat → RegisterWord) (registers : List Nat)
    (hsize : 4 * registers.length ≤ Usize.max) (destination : Nat) (replacement : RegisterWord) :
    Slice PayloadByte × (Nat → RegisterWord) :=
  (registerPayload file registers hsize,
    fun register => if register = destination then replacement else file register)

theorem captureAndClobber_sound {Input : Type} (sem : Semantics Input)
    (file : Nat → RegisterWord) (values : Nat → Input → Fin 4 → Byte) (registers : List Nat)
    (hsize : 4 * registers.length ≤ Usize.max) (destination : Nat) (replacement : RegisterWord)
    (hfile : ∀ register ∈ registers, RegisterWordSound sem (file register) (values register)) :
    let result := captureAndClobber file registers hsize destination replacement
    PayloadSound sem result.1 (registerValues values registers) ∧
      result.2 destination = replacement ∧
      ∀ register, register ≠ destination → result.2 register = file register := by
  exact ⟨registerPayload_sound sem file values registers hsize hfile,
    by simp [captureAndClobber], by
      intro register hne
      simp [captureAndClobber, hne]⟩

theorem literal_write_preserves_representation {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (state : Slice Cell) (destination : Key) (data : Slice U8)
    (hstate : Represents sem memory state) (hspan : ValidWriteSpan state destination data.val.length) :
    ∃ next, apply_write_effect state destination (literalPayload data) = .ok next ∧
      Represents sem (storeBytes memory destination data.val.length (literalValues data)) next := by
  obtain ⟨next, hrun, _, hrep, _, _⟩ := apply_write_effect_preserves_representation sem memory
    state destination (literalPayload data) (literalValues data) hstate
    (by simpa using hspan) (literalPayload_sound sem data)
  exact ⟨next, hrun, by simpa using hrep⟩

theorem captured_register_write_preserves_representation {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (state : Slice Cell) (destination : Key)
    (file : Nat → RegisterWord) (values : Nat → Input → Fin 4 → Byte) (registers : List Nat)
    (hsize : 4 * registers.length ≤ Usize.max) (clobbered : Nat) (replacement : RegisterWord)
    (hstate : Represents sem memory state)
    (hspan : ValidWriteSpan state destination (4 * registers.length))
    (hfile : ∀ register ∈ registers, RegisterWordSound sem (file register) (values register)) :
    let result := captureAndClobber file registers hsize clobbered replacement
    ∃ next, apply_write_effect state destination result.1 = .ok next ∧
      Represents sem (storeBytes memory destination (4 * registers.length)
        (registerValues values registers)) next ∧ result.2 clobbered = replacement := by
  have hcaptured := captureAndClobber_sound sem file values registers hsize clobbered replacement hfile
  obtain ⟨next, hrun, _, hrep, _, _⟩ := apply_write_effect_preserves_representation sem memory
    state destination (registerPayload file registers hsize) (registerValues values registers)
    hstate (by simpa using hspan) hcaptured.1
  exact ⟨next, hrun, by simpa using hrep, hcaptured.2.1⟩

/-- A scheduler must justify these facts; dependency flags alone do not establish them. -/
structure StableStoreExecution {Input : Type} (sem : Semantics Input)
    (guard : Input → Bool) (address : Input → Key) : Prop where
  guard_stable : ∀ input, sem.domain input → guard input = guard sem.capture
  address_stable : guard sem.capture = true → ∀ input, sem.domain input →
    address input = address sem.capture

def dynamicStore {Input : Type} (memory : Memory Input) (guard : Input → Bool)
    (address : Input → Key) (width : Nat) (values : Input → Nat → Byte) : Memory Input :=
  fun input key => if guard input then storeBytes memory (address input) width values input key
    else memory input key

theorem dynamicStore_captured_execution {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (guard : Input → Bool) (address : Input → Key)
    (width : Nat) (values : Input → Nat → Byte)
    (hstable : StableStoreExecution sem guard address) (input : Input) (hinput : sem.domain input)
    (key : Address) :
    dynamicStore memory guard address width values input key =
      if guard sem.capture then storeBytes memory (address sem.capture) width values input key
        else memory input key := by
  cases hguard : guard sem.capture with
  | false => simp [dynamicStore, hstable.guard_stable input hinput, hguard]
  | true => simp [dynamicStore, hstable.guard_stable input hinput, hguard,
      hstable.address_stable hguard input hinput]

end MemoryWrite.Producers
