/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.Code.Types

open Aeneas Aeneas.Std PtxvIr

set_option autoImplicit false

namespace Instruction

deriving instance DecidableEq for File
deriving instance DecidableEq for Front
deriving instance DecidableEq for AtomOp

structure Key where
  file : File
  index : Nat
  deriving DecidableEq

def key (reg : Reg) : Key := ⟨reg.file, reg.idx.val⟩

def consecutive (first : Reg) (count : Nat) : List Key :=
  (List.range count).map fun index => ⟨first.file, first.idx.val + index⟩

@[simp] theorem consecutive_length (first : Reg) (count : Nat) :
    (consecutive first count).length = count := by simp [consecutive]

theorem consecutive_succ (first : Reg) (count : Nat) :
    consecutive first (count + 1) = consecutive first count ++
      [⟨first.file, first.idx.val + count⟩] := by
  simp [consecutive, List.range_succ, List.map_append]

theorem consecutive_mem (first : Reg) (count : Nat) (reg : Key) :
    reg ∈ consecutive first count ↔
      reg.file = first.file ∧ first.idx.val ≤ reg.index ∧ reg.index < first.idx.val + count := by
  simp only [consecutive, List.mem_map, List.mem_range]
  constructor
  · rintro ⟨index, hi, rfl⟩
    simp only [true_and]
    omega
  · rintro ⟨hf, hlo, hhi⟩
    refine ⟨reg.index - first.idx.val, by omega, ?_⟩
    cases reg
    simp_all

def Bounded (registers : List Key) : Prop :=
  ∀ reg, reg ∈ registers → reg.index < 65536

@[simp] theorem bounded_append (left right : List Key) :
    Bounded (left ++ right) ↔ Bounded left ∧ Bounded right := by
  simp [Bounded, List.mem_append, or_imp, forall_and]

theorem consecutive_bounded (first : Reg) (count : Nat)
    (h : Bounded (consecutive first count)) : first.idx.val + count ≤ 65536 := by
  by_cases hc : count = 0
  · subst count
    scalar_tac
  · have hm : (⟨first.file, first.idx.val + (count - 1)⟩ : Key) ∈
        consecutive first count := by rw [consecutive_mem]; simp; omega
    have hb := h _ hm
    dsimp only at hb
    omega

/-- PTX vectors store one element per named virtual register; scalar b128 stores two
64-bit pieces. SASS load destinations store one 32-bit word per register. -/
def loadWrites (destination : Reg) (bytes elements : U8) : List Key :=
  consecutive destination <| if destination.file = File.V then
    if elements.val > 1 then elements.val else max (bytes.val / 8) 1
  else max (bytes.val / 4) 1

def scalarSlots (file : File) (count : Nat) : Nat :=
  if file = .V then 1 else min (max count 1) 2

def operandReads (value : Val) : List Key :=
  match value with
  | .Reg first slots => consecutive first (scalarSlots first.file slots.val)
  | _ => []

def vectorReads (value : Val) : List Key :=
  match value with
  | .Reg first elements => consecutive first
      (if first.file = .V then max elements.val 1 else scalarSlots first.file elements.val)
  | _ => []

def storeSlots (file : File) (bytes elements : Nat) (asynchronous : Bool) : Nat :=
  if file = .V ∧ elements ≥ 2 then elements
  else if asynchronous then
    if file = .V then 1 else if bytes > 4 then bytes / 4 else scalarSlots file elements
  else if bytes > 8 then bytes / 4 else scalarSlots file elements

def storeValueReads (asynchronous : Bool) (bytes : U8) (value : Val) : List Key :=
  match value with
  | .Reg first elements => consecutive first
      (storeSlots first.file bytes.val elements.val asynchronous)
  | _ => []

def addressReads (address : Addr) : List Key :=
  address.base.toList.flatMap operandReads ++ address.base2.toList.flatMap operandReads

/-- Actual read events of a successful CAS include the implicit replacement operand.
This is selected from the Op fields, independently of operand enumeration. -/
def casReplacement (bytes : U8) (compare : Val) (replacement : Option Val) : List Key :=
  match replacement with
  | some value => operandReads value
  | none => match compare with
    | .Reg first _ =>
      let words := if bytes.val = 8 ∧ first.file ≠ .V then 2 else 1
      (List.range words).map fun index => ⟨first.file, first.idx.val + words + index⟩
    | _ => []

/-- Masked 256-bit loads select memory words, then pack each selected half into
consecutive destination registers. The mask selects source positions, not destination holes. -/
def halfWords (mask : U8) (start : Nat) : List Nat :=
  (List.range 4).filter fun index => mask.bv.getLsbD (start + index)

def maskedWrites (lo hi : Reg) (mask : U8) : List Key :=
  consecutive lo (halfWords mask 0).length ++ consecutive hi (halfWords mask 4).length

def atomicValueReads (operation : AtomOp) (bytes : U8) (value : Val) : List Key :=
  match operation, value with
  | .Exch, _ => vectorReads value
  | .FAddFtz, .Reg first _ => if bytes.val > 4 then
      consecutive first (bytes.val / 4) else operandReads value
  | _, _ => operandReads value

def replacementReads (operation : AtomOp) (bytes : U8) (compare : Val)
    (replacement : Option Val) : List Key :=
  match replacement with
  | some value => operandReads value
  | none => match operation with
    | .Cas | .CasSpin => casReplacement bytes compare none
    | _ => []

def memoryReads (operation : Op) : Option (List Key) :=
  match operation with
  | .Ld _ _ _ _ address _ _ _ | .Ld256 _ _ address _ _ _ => some (addressReads address)
  | .St _ bytes address value _ => some (addressReads address ++ storeValueReads false bytes value)
  | .StAsync address barrier value bytes =>
      some (operandReads address ++ operandReads barrier ++ storeValueReads true bytes value)
  | .St256 lo hi address _ =>
      some (addressReads address ++ consecutive lo 4 ++ consecutive hi 4)
  | .Atom operation _ bytes address value _ _ _ replacement =>
      some (addressReads address ++ atomicValueReads operation bytes value ++
        replacementReads operation bytes value replacement)
  | _ => none

def atomicWrites (operation : AtomOp) (bytes : U8)
    (destination predicate : Option Reg) : List Key :=
  destination.toList.flatMap (fun first =>
    consecutive first <| if operation = .CasSpin then 1
      else if first.file = .V ∧ ¬(operation = .FAddFtz ∧ (bytes.val = 8 ∨ bytes.val = 16))
        then 1 else max (bytes.val / 4) 1) ++ predicate.toList.map key

def memoryWrites (operation : Op) : Option (List Key) :=
  match operation with
  | .Ld destination elements _ bytes _ _ _ _ => some (loadWrites destination bytes elements)
  | .St _ _ _ _ _ | .StAsync _ _ _ _ | .St256 _ _ _ _ => some []
  | .Atom operation _ bytes _ _ destination _ predicate _ =>
      some (atomicWrites operation bytes destination predicate)
  | .Ld256 hi lo _ _ mask _ => some (maskedWrites lo hi mask)
  | _ => none

/-- An enabled instruction writes exactly its semantic destinations. A disabled instruction
has no must-write destinations; its may-write set remains useful for dependent guards. -/
def mustWrite (enabled : Bool) (writes : List Key) : List Key :=
  if enabled then writes else []

theorem disabled_writes (writes : List Key) : mustWrite false writes = [] := rfl

/-- Completeness of an enumerated write set discharges the frame side of a whole-state
invariant. The transfer must still establish the invariant at every enumerated destination. -/
theorem state_sound_of_effect_complete {State Value : Type} (observe : State → Key → Value)
    (sound : Key → Value → Prop) (before after : State) (actual enumerated : List Key)
    (hbefore : ∀ reg, sound reg (observe before reg))
    (hcoverage : ∀ reg, reg ∈ actual → reg ∈ enumerated)
    (hframe : ∀ reg, reg ∉ actual → observe after reg = observe before reg)
    (htransfer : ∀ reg, reg ∈ enumerated → sound reg (observe after reg)) :
    ∀ reg, sound reg (observe after reg) := by
  intro reg
  by_cases hw : reg ∈ actual
  · exact htransfer reg (hcoverage reg hw)
  · rw [hframe reg hw]
    exact hbefore reg

/-- A source-dependent analysis can safely classify an instruction as constant only when
its source list covers every read event and each listed source is constant across inputs. -/
theorem reads_constant_of_effect_complete {Input Value : Type}
    (read : Input → Key → Value) (capture : Input) (actual enumerated : List Key)
    (hcoverage : ∀ reg, reg ∈ actual → reg ∈ enumerated)
    (hconstant : ∀ reg, reg ∈ enumerated → ∀ input, read input reg = read capture reg) :
    ∀ reg, reg ∈ actual → ∀ input, read input reg = read capture reg := by
  intro reg hreg input
  exact hconstant reg (hcoverage reg hreg) input

end Instruction
