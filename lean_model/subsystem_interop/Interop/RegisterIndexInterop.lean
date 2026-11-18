/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.WriteRefinement
import Instruction.OwnerRefinement
import MemoryWrite.RegisterIndexPath

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace Instruction

abbrev ProgramKey := PtxvMemoryWrite.symbolic.RegisterKey

def programKey (register : Reg) : ProgramKey :=
  ⟨0#u32, IScalar.hcast .U8 (read_discriminant register.file), register.idx⟩

def programKeyView (key : ProgramKey) : RegisterOwner.Key :=
  ⟨key.thread.val, key.file.val, key.index.val⟩

def logicalProgramKey (key : Key) : RegisterOwner.Key :=
  ⟨0, RegisterOwner.fileTag key.file, key.index⟩

def programKeyFromSource (register : Reg) : Result ProgramKey := do
  let (owner, file, index) ← register_owner.canonical_key 0#usize (Slice.from [] (by simp)) register
  PtxvMemoryWrite.symbolic.register_key owner file index

theorem program_key_view (register : Reg) :
    programKeyView (programKey register) = logicalProgramKey (key register) := by
  simp only [programKeyView, programKey, logicalProgramKey, key,
    RegisterOwner.file_tag_discriminant]
  rfl

theorem program_key_injective (left right : Reg) (equal : programKey left = programKey right) : left = right := by
  have fileEqual : RegisterOwner.fileTag left.file = RegisterOwner.fileTag right.file := by
    have same := congrArg (fun value => value.file.val) equal
    simpa only [programKey, RegisterOwner.file_tag_discriminant] using same
  have indexEqual : left.idx = right.idx := congrArg (fun value => value.index) equal
  have files := (RegisterOwner.file_tag_injective left.file right.file).mp fileEqual
  cases left
  cases right
  simp only [Reg.mk.injEq]
  exact ⟨files, indexEqual⟩

@[step]
theorem rust_program_key (register : Reg) :
    programKeyFromSource register ⦃ result => result = programKey register ⦄ := by
  unfold programKeyFromSource
  step with RegisterOwner.rust_canonical_key 0#usize (Slice.from [] (by simp)) register
    as ⟨owner, file, index, observed⟩
  simp only [RegisterOwner.keyView, RegisterOwner.canonicalKey, Slice.from_val,
    List.map_nil, List.getElem?_nil, Option.getD_none, ite_self] at observed
  have ownerValue : owner.val = 0 := by
    have same := congrArg RegisterOwner.Key.owner observed
    simpa only [ite_self, Nat.zero_mod] using same
  have fileValue : file.val = RegisterOwner.fileTag register.file :=
    congrArg RegisterOwner.Key.file observed
  have indexValue : index.val = register.idx.val :=
    congrArg RegisterOwner.Key.index observed
  simp only [PtxvMemoryWrite.symbolic.register_key, WP.spec_ok, programKey,
    PtxvMemoryWrite.symbolic.RegisterKey.mk.injEq]
  exact ⟨UScalar.eq_of_val_eq ownerValue, UScalar.eq_of_val_eq
    (fileValue.trans (RegisterOwner.file_tag_discriminant register.file).symm),
    UScalar.eq_of_val_eq indexValue⟩

def programObservation (registers : Option (alloc.vec.Vec Reg)) : Option (List RegisterOwner.Key) :=
  registers.map fun values => values.val.map (fun register => programKeyView (programKey register))

theorem program_observation_refines_register_observation (registers : Option (alloc.vec.Vec Reg)) :
    programObservation registers = (registerObservation registers).map (List.map logicalProgramKey) := by
  cases registers with
  | none => rfl
  | some values =>
      simp only [programObservation, registerObservation, Option.map_some, List.map_map]
      congr 1
      exact List.map_congr_left (fun register _ => program_key_view register)

@[step]
theorem memory_program_destinations_spec (operation : Op)
    (bounded : optionBounded (memoryWrites operation)) :
    register_effect.memory_writes operation ⦃ result =>
      programObservation result = (memoryWrites operation).map (List.map logicalProgramKey) ⦄ := by
  step with memory_writes_spec operation bounded as ⟨result, meaning⟩
  rw [program_observation_refines_register_observation, meaning]

end Instruction
