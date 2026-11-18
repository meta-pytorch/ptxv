/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.MaskRefinement
import Instruction.AtomicRefinement
import Instruction.ControlDataRefinement

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction

def atomicCount (operation : AtomOp) (file : File) (bytes : U8) : Nat :=
  if operation = .CasSpin then 1
  else if file = .V ∧ ¬(operation = .FAddFtz ∧ (bytes.val = 8 ∨ bytes.val = 16))
    then 1 else max (bytes.val / 4) 1

@[step] theorem atom_count_spec (operation : AtomOp) (file : File) (bytes : U8) :
    register_effect.atom_count operation file bytes ⦃ result =>
      result.val = atomicCount operation file bytes ∧ result.val ≤ 255 ⦄ := by
  have hb8val : bytes.val = 8 ↔ bytes = 8#u8 := by scalar_tac
  have hb16val : bytes.val = 16 ↔ bytes = 16#u8 := by scalar_tac
  cases operation <;> cases file <;>
    by_cases hb8 : bytes = 8#u8 <;> by_cases hb16 : bytes = 16#u8
  all_goals simp [register_effect.atom_count, AtomOp.Insts.CoreCmpPartialEqAtomOp.eq,
    AtomOp.read_discriminant, File.Insts.CoreCmpPartialEqFile.eq, File.read_discriminant,
    hb8, hb16, atomicCount, hb8val, hb16val]
  all_goals exact word_count_spec _

@[step] theorem atomic_writes_spec (operation : AtomOp) (space : Space) (bytes : U8)
    (address : Addr) (value : Val) (destination : Option Reg) (sync : MemSync)
    (predicate : Option Reg) (replacement : Option Val)
    (hindex : Bounded (atomicWrites operation bytes destination predicate)) :
    register_effect.memory_writes
      (.Atom operation space bytes address value destination sync predicate replacement)
      ⦃ result => ∃ registers, result = some registers ∧
        registers.val.map key = atomicWrites operation bytes destination predicate ⦄ := by
  cases destination with
  | none =>
    simp only [register_effect.memory_writes, bind_ok]
    cases predicate with
    | none => simp [atomicWrites, empty_register_vector]
    | some predicate =>
      step with alloc.vec.Vec.push_spec (alloc.vec.Vec.new Reg) predicate
        (by simp only [empty_register_vector]; scalar_tac) as ⟨registers, hregisters⟩
      exact ⟨registers, rfl, by simp [hregisters, atomicWrites, empty_register_vector]⟩
  | some destination =>
    have hmeaning : atomicWrites operation bytes (some destination) predicate =
        consecutive destination (atomicCount operation destination.file bytes) ++
          predicate.toList.map key := by simp [atomicWrites, atomicCount]
    have hparts : Bounded (consecutive destination (atomicCount operation destination.file bytes)) ∧
        Bounded (predicate.toList.map key) := by simpa only [hmeaning, bounded_append] using hindex
    simp only [register_effect.memory_writes]
    step as ⟨count, hcount, hcountbound⟩
    step with append_spec (alloc.vec.Vec.new Reg) destination count
      (by rw [hcount]; exact consecutive_bounded destination _ hparts.1)
      (by simp only [empty_register_vector]; scalar_tac) as ⟨middle, hmmap, hmlen⟩
    simp only [empty_register_vector, List.length_nil, Nat.zero_add] at hmlen
    cases predicate with
    | none =>
      simp [atomicWrites, atomicCount, hmmap, hcount, empty_register_vector]
    | some predicate =>
      step with alloc.vec.Vec.push_spec middle predicate (by scalar_tac)
        as ⟨registers, hregisters⟩
      refine ⟨registers, rfl, ?_⟩
      simp [hregisters, hmmap, empty_register_vector, hmeaning, hcount]

@[step] theorem memory_writes_spec (operation : Op)
    (hindex : optionBounded (memoryWrites operation)) :
    register_effect.memory_writes operation
      ⦃ result => registerObservation result = memoryWrites operation ⦄ := by
  cases operation <;> try (solve | simp [register_effect.memory_writes, memoryWrites,
    registerObservation, empty_register_vector])
  case Ld destination elements space bytes address sync signed nc =>
    step with load_writes_spec destination elements bytes space address sync signed nc hindex
      as ⟨result, registers, heq, hmeaning⟩
    simp [registerObservation, heq, memoryWrites, hmeaning]
  case Ld256 hi lo address sync mask nc =>
    step with masked_writes_spec hi lo address sync mask nc hindex
      as ⟨result, registers, heq, hmeaning⟩
    simp [registerObservation, heq, memoryWrites, hmeaning]
  case Atom operation space bytes address value destination sync predicate replacement =>
    step with atomic_writes_spec operation space bytes address value destination sync predicate replacement hindex
      as ⟨result, registers, heq, hmeaning⟩
    simp [registerObservation, heq, memoryWrites, hmeaning]

end Instruction
