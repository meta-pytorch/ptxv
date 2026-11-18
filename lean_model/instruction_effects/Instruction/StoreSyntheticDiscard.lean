/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.StoreWindowRefinement
import Instruction.ReadRefinement

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction

theorem ptx_memory_reads_preserved (operation : Op) :
    register_effect.memory_reads_for_front .Ptx operation =
      register_effect.memory_reads operation := by
  cases operation <;> try rfl
  case St space bytes address value sync =>
    cases value <;> cases space <;>
      simp [register_effect.memory_reads_for_front, register_effect.repeated_virtual,
        Front.Insts.CoreCmpPartialEqFront.eq, Front.read_discriminant]

@[step] theorem synthetic_sass_discard_reads_total (index : U16) (address : Addr)
    (sync : Option MemSync) (haddress : Bounded (addressReads address)) :
    register_effect.memory_reads_for_front .Sass
      (.St .Global 128#u8 address (.Reg ⟨.V, index⟩ 1#u8) sync) ⦃ result =>
      registerObservation result = some (addressReads address ++ [key ⟨.V, index⟩]) ⦄ := by
  simp only [register_effect.memory_reads_for_front]
  step with repeated_virtual_spec .Sass .V 1#u8 128#u8 as ⟨repeated, hrepeated⟩
  have hr : repeated = true := by simpa using hrepeated
  simp only [hr, ite_true, lift, bind_ok]
  step with address_spec (alloc.vec.Vec.new Reg) address haddress
    (by have h := addressReads_length address
        simp only [empty_register_vector]
        scalar_tac) as ⟨sources, hsources, hlength⟩
  step with alloc.vec.Vec.push_spec sources (⟨.V, index⟩ : Reg)
    (by have h := addressReads_length address
        simp only [empty_register_vector, List.length_nil, Nat.zero_add] at hlength
        scalar_tac) as ⟨pushed, hpushed⟩
  simp [registerObservation, hpushed, List.map_append, hsources, empty_register_vector]

theorem synthetic_sass_discard_windows_total (index : U16) (address : Addr) :
    store_window.sources .Sass
      (.St .Global 128#u8 address (.Reg ⟨.V, index⟩ 1#u8) none) ⦃ result =>
      result.val.map byteWindow = repeatedWindows ⟨.V, index⟩ 16 128 ∧
      result.val.length = 16 ⦄ := by
  simp [store_window.sources, store_window.descriptor,
    File.Insts.CoreCmpPartialEqFile.eq, File.read_discriminant]
  step with repeated_virtual_spec .Sass .V 1#u8 128#u8 as ⟨repeated, hrepeated⟩
  have hr : repeated = true := by simpa using hrepeated
  simp only [hr, ite_true]
  step with repeated_windows_spec (⟨.V, index⟩ : Reg) 128#u8 as ⟨windows, hwindows⟩
  have hmodel : windows.val.map byteWindow = repeatedWindows ⟨.V, index⟩ 16 128 := by
    simpa using hwindows
  refine ⟨hmodel, ?_⟩
  have hlength := congrArg List.length hmodel
  simpa [repeatedWindows] using hlength

theorem synthetic_sass_discard_sources_and_windows (index : U16) (address : Addr)
    (haddress : Bounded (addressReads address)) :
    (register_effect.memory_reads_for_front .Sass
      (.St .Global 128#u8 address (.Reg ⟨.V, index⟩ 1#u8) none) ⦃ result =>
      registerObservation result = some (addressReads address ++ [key ⟨.V, index⟩]) ⦄) ∧
    (store_window.sources .Sass
      (.St .Global 128#u8 address (.Reg ⟨.V, index⟩ 1#u8) none) ⦃ result =>
      result.val.map byteWindow = repeatedWindows ⟨.V, index⟩ 16 128 ∧
      result.val.length = 16 ⦄) := by
  exact ⟨synthetic_sass_discard_reads_total index address none haddress,
    synthetic_sass_discard_windows_total index address⟩


end Instruction
