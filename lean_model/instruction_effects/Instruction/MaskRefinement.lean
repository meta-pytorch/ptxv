/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.Refinement
import Mathlib.Tactic.IntervalCases

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 8000000
set_option maxRecDepth 4096
namespace Instruction

def bitSum (mask : U8) : Nat :=
  (mask &&& 1#u8 != 0#u8).toNat + (mask &&& 2#u8 != 0#u8).toNat +
    (mask &&& 4#u8 != 0#u8).toNat + (mask &&& 8#u8 != 0#u8).toNat

theorem bitSum_low (mask : U8) : bitSum mask = (halfWords mask 0).length := by
  rcases mask with ⟨⟨value, hvalue⟩⟩
  change value < 256 at hvalue
  have hcanonical : ({ bv := { toFin := ⟨value, hvalue⟩ } } : U8) = ⟨BitVec.ofNat 8 value⟩ := by
    apply UScalar.eq_of_val_eq
    simp [UScalar.val, Nat.mod_eq_of_lt hvalue]
  rw [hcanonical]
  interval_cases value <;> decide

theorem bitSum_high (mask : U8) :
    bitSum ⟨mask.bv >>> 4⟩ = (halfWords mask 4).length := by
  rcases mask with ⟨⟨value, hvalue⟩⟩
  change value < 256 at hvalue
  have hcanonical : ({ bv := { toFin := ⟨value, hvalue⟩ } } : U8) = ⟨BitVec.ofNat 8 value⟩ := by
    apply UScalar.eq_of_val_eq
    simp [UScalar.val, Nat.mod_eq_of_lt hvalue]
  rw [hcanonical]
  interval_cases value <;> decide

@[simp] theorem bool_word_val (bit : Bool) :
    (UScalar.cast_fromBool .U16 bit).val = bit.toNat := by cases bit <;> rfl

theorem bool_count_bound (bit : Bool) : bit.toNat ≤ 1 := by cases bit <;> decide

@[step] theorem selected_half_spec (mask : U8) :
    register_effect.selected_half mask ⦃ result => result.val = bitSum mask ∧ result.val ≤ 4 ⦄ := by
  have h0 := bool_count_bound (mask &&& 1#u8 != 0#u8)
  have h1 := bool_count_bound (mask &&& 2#u8 != 0#u8)
  have h2 := bool_count_bound (mask &&& 4#u8 != 0#u8)
  have h3 := bool_count_bound (mask &&& 8#u8 != 0#u8)
  unfold register_effect.selected_half
  simp only [lift, bind_ok]
  step as ⟨first, hfirst⟩
  · simp only [bool_word_val]; scalar_tac
  step as ⟨second, hsecond⟩
  · simp only [bool_word_val] at hfirst ⊢; scalar_tac
  step as ⟨result, hresult⟩
  · simp only [bool_word_val] at hfirst hsecond ⊢; scalar_tac
  simp only [bool_word_val] at hfirst hsecond hresult
  constructor
  · simp only [bitSum]; omega
  · omega

@[step] theorem selected_halves_spec (mask : U8) :
    register_effect.selected_halves mask ⦃ result =>
      result.1.val = (halfWords mask 0).length ∧
      result.2.val = (halfWords mask 4).length ∧
      result.1.val ≤ 4 ∧ result.2.val ≤ 4 ⦄ := by
  unfold register_effect.selected_halves
  step as ⟨low, hlow, hlowbound⟩
  step as ⟨shifted, hshiftval, hshiftbv⟩
  have heq : shifted = ⟨mask.bv >>> 4⟩ := congrArg UScalar.mk hshiftbv
  step as ⟨high, hhigh, hhighbound⟩
  exact ⟨by simpa only [bitSum_low] using hlow,
    by simpa only [heq, bitSum_high] using hhigh, hlowbound, hhighbound⟩

@[step] theorem masked_writes_spec (hi lo : Reg) (address : Addr)
    (sync : Option MemSync) (mask : U8) (nc : Bool)
    (hindex : Bounded (maskedWrites lo hi mask)) :
    register_effect.memory_writes (.Ld256 hi lo address sync mask nc)
      ⦃ result => ∃ registers, result = some registers ∧
        registers.val.map key = maskedWrites lo hi mask ⦄ := by
  have hparts : Bounded (consecutive lo (halfWords mask 0).length) ∧
      Bounded (consecutive hi (halfWords mask 4).length) := by
    simpa [maskedWrites] using hindex
  simp only [register_effect.memory_writes]
  step as ⟨low, high, hlow, hhigh, hlowbound, hhighbound⟩
  step with append_spec (alloc.vec.Vec.new Reg) lo low
    (by rw [hlow]; exact consecutive_bounded lo _ hparts.1)
    (by simp only [empty_register_vector]; scalar_tac) as ⟨middle, hmmap, hmlen⟩
  simp only [empty_register_vector, List.length_nil, Nat.zero_add] at hmlen
  step with append_spec middle hi high
    (by rw [hhigh]; exact consecutive_bounded hi _ hparts.2)
    (by scalar_tac) as ⟨registers, hmap, hlength⟩
  refine ⟨registers, rfl, ?_⟩
  simpa [hmmap, empty_register_vector, hlow, hhigh, maskedWrites] using hmap

end Instruction
