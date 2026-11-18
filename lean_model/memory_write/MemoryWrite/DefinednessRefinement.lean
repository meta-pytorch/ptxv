/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Definedness
import MemoryWrite.Code.Funs

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite.Definedness

theorem scalar_ne_zero (value : U64) :
    (value != 0#u64) = decide (value.val ≠ 0) := by
  have h : value = 0#u64 ↔ value.val = 0 := by
    constructor
    · intro hv
      exact congrArg UScalar.val hv
    · intro hv
      exact UScalar.eq_of_val_eq hv
  simp only [bne, beq_eq_decide, h, decide_not]

@[step]
theorem rust_selected_undefined_spec (mask : U64) (start width : U32) :
    definedness.selected_undefined mask start width
      ⦃ result => result = decide (selected mask.val start.val width.val ≠ 0) ⦄ := by
  unfold definedness.selected_undefined
  split
  · rename_i hs
    have hsval : 64 ≤ start.val := by scalar_tac
    have hz : mask.val >>> start.val = 0 := Nat.shiftRight_eq_zero _ _
      (lt_of_lt_of_le mask.bv.isLt (Nat.pow_le_pow_right Nat.zero_lt_two hsval))
    simp [selected, hz]
  · rename_i hs
    step as ⟨shifted, hshifted, hbv⟩; try scalar_tac
    split
    · rename_i hw
      have hwval : 64 ≤ width.val := by scalar_tac
      have hlt : mask.val >>> start.val < 2 ^ width.val :=
        lt_of_lt_of_le (lt_of_le_of_lt (Nat.shiftRight_le _ _) mask.bv.isLt)
          (Nat.pow_le_pow_right Nat.zero_lt_two hwval)
      simp [scalar_ne_zero, selected, hshifted, Nat.mod_eq_of_lt hlt]
    · rename_i hw
      have hwval : width.val < 64 := by scalar_tac
      step as ⟨power, hpower, hpowerbv⟩; try scalar_tac
      have hp : power.val = 2 ^ width.val := by
        simpa [Nat.shiftLeft_eq, U64.size, U64.numBits,
          Nat.mod_eq_of_lt (Nat.pow_lt_pow_right Nat.one_lt_two hwval)] using hpower
      have hpositive : 1 ≤ power.val := by
        rw [hp]
        have := Nat.two_pow_pos width.val
        omega
      step as ⟨low, hlow⟩; try scalar_tac
      have hl : low.val = 2 ^ width.val - 1 := by rw [hlow, hp]
      have hvalue : (shifted &&& low).val = selected mask.val start.val width.val := by
        rw [UScalar.val_and, hshifted, hl, Nat.and_two_pow_sub_one_eq_mod]
        rfl
      simp only [lift, bind_ok, WP.spec_ok]
      rw [scalar_ne_zero, hvalue]

theorem rust_selected_defined_iff (mask : U64) (start width : U32) :
    definedness.selected_undefined mask start width
      ⦃ result => result = false ↔ defined mask.val start.val width.val ⦄ := by
  step as ⟨result, hr⟩
  simp [hr, selected_zero_iff_defined]

theorem rust_conversion_of_defined_bits {Output : Type} (convert : Nat → Output)
    (captured actual : Nat) (mask : U64) (start width : U32)
    (hc : consistent captured mask.val actual) :
    definedness.selected_undefined mask start width ⦃ result => result = false →
      convert (selected actual start.val width.val) =
        convert (selected captured start.val width.val) ⦄ := by
  step with rust_selected_defined_iff mask start width as ⟨result, hr⟩
  exact conversion_of_defined_bits convert captured mask.val actual start.val width.val hc
    (hr.mp ‹result = false›)

@[step]
theorem rust_byte_mask_loop_spec (mask : U64) (count : U32) (result : U64) (byte : U32)
    (hb : byte.val ≤ 8) (hc : byte.val ≤ count.val)
    (hr : result.val = byteMask mask.val byte.val) :
    definedness.byte_mask_loop mask count result byte
      ⦃ output => output.val = byteMask mask.val (min count.val 8) ⦄ := by
  unfold definedness.byte_mask_loop
  apply loop.spec_decr_nat
    (fun current : U64 × U32 => 8 - current.2.val)
    (fun current : U64 × U32 => current.2.val ≤ 8 ∧ current.2.val ≤ count.val ∧
      current.1.val = byteMask mask.val current.2.val)
  · intro current hcurrent
    rcases current with ⟨result, byte⟩
    obtain ⟨hb, hc, hr⟩ := hcurrent
    unfold definedness.byte_mask_loop.body
    by_cases hcount : byte < count
    · simp only [hcount, ite_true]
      by_cases hbyte : byte < 8#u32
      · simp only [hbyte, ite_true]
        have hbyteval : byte.val < 8 := by scalar_tac
        have hcountval : byte.val < count.val := by scalar_tac
        step as ⟨offset, hoffset⟩; try scalar_tac
        have hoff : offset.val = 8 * byte.val := by scalar_tac
        step with rust_selected_undefined_spec mask offset 8#u32 as ⟨undefined, hundefined⟩
        have hu : undefined = byteUndefined mask.val byte.val := by
          simpa [byteUndefined, hoff] using hundefined
        have hupdate :
            (if undefined then do
              let power ← 1#u64 <<< byte
              ok (result ||| power)
            else ok result) ⦃ updated => updated.val = byteMask mask.val (byte.val + 1) ⦄ := by
          cases hundefined : undefined
          · have hfalse : byteUndefined mask.val byte.val = false := hu.symm.trans hundefined
            simp [byteMask, hfalse, hr]
          · have htrue : byteUndefined mask.val byte.val = true := hu.symm.trans hundefined
            simp only [ite_true]
            step as ⟨power, hpower, hpowerbv⟩; try scalar_tac
            have hp : power.val = 2 ^ byte.val := by
              simpa [Nat.shiftLeft_eq, U64.size, U64.numBits,
                Nat.mod_eq_of_lt (Nat.pow_lt_pow_right Nat.one_lt_two (by omega : byte.val < 64))]
                using hpower
            simp [UScalar.val_or, hr, hp, byteMask, htrue]
        step with hupdate as ⟨updated, hupdated⟩
        step as ⟨next, hnext⟩; try scalar_tac
      · simp only [hbyte, ite_false, WP.spec_ok]
        have hend : byte.val = min count.val 8 := by scalar_tac
        exact hr.trans (congrArg (byteMask mask.val) hend)
    · simp only [hcount, ite_false, WP.spec_ok]
      have hend : byte.val = min count.val 8 := by scalar_tac
      exact hr.trans (congrArg (byteMask mask.val) hend)
  · exact ⟨hb, hc, hr⟩

@[step]
theorem rust_byte_mask_spec (mask : U64) (count : U32) :
    definedness.byte_mask mask count
      ⦃ output => output.val = byteMask mask.val (min count.val 8) ⦄ := by
  unfold definedness.byte_mask
  step with rust_byte_mask_loop_spec mask count 0#u64 0#u32
    (by simp) (by simp) (by simp [byteMask])
  assumption

theorem rust_store_definedness_is_byte_local (mask : U64) (count : U32) :
    definedness.byte_mask mask count ⦃ output => ∀ byte,
      output.val.testBit byte =
        (decide (byte < count.val ∧ byte < 8) && byteUndefined mask.val byte) ⦄ := by
  step as ⟨output, hout⟩
  rename_i byte
  rw [hout, byte_mask_bit]
  simp

@[step]
theorem rust_merge_byte_mask_spec (previous source : U64) (first count : U32) :
    definedness.merge_byte_mask previous source first count
      ⦃ output => output.val = mergeByteMask previous.val source.val first.val count.val ⦄ := by
  unfold definedness.merge_byte_mask
  split
  · rename_i hfirst
    have hf : ¬first.val < 64 := by scalar_tac
    simp [mergeByteMask, hf]
  · rename_i hfirst
    have hf : first.val < 64 := by scalar_tac
    step with rust_byte_mask_spec source count as ⟨bytes, hbytes⟩
    step as ⟨shifted, hshifted, hbv⟩; try scalar_tac
    simp [mergeByteMask, hf, UScalar.val_or, hshifted, hbytes, U64.size, U64.numBits]

theorem rust_vector_store_definedness_is_local (previous source : U64) (first count : U32) :
    definedness.merge_byte_mask previous source first count ⦃ output => ∀ byte,
      output.val.testBit byte = (previous.val.testBit byte ||
        placedUndefined source.val first.val count.val byte) ⦄ := by
  step with rust_merge_byte_mask_spec previous source first count as ⟨output, hout⟩
  rename_i byte
  rw [hout, merge_byte_mask_bit]

@[step]
theorem rust_written_undefined_spec (source previous unordered : Bool) :
    definedness.written_undefined source previous unordered
      ⦃ result => result = writtenUndefined source previous unordered ⦄ := by
  cases source <;> cases unordered <;> simp [definedness.written_undefined, writtenUndefined]

end MemoryWrite.Definedness
