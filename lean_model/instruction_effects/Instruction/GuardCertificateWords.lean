/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GuardCertificateLookup

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace Instruction.GuardCertificate

theorem shift_sign_extension {width : Nat} (word : BitVec 64)
    (positive : 0 < width) (bounded : width ≤ 64) :
    (word <<< (64 - width)).sshiftRight (64 - width) =
      (word.setWidth width).signExtend 64 := by
  apply BitVec.eq_of_getLsbD_eq
  intro index
  rw [BitVec.getLsbD_sshiftRight, BitVec.getLsbD_signExtend]
  by_cases inside : index < 64
  · have notPast : ¬64 ≤ index := by omega
    simp only [notPast, decide_false, Bool.not_false, Bool.true_and, inside, decide_true]
    by_cases low : index < width
    · have shiftedInside : 64 - width + index < 64 := by omega
      have notBelow : ¬64 - width + index < 64 - width := by omega
      simp only [low, shiftedInside, ite_true, BitVec.getLsbD_shiftLeft, decide_true,
        notBelow, decide_false, Bool.not_false, Bool.true_and, BitVec.getLsbD_setWidth]
      intro _
      congr 1
      omega
    · have shiftedOutside : ¬64 - width + index < 64 := by omega
      have lastInside : 64 - 1 < 64 := by omega
      have lastNotBelow : ¬64 - 1 < 64 - width := by omega
      have narrowLast : width - 1 < width := by omega
      simp only [low, shiftedOutside, ite_false, BitVec.msb_eq_getLsbD_last,
        BitVec.getLsbD_shiftLeft, lastInside, decide_true, lastNotBelow, decide_false,
        Bool.not_false, Bool.true_and, BitVec.getLsbD_setWidth, narrowLast]
      intro _
      congr 1
      omega
  · have past : 64 ≤ index := by omega
    simp [past, inside]

def wordAt (width : Nat) (word : U64) : BitVec width := BitVec.ofNat width word.val

def compareWords (comparison : Cmp) (width : Nat) (left right : U64) : Bool :=
  let x := wordAt width left
  let y := wordAt width right
  match comparison with
  | .Eq => decide (x.toNat = y.toNat)
  | .Ne => decide (x.toNat ≠ y.toNat)
  | .LtU => decide (x.toNat < y.toNat)
  | .LeU => decide (x.toNat ≤ y.toNat)
  | .GtU => decide (x.toNat > y.toNat)
  | .GeU => decide (x.toNat ≥ y.toNat)
  | .LtS => decide (x.toInt < y.toInt)
  | .LeS => decide (x.toInt ≤ y.toInt)
  | .GtS => decide (x.toInt > y.toInt)
  | .GeS => decide (x.toInt ≥ y.toInt)


@[step]
theorem rust_compare_mask (bits : U8) (positive : 0 < bits.val) (bounded : bits.val ≤ 64) :
    (if bits = 64#u8 then .ok core.num.U64.MAX else do
      let shifted ← 1#u64 <<< bits
      shifted - 1#u64) ⦃ mask => mask.val = 2 ^ bits.val - 1 ⦄ := by
  split
  · rename_i full
    subst bits
    simp [core.num.U64.MAX, U64.rMax, WP.spec_ok]
  · rename_i small
    have less : bits.val < 64 := by scalar_tac
    have powerSmall : 2 ^ bits.val < U64.size := by
      simp only [U64.size, U64.numBits]
      exact Nat.pow_lt_pow_right (by decide) less
    step with U64.ShiftLeft_spec (1#u64) bits less as ⟨shifted, valueAt, _⟩
    have shiftedAt : shifted.val = 2 ^ bits.val := by
      simpa only [Nat.shiftLeft_eq, Nat.one_mul, Nat.mod_eq_of_lt powerSmall] using valueAt
    step with U64.sub_spec (x := shifted) (y := 1#u64)
      (by have := Nat.two_pow_pos bits.val; scalar_tac) as ⟨mask, maskAt⟩
    scalar_tac

theorem masked_word (word mask : U64) (width : Nat)
    (maskAt : mask.val = 2 ^ width - 1) :
    (word &&& mask).val = (wordAt width word).toNat := by
  simp only [UScalar.val_and, maskAt, Nat.and_two_pow_sub_one_eq_mod,
    wordAt, BitVec.toNat_ofNat]

theorem masked_narrow_word (word mask : U64) (width : Nat)
    (maskAt : mask.val = 2 ^ width - 1) :
    (word &&& mask).bv.setWidth width = wordAt width word := by
  apply BitVec.eq_of_toNat_eq
  simp only [BitVec.toNat_setWidth, UScalar.bv_toNat, masked_word word mask width maskAt]
  exact Nat.mod_eq_of_lt (wordAt width word).isLt


theorem scalar_beq (left right : U64) :
    (left == right) = decide (left.val = right.val) := by
  apply Bool.eq_iff_iff.mpr
  simp only [beq_iff_eq, decide_eq_true_eq, UScalar.eq_equiv]

@[step]
theorem rust_compare (comparison : Cmp) (bits : U8) (x y : U64)
    (positive : 0 < bits.val) (bounded : bits.val ≤ 64) :
    guard_certificate.compare comparison bits x y
      ⦃ result => result = some (compareWords comparison bits.val x y) ⦄ := by
  have nonzero : bits ≠ 0#u8 := by scalar_tac
  have notLarge : ¬bits > 64#u8 := by scalar_tac
  unfold guard_certificate.compare
  simp only [nonzero, notLarge, ite_false]
  step with rust_compare_mask bits positive bounded as ⟨mask, maskAt⟩
  simp only [lift, bind_ok]
  step with U8.sub_spec (x := 64#u8) (y := bits) (by scalar_tac)
    as ⟨shift, shiftAt⟩
  have shiftValue : shift.val = 64 - bits.val := by scalar_tac
  have shiftBound : shift.val < 64 := by omega
  step with U64.ShiftLeft_spec (x &&& mask) shift shiftBound as ⟨lx, _, lxAt⟩
  step with I64.ShiftRight_spec (UScalar.hcast .I64 lx) shift shiftBound
    as ⟨sx, _, sxAt⟩
  step with U64.ShiftLeft_spec (y &&& mask) shift shiftBound as ⟨ly, _, lyAt⟩
  step with I64.ShiftRight_spec (UScalar.hcast .I64 ly) shift shiftBound
    as ⟨sy, _, syAt⟩
  have signedX : sx.val = (wordAt bits.val x).toInt := by
    have signed := congrArg BitVec.toInt sxAt
    change sx.val = _ at signed
    simp only [U64.bv] at lxAt
    simp only [I64.bv, UScalar.hcast_bv_eq, IScalarTy.I64_numBits_eq, UScalarTy.U64_numBits_eq, BitVec.setWidth_eq] at signed
    rw [lxAt, shiftValue,
      shift_sign_extension _ positive bounded, masked_narrow_word x mask bits.val maskAt,
      BitVec.toInt_signExtend_of_le bounded] at signed
    exact signed
  have signedY : sy.val = (wordAt bits.val y).toInt := by
    have signed := congrArg BitVec.toInt syAt
    change sy.val = _ at signed
    simp only [U64.bv] at lyAt
    simp only [I64.bv, UScalar.hcast_bv_eq, IScalarTy.I64_numBits_eq, UScalarTy.U64_numBits_eq, BitVec.setWidth_eq] at signed
    rw [lyAt, shiftValue,
      shift_sign_extension _ positive bounded, masked_narrow_word y mask bits.val maskAt,
      BitVec.toInt_signExtend_of_le bounded] at signed
    exact signed
  cases comparison <;>
    simp [WP.spec_ok, compareWords, bne, scalar_beq, UScalar.eq_equiv, UScalar.lt_equiv,
      UScalar.le_equiv, IScalar.lt_equiv, IScalar.le_equiv,
      masked_word x mask bits.val maskAt, masked_word y mask bits.val maskAt,
      signedX, signedY]

theorem successful_compare (comparison : Cmp) (bits : U8) (x y : U64) (truth : Bool)
    (returned : guard_certificate.compare comparison bits x y = .ok (some truth)) :
    0 < bits.val ∧ bits.val ≤ 64 ∧ truth = compareWords comparison bits.val x y := by
  by_cases zero : bits = 0#u8
  · simp [guard_certificate.compare, zero] at returned
  by_cases large : bits > 64#u8
  · simp [guard_certificate.compare, zero, large] at returned
  have positive : 0 < bits.val := by scalar_tac
  have bounded : bits.val ≤ 64 := by scalar_tac
  have exactResult := rust_compare comparison bits x y positive bounded
  simp only [returned, WP.spec_ok] at exactResult
  exact ⟨positive, bounded, Option.some.inj exactResult⟩

end Instruction.GuardCertificate
