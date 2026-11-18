/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Theorem

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace MemoryWrite

deriving instance DecidableEq for atomic_integer.IntegerOp, atomic_integer.AtomicError,
  atomic_integer.IntegerResult

def integerSigned (bits value : Nat) : Int := Int.bmod value (2 ^ bits)

def integerUpdate (bits : Nat) (op : atomic_integer.IntegerOp)
    (old value compare : Nat) : Nat :=
  match op with
  | .Or => old ||| value
  | .And => old &&& value
  | .Xor => old ^^^ value
  | .Add => (old + value) % 2 ^ bits
  | .Exch => value
  | .MinU => min old value
  | .MaxU => max old value
  | .MinS => if integerSigned bits old < integerSigned bits value then old else value
  | .MaxS => if integerSigned bits old > integerSigned bits value then old else value
  | .Inc => if old ≥ value then 0 else old + 1
  | .Dec => if old = 0 ∨ old > value then value else old - 1
  | .Cas | .CasSpin => if old = compare then value else old

def integerIsCas (op : atomic_integer.IntegerOp) : Bool :=
  match op with
  | .Cas | .CasSpin => true
  | _ => false

def integerWord (value : Nat) : U64 := ⟨BitVec.ofNat 64 value⟩

@[simp]
theorem integerWord_val (value : Nat) : (integerWord value).val = value % 2 ^ 64 := by
  rfl

@[simp]
theorem integerWord_value (value : U64) : integerWord value.val = value := by
  apply UScalar.eq_of_val_eq
  rw [integerWord_val]
  exact Nat.mod_eq_of_lt value.bv.isLt

def integerNormalize (bytes : U8) (value : U64) : U64 :=
  if bytes = 4#u8 then UScalar.cast .U64 (UScalar.cast .U32 value) else value

def integerModel (input : atomic_integer.IntegerInput) :
    core.result.Result atomic_integer.IntegerResult atomic_integer.AtomicError :=
  if input.bytes = 4#u8 ∨
      (input.bytes = 8#u8 ∧ input.op ≠ .Inc ∧ input.op ≠ .Dec) then
    let bits := if input.bytes = 4#u8 then 32 else 64
    let old := integerNormalize input.bytes input.old
    let value := integerNormalize input.bytes input.value
    let compare := integerNormalize input.bytes input.compare
    .Ok ⟨old, integerWord (integerUpdate bits input.op old.val value.val compare.val),
      integerIsCas input.op && decide (old = compare)⟩
  else .Err .UnsupportedWidth

@[step]
theorem rust_integer_is_cas_spec (op : atomic_integer.IntegerOp) :
    (atomic_integer.is_cas op : Result Bool) ⦃ result => result = integerIsCas op ⦄ := by
  cases op <;> simp [atomic_integer.is_cas, integerIsCas]

@[step]
theorem rust_integer_replacement32_spec (op : atomic_integer.IntegerOp)
    (old value compare : U32) :
    atomic_integer.replacement32 op old value compare
      ⦃ result => result.val = integerUpdate 32 op old.val value.val compare.val ⦄ := by
  cases op
  case Or => simp [atomic_integer.replacement32, integerUpdate]
  case And => simp [atomic_integer.replacement32, integerUpdate]
  case Xor => simp [atomic_integer.replacement32, integerUpdate]
  case Add => simp [atomic_integer.replacement32, integerUpdate, U32.size, U32.numBits]
  case Exch => simp [atomic_integer.replacement32, integerUpdate]
  case MinU =>
    simp only [atomic_integer.replacement32, integerUpdate]
    split <;> simp only [WP.spec_ok] <;> scalar_tac
  case MaxU =>
    simp only [atomic_integer.replacement32, integerUpdate]
    split <;> simp only [WP.spec_ok] <;> scalar_tac
  case MinS =>
    simp only [atomic_integer.replacement32, lift, bind_ok]
    have hcmp : (UScalar.hcast .I32 old < UScalar.hcast .I32 value) ↔
        integerSigned 32 old.val < integerSigned 32 value.val := by
      change (UScalar.hcast .I32 old).val < (UScalar.hcast .I32 value).val ↔ _
      simp only [UScalar.hcast_val_eq, IScalarTy.numBits, integerSigned]
    simp only [hcmp, integerUpdate]
    split <;> simp_all only [WP.spec_ok]
  case MaxS =>
    simp only [atomic_integer.replacement32, lift, bind_ok]
    have hcmp : (UScalar.hcast .I32 old > UScalar.hcast .I32 value) ↔
        integerSigned 32 old.val > integerSigned 32 value.val := by
      change (UScalar.hcast .I32 old).val > (UScalar.hcast .I32 value).val ↔ _
      simp only [UScalar.hcast_val_eq, IScalarTy.numBits, integerSigned]
    simp only [hcmp, integerUpdate]
    split <;> simp_all only [WP.spec_ok]
  case Inc =>
    simp only [atomic_integer.replacement32]
    split
    · simp only [WP.spec_ok, integerUpdate]; split <;> scalar_tac
    · step as ⟨next, hnext⟩; try scalar_tac
      simp only [integerUpdate]; split <;> scalar_tac
  case Dec =>
    simp only [atomic_integer.replacement32]
    split
    · simp only [WP.spec_ok, integerUpdate]; split <;> scalar_tac
    · split
      · simp only [WP.spec_ok, integerUpdate]; split <;> scalar_tac
      · step as ⟨next, hnext⟩; try scalar_tac
        simp only [integerUpdate]; split <;> scalar_tac
  case Cas =>
    simp only [atomic_integer.replacement32, integerUpdate]
    split <;> simp_all only [WP.spec_ok] <;> split <;> scalar_tac
  case CasSpin =>
    simp only [atomic_integer.replacement32, integerUpdate]
    split <;> simp_all only [WP.spec_ok] <;> split <;> scalar_tac

def Integer64ResultSpec (op : atomic_integer.IntegerOp) (old value compare : U64)
    (result : core.result.Result U64 atomic_integer.AtomicError) : Prop :=
  match result with
  | .Ok replacement => op ≠ .Inc ∧ op ≠ .Dec ∧
      replacement.val = integerUpdate 64 op old.val value.val compare.val
  | .Err _ => op = .Inc ∨ op = .Dec

@[step]
theorem rust_integer_replacement64_spec (op : atomic_integer.IntegerOp)
    (old value compare : U64) :
    atomic_integer.replacement64 op old value compare
      ⦃ result => Integer64ResultSpec op old value compare result ⦄ := by
  cases op
  case Or => simp [atomic_integer.replacement64, lift, bind_ok, Integer64ResultSpec, integerUpdate]
  case And => simp [atomic_integer.replacement64, lift, bind_ok, Integer64ResultSpec, integerUpdate]
  case Xor => simp [atomic_integer.replacement64, lift, bind_ok, Integer64ResultSpec, integerUpdate]
  case Add => simp [atomic_integer.replacement64, lift, bind_ok, Integer64ResultSpec, integerUpdate,
    U64.size, U64.numBits]
  case Exch => simp [atomic_integer.replacement64, Integer64ResultSpec, integerUpdate]
  case MinU =>
    simp only [atomic_integer.replacement64]
    split <;> simp [Integer64ResultSpec, integerUpdate] <;> scalar_tac
  case MaxU =>
    simp only [atomic_integer.replacement64]
    split <;> simp [Integer64ResultSpec, integerUpdate] <;> scalar_tac
  case MinS =>
    simp only [atomic_integer.replacement64, lift, bind_ok]
    have hcmp : (UScalar.hcast .I64 old < UScalar.hcast .I64 value) ↔
        integerSigned 64 old.val < integerSigned 64 value.val := by
      change (UScalar.hcast .I64 old).val < (UScalar.hcast .I64 value).val ↔ _
      simp only [UScalar.hcast_val_eq, IScalarTy.numBits, integerSigned]
    simp only [hcmp]
    split <;> simp_all [Integer64ResultSpec, integerUpdate]
  case MaxS =>
    simp only [atomic_integer.replacement64, lift, bind_ok]
    have hcmp : (UScalar.hcast .I64 old > UScalar.hcast .I64 value) ↔
        integerSigned 64 old.val > integerSigned 64 value.val := by
      change (UScalar.hcast .I64 old).val > (UScalar.hcast .I64 value).val ↔ _
      simp only [UScalar.hcast_val_eq, IScalarTy.numBits, integerSigned]
    simp only [hcmp]
    split <;> simp_all [Integer64ResultSpec, integerUpdate]
  case Inc => simp [atomic_integer.replacement64, Integer64ResultSpec]
  case Dec => simp [atomic_integer.replacement64, Integer64ResultSpec]
  case Cas =>
    simp only [atomic_integer.replacement64]
    split <;> simp_all [Integer64ResultSpec, integerUpdate]
  case CasSpin =>
    simp only [atomic_integer.replacement64]
    split <;> simp_all [Integer64ResultSpec, integerUpdate]

theorem integer_normalize32 (value : U64) :
    (UScalar.cast .U32 value).val = value.val % 2 ^ 32 := by
  simpa only [UScalarTy.numBits] using UScalar.cast_val_eq .U32 value

theorem integer_normalize_val (bytes : U8) (value : U64) :
    (integerNormalize bytes value).val =
      if bytes = 4#u8 then value.val % 2 ^ 32 else value.val := by
  unfold integerNormalize
  split
  · rw [U32.cast_U64_val_eq, integer_normalize32]
  · rfl

theorem integer_normalize_width (bytes : U8) (value : U64)
    (hbytes : bytes = 4#u8 ∨ bytes = 8#u8) :
    (integerNormalize bytes value).val = value.val % 2 ^ (8 * bytes.val) := by
  rcases hbytes with rfl | rfl
  · simp only [integer_normalize_val]
    rfl
  · simp only [integer_normalize_val]
    exact (Nat.mod_eq_of_lt value.bv.isLt).symm

theorem integer_widen32 (value : U32) :
    UScalar.cast .U64 value = integerWord value.val := by
  apply UScalar.eq_of_val_eq
  rw [U32.cast_U64_val_eq, integerWord_val]
  exact (Nat.mod_eq_of_lt (by have := value.hmax; scalar_tac)).symm

theorem integer_word_of_value (value : U64) (expected : Nat) (h : value.val = expected) :
    value = integerWord expected := by
  rw [←h, integerWord_value]

@[step]
theorem rust_integer_eval_spec (input : atomic_integer.IntegerInput) :
    atomic_integer.eval input ⦃ result => result = integerModel input ⦄ := by
  unfold atomic_integer.eval
  by_cases hfour : input.bytes = 4#u8
  · simp only [hfour, ite_true, lift, bind_ok]
    step with rust_integer_replacement32_spec input.op
      (UScalar.cast .U32 input.old) (UScalar.cast .U32 input.value)
      (UScalar.cast .U32 input.compare) as ⟨replacement, hreplacement⟩
    have hword : UScalar.cast .U64 replacement =
        integerWord (integerUpdate 32 input.op
          (UScalar.cast .U32 input.old).val (UScalar.cast .U32 input.value).val
          (UScalar.cast .U32 input.compare).val) := by
      rw [integer_widen32, hreplacement]
    step with rust_integer_is_cas_spec input.op as ⟨cas, hcas⟩
    simp only [integerModel, hfour, true_or, ite_true, integerNormalize,
      U32.cast_U64_val_eq, hword]
    rw [hcas]
    cases integerIsCas input.op <;> simp
  · simp only [hfour, ite_false]
    by_cases height : input.bytes = 8#u8
    · simp only [height, ite_true]
      step with rust_integer_replacement64_spec input.op input.old input.value input.compare
        as ⟨result, hresult⟩
      cases result with
      | Err error =>
        have hop : input.op = .Inc ∨ input.op = .Dec := hresult
        cases error
        rcases hop with hop | hop <;> simp [integerModel, hfour, hop]
      | Ok replacement =>
        obtain ⟨hninc, hndec, hreplacement⟩ := hresult
        have hword : replacement =
            integerWord (integerUpdate 64 input.op input.old.val input.value.val input.compare.val) :=
          integer_word_of_value replacement _ hreplacement
        step with rust_integer_is_cas_spec input.op as ⟨cas, hcas⟩
        simp only [integerModel, height, integerNormalize, hword]
        rw [hcas]
        cases integerIsCas input.op <;> simp [hninc, hndec]
    · simp [integerModel, hfour, height]

end MemoryWrite
