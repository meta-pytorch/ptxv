/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RustAtomic

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite

structure AtomicFloatSemantics where
  addF32Ftz : U32 → U32 → U32
  addF64 : U64 → U64 → U64
  addBf16 : U16 → U16 → U16

structure AtomicFloatContract {F : Type} (implementation : atomic_exec.FloatingArithmetic F)
    (semantics : AtomicFloatSemantics) : Prop where
  f32 : ∀ old value, implementation.add_f32_ftz old value
    ⦃ result => result = semantics.addF32Ftz old value ⦄
  f64 : ∀ old value, implementation.add_f64 old value
    ⦃ result => result = semantics.addF64 old value ⦄
  bf16 : ∀ old value, implementation.add_bf16 old value
    ⦃ result => result = semantics.addBf16 old value ⦄

def atomicIsCas : atomic_exec.Operation → Bool
  | .Integer .Cas => true
  | _ => false

def atomicIsSpin : atomic_exec.Operation → Bool
  | .Integer .CasSpin => true
  | _ => false

def atomicFinishModel (input : atomic_exec.AtomicInput)
    (computed : atomic_integer.IntegerResult) : atomic_exec.AtomicOutcome :=
  let didWrite := !atomicIsCas input.op || computed.swapped
  let spin := atomicIsSpin input.op
  { bytes := input.bytes
    old := computed.old
    replacement := computed.replacement
    swapped := computed.swapped
    did_write := didWrite
    returned := if spin then if computed.swapped then 1#u64 else 0#u64 else computed.old
    return_words := if spin then 1#u8 else if input.bytes = 8#u8 then 2#u8 else 1#u8
    predicate := if spin then computed.swapped else true
    advance_epoch := didWrite }

def atomicHighHalf (value : U64) : U16 :=
  ⟨value.bv.extractLsb' 16 16⟩

def atomicPackHalves (low high : U16) : U64 :=
  UScalar.cast .U64 (⟨high.bv ++ low.bv⟩ : U32)

def atomicFloatResult (semantics : AtomicFloatSemantics) (input : atomic_exec.AtomicInput) :
    atomic_integer.IntegerResult :=
  match input.op with
  | .AddF32Ftz =>
    let old := UScalar.cast .U32 input.old
    let value := UScalar.cast .U32 input.value
    ⟨UScalar.cast .U64 old, UScalar.cast .U64 (semantics.addF32Ftz old value), false⟩
  | .AddF64 => ⟨input.old, semantics.addF64 input.old input.value, false⟩
  | .AddBf16 =>
    let old := UScalar.cast .U16 input.old
    let value := UScalar.cast .U16 input.value
    ⟨UScalar.cast .U64 old, UScalar.cast .U64 (semantics.addBf16 old value), false⟩
  | .AddBf16x2 =>
    let low := semantics.addBf16 (UScalar.cast .U16 input.old) (UScalar.cast .U16 input.value)
    let high := semantics.addBf16 (atomicHighHalf input.old) (atomicHighHalf input.value)
    ⟨UScalar.cast .U64 (UScalar.cast .U32 input.old), atomicPackHalves low high, false⟩
  | .Integer _ => ⟨input.old, input.old, false⟩

def atomicFloatWidth : atomic_exec.Operation → U8
  | .AddF32Ftz | .AddBf16x2 => 4#u8
  | .AddF64 => 8#u8
  | .AddBf16 => 2#u8
  | .Integer _ => 0#u8

def AtomicSupported (op : atomic_exec.Operation) (bytes : U8) : Prop :=
  match op with
  | .Integer .Inc | .Integer .Dec => bytes = 4#u8
  | .Integer _ => bytes = 4#u8 ∨ bytes = 8#u8
  | .AddF32Ftz | .AddBf16x2 => bytes = 4#u8
  | .AddF64 => bytes = 8#u8
  | .AddBf16 => bytes = 2#u8

instance (op : atomic_exec.Operation) (bytes : U8) : Decidable (AtomicSupported op bytes) := by
  cases op with
  | Integer op => cases op <;> unfold AtomicSupported <;> infer_instance
  | _ => unfold AtomicSupported; infer_instance

@[step]
theorem rust_atomic_validate_spec (op : atomic_exec.Operation) (bytes : U8) :
    atomic_exec.validate op bytes ⦃ result => result =
      if AtomicSupported op bytes then .Ok () else .Err .UnsupportedWidth ⦄ := by
  cases op with
  | Integer op =>
    cases op <;> unfold atomic_exec.validate AtomicSupported <;>
      by_cases hfour : bytes = 4#u8 <;> by_cases height : bytes = 8#u8 <;>
      simp_all
  | _ => simp [atomic_exec.validate, AtomicSupported]; split <;> simp_all

@[step]
theorem rust_atomic_finish_spec (input : atomic_exec.AtomicInput)
    (computed : atomic_integer.IntegerResult) :
    atomic_exec.finish input computed
      ⦃ output => output = atomicFinishModel input computed ⦄ := by
  rcases input with ⟨op, bytes, old, value, compare⟩
  rcases computed with ⟨prior, replacement, swapped⟩
  cases op with
  | Integer op =>
    cases op <;> cases swapped <;> by_cases hwidth : bytes = 8#u8 <;>
      simp [atomic_exec.finish, atomicFinishModel, atomicIsCas, atomicIsSpin, hwidth,
        UScalar.cast_fromBool]
    all_goals decide
  | _ =>
    cases swapped <;> by_cases hwidth : bytes = 8#u8 <;>
      simp [atomic_exec.finish, atomicFinishModel, atomicIsCas, atomicIsSpin, hwidth,
        UScalar.cast_fromBool]

theorem atomicHighHalf_shift (value : U64) :
    UScalar.cast .U16 (⟨value.bv.ushiftRight 16⟩ : U64) = atomicHighHalf value := by
  apply U16.bv_eq_imp_eq
  simp only [UScalar.cast, atomicHighHalf, UScalarTy.numBits]
  apply BitVec.eq_of_getLsbD_eq
  intro index hindex
  simp [hindex, Nat.add_comm]

theorem atomicPackHalves_or (low high : U16) :
    UScalar.cast .U64 low ||| (⟨(UScalar.cast .U64 high).bv.shiftLeft 16⟩ : U64) =
      atomicPackHalves low high := by
  apply U64.bv_eq_imp_eq
  simp only [UScalar.bv_or, UScalar.cast, atomicPackHalves, UScalarTy.numBits]
  apply BitVec.eq_of_getLsbD_eq
  intro index hindex
  simp only [BitVec.getLsbD_setWidth, BitVec.getLsbD_or, BitVec.getLsbD_append]
  by_cases hlow : index < 16
  · simp [hindex, hlow]
  · by_cases hhigh : index < 32
    · simp [hindex, hlow, show index - 16 < 16 by omega,
        BitVec.getLsbD_of_ge low.bv index (by omega)]
    · simp [hindex, hlow,
        BitVec.getLsbD_of_ge low.bv index (by omega),
        BitVec.getLsbD_of_ge high.bv (index - 16) (by omega)]

theorem atomic_shift_right_sixteen (value : U64) :
    value >>> 16#i32 = .ok (⟨value.bv.ushiftRight 16⟩ : U64) := by rfl

theorem atomic_shift_left_sixteen (value : U64) :
    value <<< 16#i32 = .ok (⟨value.bv.shiftLeft 16⟩ : U64) := by rfl

theorem rust_atomic_execute_rejects {F : Type} (implementation : atomic_exec.FloatingArithmetic F)
    (input : atomic_exec.AtomicInput) (hwidth : ¬AtomicSupported input.op input.bytes) :
    atomic_exec.execute implementation input ⦃ result => result = .Err .UnsupportedWidth ⦄ := by
  unfold atomic_exec.execute
  apply WP.spec_bind (rust_atomic_validate_spec input.op input.bytes)
  intro validated hvalidated
  rw [ite_eq_right hwidth] at hvalidated
  rw [hvalidated]
  simp

theorem rust_atomic_float_execute_spec {F : Type}
    (implementation : atomic_exec.FloatingArithmetic F) (semantics : AtomicFloatSemantics)
    (hfloat : AtomicFloatContract implementation semantics)
    (input : atomic_exec.AtomicInput) (hoperation : ∀ op, input.op ≠ .Integer op) :
    atomic_exec.execute implementation input ⦃ result => result =
      if AtomicSupported input.op input.bytes then
        .Ok (atomicFinishModel input (atomicFloatResult semantics input))
      else .Err .UnsupportedWidth ⦄ := by
  by_cases hwidth : AtomicSupported input.op input.bytes
  · simp only [ite_eq_left hwidth]
    unfold atomic_exec.execute
    apply WP.spec_bind (rust_atomic_validate_spec input.op input.bytes)
    intro validated hvalidated
    rw [ite_eq_left hwidth] at hvalidated
    rw [hvalidated]
    cases hop : input.op with
    | Integer op => exact False.elim (hoperation op hop)
    | AddF32Ftz =>
      simp only [lift, bind_ok]
      apply WP.spec_bind (hfloat.f32 _ _)
      intro result hresult
      rw [hresult]
      apply WP.spec_bind (rust_atomic_finish_spec input _)
      intro output houtput
      simpa only [WP.spec_ok, atomicFloatResult, hop] using congrArg core.result.Result.Ok houtput
    | AddF64 =>
      apply WP.spec_bind (hfloat.f64 _ _)
      intro result hresult
      rw [hresult]
      apply WP.spec_bind (rust_atomic_finish_spec input _)
      intro output houtput
      simpa only [WP.spec_ok, atomicFloatResult, hop] using congrArg core.result.Result.Ok houtput
    | AddBf16 =>
      simp only [lift, bind_ok]
      apply WP.spec_bind (hfloat.bf16 _ _)
      intro result hresult
      rw [hresult]
      apply WP.spec_bind (rust_atomic_finish_spec input _)
      intro output houtput
      simpa only [WP.spec_ok, atomicFloatResult, hop] using congrArg core.result.Result.Ok houtput
    | AddBf16x2 =>
      simp only [lift, bind_ok]
      apply WP.spec_bind (hfloat.bf16 _ _)
      intro low hlow
      rw [hlow]
      simp only [bind_ok, atomic_shift_right_sixteen, atomicHighHalf_shift]
      apply WP.spec_bind (hfloat.bf16 _ _)
      intro high hhigh
      rw [hhigh]
      simp only [bind_ok, atomic_shift_left_sixteen, atomicPackHalves_or]
      apply WP.spec_bind (rust_atomic_finish_spec input _)
      intro output houtput
      simpa only [WP.spec_ok, atomicFloatResult, hop] using congrArg core.result.Result.Ok houtput
  · simpa only [ite_eq_right hwidth] using rust_atomic_execute_rejects implementation input hwidth

end MemoryWrite
