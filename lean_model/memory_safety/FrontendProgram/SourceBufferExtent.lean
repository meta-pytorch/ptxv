/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.SourceBytes

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace PtxFrontendProgram

def DtypeBytes (dtype : List U8) (width : Nat) : Prop :=
  (dtype, width) ∈ [(utf8 "u8",1),(utf8 "i8",1),(utf8 "u16",2),(utf8 "i16",2),
    (utf8 "u32",4),(utf8 "i32",4),(utf8 "u64",8),(utf8 "i64",8)]

@[step]
theorem source_dtype_bytes (dtype : Slice U8) :
    strict_memory_frontend_launch.dtype_bytes dtype
      ⦃ result => ∀ width, result = some width → DtypeBytes dtype.val width.val ⦄ := by
  unfold strict_memory_frontend_launch.dtype_bytes
  simp only [lift, bind_ok]
  repeat' first | step | split
  all_goals simp_all [DtypeBytes, utf8, Array.to_slice, Array.make]
  all_goals decide

private def remainingShape (shape : List U64) (index : Nat) : Nat :=
  ((shape.drop index).map UScalar.val).prod

private theorem extent_loop (shape : alloc.vec.Vec U64) (bytes : Option U64) (index : Usize)
    (bound : index.val ≤ shape.val.length) :
    strict_memory_frontend_launch.buffer_extent_loop shape bytes index
      ⦃ output => ∀ result, output = some result → ∃ before,
        bytes = some before ∧ result.val = before.val * remainingShape shape.val index.val ⦄ := by
  unfold strict_memory_frontend_launch.buffer_extent_loop
  apply loop.spec_decr_nat
    (fun current => shape.val.length - current.2.val)
    (fun current => current.2.val ≤ shape.val.length ∧
      ∀ value, current.1 = some value → ∃ before, bytes = some before ∧
        value.val * remainingShape shape.val current.2.val = before.val * remainingShape shape.val index.val)
  · rintro ⟨current,indexNow⟩ ⟨bound,invariant⟩
    unfold strict_memory_frontend_launch.buffer_extent_loop.body
    cases current with
    | none => simp [core.option.Option.is_some]
    | some before =>
      simp only [core.option.Option.is_some, Option.isSome_some, ite_true]
      by_cases inside : indexNow < alloc.vec.Vec.len shape
      · simp only [inside, ite_true]
        step as ⟨dimension, dimensionEq⟩
        have multiplication := U64.checked_mul_bv_spec before dimension
        cases multiplied : U64.checked_mul before dimension with
        | none =>
          step as ⟨next, nextVal⟩
          refine ⟨?_, ?_, ?_⟩
          · scalar_tac
          · intro value impossible; cases impossible
          · scalar_tac
        | some after =>
          rw [multiplied] at multiplication
          step as ⟨next, nextVal⟩
          refine ⟨?_, ?_, ?_⟩
          · scalar_tac
          · intro value same
            cases same
            obtain ⟨initial, original, equation⟩ := invariant before rfl
            refine ⟨initial, original, ?_⟩
            have listInside : indexNow.val < shape.val.length := by scalar_tac
            have suffix : remainingShape shape.val indexNow.val =
                dimension.val * remainingShape shape.val (indexNow.val+1) := by
              unfold remainingShape
              rw [List.drop_eq_getElem_cons listInside]
              simp only [List.map_cons, List.prod_cons, ←dimensionEq]
            simpa only [nextVal, multiplication.2.1, suffix, Nat.mul_assoc] using equation
          · scalar_tac
      · simp only [inside, ite_false, WP.spec_ok]
        intro result same
        cases same
        obtain ⟨initial, original, equation⟩ := invariant before rfl
        refine ⟨initial, original, ?_⟩
        have past : shape.val.length ≤ indexNow.val := by scalar_tac
        simpa only [remainingShape, List.drop_eq_nil_of_le past, List.map_nil, List.prod_nil, Nat.mul_one] using equation
  · refine ⟨bound, ?_⟩
    intro value same
    exact ⟨value,same,rfl⟩

theorem source_buffer_extent (buffer : Buffer) (extent : U64)
    (computed : strict_memory_frontend_launch.buffer_extent buffer = .ok (some extent)) :
    BufferExtent buffer extent.val := by
  unfold strict_memory_frontend_launch.buffer_extent at computed
  obtain ⟨dtype, stringComputed, computed⟩ := source_bind_success _ _ _ computed
  obtain ⟨widthOption, dtypeComputed, computed⟩ := source_bind_success _ _ _ computed
  have loopMeaning := extent_loop buffer.shape widthOption 0#usize (by simp)
  rw [computed] at loopMeaning
  simp only [WP.spec_ok] at loopMeaning
  obtain ⟨width, selected, extentEq⟩ := loopMeaning extent rfl
  subst widthOption
  have dtypeMeaning := source_dtype_bytes dtype
  rw [dtypeComputed] at dtypeMeaning
  simp only [WP.spec_ok] at dtypeMeaning
  have format := dtypeMeaning width rfl
  rw [source_string_bytes buffer.dtype dtype stringComputed] at format
  refine ⟨width.val, ?_, ?_⟩
  · simpa only [DtypeBytes, List.mem_cons, List.mem_singleton, List.not_mem_nil, or_false, Prod.mk.injEq, utf8_eq_iff] using format
  · simpa [remainingShape] using extentEq

end PtxFrontendProgram
