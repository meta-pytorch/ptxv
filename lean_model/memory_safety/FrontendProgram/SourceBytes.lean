/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Construction

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace PtxFrontendProgram

private def bytesSuffix (left right : List U8) (index : Nat) (same : Bool) : Bool :=
  same && decide (left.drop index = right.drop index)

private theorem bytes_loop_false (left right : Slice U8) (index : Usize) :
    strict_memory_frontend_launch.bytes_equal_loop left right false index = .ok false := by
  unfold strict_memory_frontend_launch.bytes_equal_loop
  rw [loop.eq_def]
  simp [strict_memory_frontend_launch.bytes_equal_loop.body]

private theorem bytes_loop (left right : Slice U8) (same : Bool) (index : Usize)
    (lengths : left.val.length = right.val.length) (bound : index.val ≤ left.val.length) :
    strict_memory_frontend_launch.bytes_equal_loop left right same index
      ⦃ output => output = bytesSuffix left.val right.val index.val same ⦄ := by
  unfold strict_memory_frontend_launch.bytes_equal_loop
  apply loop.spec_decr_nat
    (fun current => left.val.length - current.2.val)
    (fun current => current.2.val ≤ left.val.length ∧
      bytesSuffix left.val right.val current.2.val current.1 = bytesSuffix left.val right.val index.val same)
  · rintro ⟨currentSame, currentIndex⟩ ⟨bound, invariant⟩
    unfold strict_memory_frontend_launch.bytes_equal_loop.body
    cases currentSame with
    | false => simpa [bytesSuffix] using invariant
    | true =>
      simp only [ite_true]
      by_cases inside : currentIndex < Slice.len left
      · simp only [inside, ite_true]
        step as ⟨leftByte, leftEq⟩
        step as ⟨rightByte, rightEq⟩
        step as ⟨next, nextVal⟩
        refine ⟨?_, ?_, ?_⟩
        · scalar_tac
        · have leftBound : currentIndex.val < left.val.length := by scalar_tac
          have rightBound : currentIndex.val < right.val.length := by omega
          unfold bytesSuffix at invariant
          rw [List.drop_eq_getElem_cons leftBound, List.drop_eq_getElem_cons rightBound] at invariant
          simp only [Bool.true_and] at invariant
          simpa only [bytesSuffix, nextVal, List.cons.injEq, Bool.decide_and, ← leftEq, ← rightEq] using invariant
        · scalar_tac
      · simp only [inside, ite_false, WP.spec_ok]
        have pastLeft : left.val.length ≤ currentIndex.val := by scalar_tac
        have pastRight : right.val.length ≤ currentIndex.val := by omega
        simpa only [bytesSuffix, List.drop_eq_nil_of_le pastLeft,
          List.drop_eq_nil_of_le pastRight, decide_true, Bool.and_true] using invariant
  · exact ⟨bound, rfl⟩

@[step]
theorem source_bytes_equal (left right : Slice U8) :
    strict_memory_frontend_launch.bytes_equal left right
      ⦃ output => output = decide (left.val = right.val) ⦄ := by
  unfold strict_memory_frontend_launch.bytes_equal
  by_cases equalLength : Slice.len left = Slice.len right
  · simp only [equalLength, decide_true]
    have lengths : left.val.length = right.val.length := by scalar_tac
    simpa only [bytesSuffix, List.drop_zero, Bool.true_and, UScalar.ofNatCore_val_eq] using bytes_loop left right true 0#usize lengths (by simp)
  · have different : left.val ≠ right.val := by intro h; apply equalLength; simp [Slice.len, h]
    simp only [equalLength, decide_false, bytes_loop_false, WP.spec_ok]
    simp [different]

theorem source_string_bytes (text : String) (bytes : Slice U8)
    (computed : MemorySafety.Primitives.stringBytes text = .ok bytes) :
    bytes.val = utf8 text := by
  unfold MemorySafety.Primitives.stringBytes at computed
  dsimp only at computed
  split at computed
  · simp only [ok.injEq] at computed
    rw [← computed]
    simp [utf8]
  · simp at computed

theorem utf8_injective {left right : String} (same : utf8 left = utf8 right) : left = right := by
  apply String.toByteArray_inj.mp
  apply ByteArray.ext
  apply Array.toList_inj.mp
  have bytes := congrArg (List.map (fun byte : U8 => UInt8.ofNat byte.val)) same
  simpa [utf8, List.map_map, Function.comp_def, UScalar.ofNat] using bytes

@[simp]
theorem utf8_eq_iff (left right : String) : utf8 left = utf8 right ↔ left = right :=
  ⟨utf8_injective, congrArg utf8⟩

end PtxFrontendProgram
