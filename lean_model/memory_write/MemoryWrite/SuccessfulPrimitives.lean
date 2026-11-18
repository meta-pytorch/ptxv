/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.PartialLoops

open Aeneas Aeneas.Std Result

set_option autoImplicit false

namespace MemoryWrite

theorem successful_try_mk {ty : UScalarTy} (value : Nat) (result : UScalar ty)
    (success : UScalar.tryMk ty value = .ok result) : result.val = value := by
  unfold UScalar.tryMk UScalar.tryMkOpt at success
  split at success
  · simp only [Result.ofOption] at success
    have equal := Result.ok_injective success
    rw [←equal]
    rfl
  · simp only [Result.ofOption, fail_not_ok] at success

theorem successful_add {ty : UScalarTy} (left right result : UScalar ty)
    (success : UScalar.add left right = .ok result) : result.val = left.val + right.val :=
  successful_try_mk _ _ success

theorem successful_mul {ty : UScalarTy} (left right result : UScalar ty)
    (success : UScalar.mul left right = .ok result) : result.val = left.val * right.val :=
  successful_try_mk _ _ success

theorem successful_vec_push {α : Type} (before : alloc.vec.Vec α) (value : α)
    (result : alloc.vec.Vec α) (success : alloc.vec.Vec.push before value = .ok result) :
    result.val = before.val ++ [value] := by
  unfold alloc.vec.Vec.push at success
  dsimp only at success
  split at success
  · have equal := Result.ok_injective success
    rw [←equal]
    simp only [alloc.vec.Vec.from_val, List.concat_eq_append]
  · simp only [fail_not_ok] at success

end MemoryWrite
