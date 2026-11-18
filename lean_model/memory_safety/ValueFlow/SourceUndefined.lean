/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.SourcePrimitives

open Aeneas Aeneas.Std Result
open PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace ValueFlow.Source

def mergeUndefined (index : Nat) (mask : Nat) (entry : U16 × U64) : Nat :=
  if entry.1.val = index then mask ||| entry.2.val else mask

@[step]
theorem source_read_undefined (entries : alloc.vec.Vec (U16 × U64)) (index : U16)
    (undefined : U64) (position : Usize) (bounded : position.val ≤ entries.val.length) :
    value_flow.read_virtual_loop entries index undefined position ⦃ result =>
      result.val = (entries.val.drop position.val).foldl (mergeUndefined index.val) undefined.val ⦄ := by
  unfold value_flow.read_virtual_loop
  apply loop.spec_decr_nat
    (fun current : U64 × Usize => entries.val.length - current.2.val)
    (fun current : U64 × Usize =>
      current.2.val ≤ entries.val.length ∧
      (entries.val.drop current.2.val).foldl (mergeUndefined index.val) current.1.val =
        (entries.val.drop position.val).foldl (mergeUndefined index.val) undefined.val)
  · rintro ⟨mask, current⟩ ⟨within, accumulated⟩
    change current.val ≤ entries.val.length at within
    change (entries.val.drop current.val).foldl (mergeUndefined index.val) mask.val = _ at accumulated
    unfold value_flow.read_virtual_loop.body
    by_cases active : current.val < entries.val.length
    · have tested : current < alloc.vec.Vec.len entries := by scalar_tac
      simp only [tested, ite_true]
      step as ⟨key, value, entryEq⟩
      split
      all_goals
        simp only [bind_ok]
        step as ⟨next, nextVal⟩
        refine ⟨by scalar_tac, ?_, by scalar_tac⟩
        rw [List.drop_eq_getElem_cons active, List.foldl_cons] at accumulated
        rw [nextVal]
        simp only [←entryEq] at accumulated
        have compare : (key.val = index.val) ↔ key = index := by
          constructor
          · exact UScalar.eq_of_val_eq
          · intro equal; exact congrArg UScalar.val equal
        simpa only [mergeUndefined, UScalar.val_or, compare, *, ite_true, ite_false] using accumulated
    · have tested : ¬current < alloc.vec.Vec.len entries := by scalar_tac
      simp only [tested, ite_false, WP.spec_ok]
      simpa only [List.drop_eq_nil_of_le (by omega : entries.val.length ≤ current.val),
        List.foldl_nil] using accumulated
  · exact ⟨bounded, rfl⟩

end ValueFlow.Source
