/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.OwnerRefinement

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace Instruction.DestinationRows

abbrev RawKey := U32 × U8 × U16

def rawKey (register : Reg) : RawKey :=
  ⟨0#u32, IScalar.hcast .U8 (read_discriminant register.file), register.idx⟩

def vecMap {α β : Type} (f : α → β) (values : alloc.vec.Vec α) : alloc.vec.Vec β :=
  alloc.vec.Vec.from (values.val.map f) (by simpa only [List.length_map] using values.property)

def rawRows (rows : Slice (alloc.vec.Vec Reg)) : alloc.vec.Vec (alloc.vec.Vec RawKey) :=
  alloc.vec.Vec.from (rows.val.map (vecMap rawKey)) (by simpa only [List.length_map] using rows.property)

@[step]
theorem rust_raw_key (register : Reg) :
    register_owner.canonical_key 0#usize (Slice.from [] (by simp)) register
      ⦃ result => result = rawKey register ⦄ := by
  apply WP.spec_mono (RegisterOwner.rust_canonical_key 0#usize (Slice.from [] (by simp)) register)
  rintro ⟨owner, file, index⟩ observed
  simp only [RegisterOwner.keyView, RegisterOwner.canonicalKey, Slice.from_val,
    List.map_nil, List.getElem?_nil, Option.getD_none, ite_self] at observed
  have ownerValue : owner.val = 0 := by
    have same := congrArg RegisterOwner.Key.owner observed
    scalar_tac
  have fileValue : file.val = RegisterOwner.fileTag register.file :=
    congrArg RegisterOwner.Key.file observed
  have indexValue : index.val = register.idx.val :=
    congrArg RegisterOwner.Key.index observed
  simp only [rawKey, Prod.mk.injEq]
  exact ⟨UScalar.eq_of_val_eq ownerValue, UScalar.eq_of_val_eq
    (fileValue.trans (RegisterOwner.file_tag_discriminant register.file).symm),
    UScalar.eq_of_val_eq indexValue⟩

@[step]
theorem rust_key_row (rows : Slice (alloc.vec.Vec Reg)) (pc : Usize)
    (inside : pc.val < rows.val.length) :
    destination_rows.keys_loop0_loop0 rows pc
      (alloc.vec.Vec.with_capacity RawKey (alloc.vec.Vec.len rows.val[pc.val])) 0#usize
      ⦃ result => result.val = rows.val[pc.val].val.map rawKey ⦄ := by
  unfold destination_rows.keys_loop0_loop0
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec RawKey × Usize => rows.val[pc.val].val.length - state.2.val)
    (fun state : alloc.vec.Vec RawKey × Usize =>
      state.2.val ≤ rows.val[pc.val].val.length ∧
      state.1.val = (rows.val[pc.val].val.take state.2.val).map rawKey)
  · rintro ⟨result, position⟩ ⟨positionBound, resultEq⟩
    dsimp only at positionBound resultEq
    unfold destination_rows.keys_loop0_loop0.body
    step with Slice.index_usize_spec rows pc inside as ⟨registers, registersEq⟩
    by_cases within : position.val < registers.val.length
    · have test : position < alloc.vec.Vec.len registers := by scalar_tac
      simp only [test, ite_true, Array.to_slice, Std.Array.empty, Std.Array.from_val, bind_ok, lift]
      step with alloc.vec.Vec.index_usize_spec registers position within as ⟨register, registerEq⟩
      step with rust_raw_key register as ⟨identity, identityEq⟩
      have sourceWithin : position.val < rows.val[pc.val].val.length := by simpa only [registersEq] using within
      have resultLength : result.val.length = position.val := by
        rw [resultEq, List.length_map, List.length_take, Nat.min_eq_left positionBound]
      step with alloc.vec.Vec.push_spec result identity
        (by rw [resultLength]; have := rows.val[pc.val].property; omega)
        as ⟨updated, updatedEq⟩
      step with Usize.add_spec (x := position) (y := 1#usize) (by scalar_tac)
        as ⟨next, nextValue⟩
      have nextVal : next.val = position.val + 1 := by scalar_tac
      refine ⟨by scalar_tac, ?_, by scalar_tac⟩
      rw [nextVal, updatedEq, resultEq, List.take_succ_eq_append_getElem sourceWithin]
      simp only [List.map_append, List.map_singleton, identityEq, registerEq, registersEq]
    · have test : ¬position < alloc.vec.Vec.len registers := by scalar_tac
      have complete : position.val = rows.val[pc.val].val.length := by rw [registersEq] at within; omega
      simp only [test, ite_false, WP.spec_ok]
      simpa only [complete, List.take_length] using resultEq
  · exact ⟨by scalar_tac, by simp [alloc.vec.Vec.with_capacity]⟩

end Instruction.DestinationRows

namespace Instruction.DestinationRows

@[step]
theorem rust_keys (rows : Slice (alloc.vec.Vec Reg)) :
    destination_rows.keys rows ⦃ result => result = rawRows rows ⦄ := by
  unfold destination_rows.keys destination_rows.keys_loop0
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec (alloc.vec.Vec RawKey) × Usize => rows.val.length - state.2.val)
    (fun state : alloc.vec.Vec (alloc.vec.Vec RawKey) × Usize =>
      state.2.val ≤ rows.val.length ∧ state.1.val = (rows.val.take state.2.val).map (vecMap rawKey))
  · rintro ⟨result, pc⟩ ⟨pcBound, resultEq⟩
    dsimp only at pcBound resultEq
    unfold destination_rows.keys_loop0.body
    by_cases inside : pc.val < rows.val.length
    · have test : pc < Slice.len rows := by scalar_tac
      simp only [test, ite_true]
      step with Slice.index_usize_spec rows pc inside as ⟨registers, registersEq⟩
      rw [registersEq]
      step with rust_key_row rows pc inside as ⟨row, rowEq⟩
      have rowAt : row = vecMap rawKey rows.val[pc.val] := by
        apply alloc.vec.Vec.ext
        unfold vecMap
        rw [alloc.vec.Vec.from_val]
        exact rowEq
      have resultLength : result.val.length = pc.val := by
        rw [resultEq, List.length_map, List.length_take, Nat.min_eq_left pcBound]
      step with alloc.vec.Vec.push_spec result row (by rw [resultLength]; have := rows.property; omega)
        as ⟨updated, updatedEq⟩
      step with Usize.add_spec (x := pc) (y := 1#usize) (by scalar_tac) as ⟨next, nextValue⟩
      have nextVal : next.val = pc.val + 1 := by scalar_tac
      refine ⟨by scalar_tac, ?_, by scalar_tac⟩
      rw [nextVal, updatedEq, resultEq, List.take_succ_eq_append_getElem inside]
      simp only [List.map_append, List.map_singleton, rowAt]
      rfl
    · have test : ¬pc < Slice.len rows := by scalar_tac
      have complete : pc.val = rows.val.length := by omega
      simp only [test, ite_false, WP.spec_ok]
      apply alloc.vec.Vec.ext
      simpa only [complete, List.take_length, rawRows, alloc.vec.Vec.from_val] using resultEq
  · exact ⟨by scalar_tac, by simp [alloc.vec.Vec.with_capacity]⟩

end Instruction.DestinationRows
