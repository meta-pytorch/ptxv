/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteWords

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

theorem WordContains.bound {words : List U64} {register : Nat}
    (written : WordContains words register) : register / 64 < words.length := by
  obtain ⟨word, lookup, _⟩ := written
  by_contra outside
  have missing : words[register / 64]? = none := List.getElem?_eq_none (by omega)
  rw [missing] at lookup
  cases lookup

theorem get_set_self (words : List U64) (position : Nat) (value : U64)
    (inside : position < words.length) : (words.set position value)[position]! = value := by
  simp only [List.getElem!_eq_getElem?_getD, List.getElem?_set_self inside, Option.getD_some]

theorem get_set_other (words : List U64) (position query : Nat) (value : U64)
    (different : position ≠ query) : (words.set position value)[query]! = words[query]! := by
  simp only [List.getElem!_eq_getElem?_getD, List.getElem?_set_ne different]

theorem insert_register_covers (words : List U64) (register : Nat) (mask : U64)
    (inside : register / 64 < words.length)
    (maskValue : mask.bv = ((1#u64).bv <<< (register % 64))) :
    let updated := words.set (register / 64) (words[register / 64]! ||| mask)
    (∀ query, WordContains words query → WordContains updated query) ∧
      WordContains updated register := by
  dsimp only
  have targetBit : (register % 64) < 64 := Nat.mod_lt _ (by decide)
  constructor
  · intro query oldBit
    have queryBound := oldBit.bound
    rw [word_contains_at _ query (by simpa only [List.length_set] using queryBound)]
    by_cases sameWord : query / 64 = register / 64
    · rw [sameWord, get_set_self words (register / 64) _ inside]
      have oldValue := (word_contains_at words query queryBound).mp oldBit
      rw [sameWord] at oldValue
      have queryBit : query % 64 < 64 := Nat.mod_lt _ (by decide)
      change ((words[register / 64]!.bv ||| mask.bv)[query % 64]) = true
      rw [BitVec.getElem_or queryBit]
      simp only [oldValue, Bool.true_or]
    · rw [get_set_other words (register / 64) (query / 64) _ (Ne.symm sameWord)]
      exact (word_contains_at words query queryBound).mp oldBit
  · rw [word_contains_at _ register (by simpa only [List.length_set] using inside),
      get_set_self words (register / 64) _ inside]
    change ((words[register / 64]!.bv ||| mask.bv)[register % 64]) = true
    rw [BitVec.getElem_or targetBit, maskValue, BitVec.getElem_shiftLeft targetBit]
    simp


def OutgoingInvariant (writes : List Usize) (original : List U64)
    (state : alloc.vec.Vec U64 × Usize) : Prop :=
  state.1.val.length = original.length ∧ state.2.val ≤ writes.length ∧
  (∀ register, WordContains original register → WordContains state.1.val register) ∧
  ∀ position < state.2.val, WordContains state.1.val writes[position]!.val

@[step]
theorem rust_outgoing (writes : alloc.vec.Vec (alloc.vec.Vec Usize)) (pc : Usize)
    (incoming : alloc.vec.Vec U64) (pcBound : pc.val < writes.val.length)
    (registerBound : ∀ register ∈ writes.val[pc.val]!.val, register.val / 64 < incoming.val.length) :
    cache_graph.may_write_sets_loop0_loop0 writes pc incoming 0#usize ⦃ outgoing =>
      outgoing.val.length = incoming.val.length ∧
      (∀ register, WordContains incoming.val register → WordContains outgoing.val register) ∧
      ∀ register ∈ writes.val[pc.val]!.val, WordContains outgoing.val register.val ⦄ := by
  unfold cache_graph.may_write_sets_loop0_loop0
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec U64 × Usize => writes.val[pc.val]!.val.length - state.2.val)
    (OutgoingInvariant writes.val[pc.val]!.val incoming.val)
  · rintro ⟨out, index⟩ ⟨sameLength, indexBound, containsIncoming, containsWrites⟩
    dsimp only at sameLength indexBound containsIncoming containsWrites
    unfold cache_graph.may_write_sets_loop0_loop0.body
    step with alloc.vec.Vec.index_usize_spec writes pc pcBound as ⟨destinations, destinationsEq⟩
    have destinationsAt : destinations = writes.val[pc.val]! := by
      rw [destinationsEq, List.Inhabited_getElem_eq_getElem! _ _ pcBound]
    by_cases inside : index.val < destinations.val.length
    · have test : index < alloc.vec.Vec.len destinations := by scalar_tac
      simp only [test, ite_true]
      step with alloc.vec.Vec.index_usize_spec destinations index inside as ⟨register, registerEq⟩
      have registerAt : register = destinations.val[index.val]! := by
        rw [registerEq, List.Inhabited_getElem_eq_getElem! _ _ inside]
      have registerMember : register ∈ writes.val[pc.val]!.val := by
        have member : register ∈ destinations.val := by rw [registerEq]; exact List.getElem_mem inside
        simpa only [destinationsAt] using member
      have rowInside : index.val < writes.val[pc.val]!.val.length := by
        simpa only [destinationsAt] using inside
      have registerWord : register.val / 64 < out.val.length := by
        rw [sameLength]
        exact registerBound register registerMember
      step as ⟨shift, shiftEq⟩
      step with U64.ShiftLeft_spec (1#u64) shift (by omega) as ⟨mask, _, maskValue⟩
      step as ⟨word, wordEq⟩
      step with alloc.vec.Vec.index_mut_usize_spec out word (by simpa only [wordEq] using registerWord)
        as ⟨old, restore, oldEq, restoreEq⟩
      simp only [lift, bind_ok]
      step with Usize.add_spec (x := index) (y := 1#usize) (by scalar_tac) as ⟨next, nextEq⟩
      have oldAt : old = out.val[register.val / 64]! := by
        rw [oldEq, List.Inhabited_getElem_eq_getElem! _ _ (by simpa only [wordEq] using registerWord), wordEq]
      have updateFacts := insert_register_covers out.val register.val mask registerWord
        (by simpa only [shiftEq] using maskValue)
      refine ⟨?_, ?_⟩
      · simp only [OutgoingInvariant, restoreEq, alloc.vec.Vec.set_val_eq, List.length_set]
        refine ⟨sameLength, by omega, ?_, ?_⟩
        · intro query oldBit
          simpa only [wordEq, oldAt] using updateFacts.1 query (containsIncoming query oldBit)
        · intro position positionBound
          by_cases current : position = index.val
          · subst position
            simpa only [wordEq, oldAt, registerAt, destinationsAt] using updateFacts.2
          · have oldBit := containsWrites position (by omega)
            simpa only [wordEq, oldAt] using updateFacts.1 _ oldBit
      · omega
    · have test : ¬index < alloc.vec.Vec.len destinations := by scalar_tac
      simp only [test, ite_false, WP.spec_ok]
      refine ⟨sameLength, containsIncoming, ?_⟩
      intro register member
      obtain ⟨position, positionBound, atPosition⟩ := List.getElem_of_mem member
      have outside : ¬index.val < writes.val[pc.val]!.val.length := by
        simpa only [destinationsAt] using inside
      have written := containsWrites position (by omega)
      simpa only [←List.Inhabited_getElem_eq_getElem! _ _ positionBound, atPosition] using written
  · exact ⟨rfl, by scalar_tac, fun _ bit => bit, by intro position positionBound; scalar_tac⟩

end MemoryWrite.MayWrite
