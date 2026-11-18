/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteAnalysis

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

theorem successful_find_pc_some (rows : Slice cache_graph.PcWrites) (pc index : Usize)
    (success : cache_graph.find_pc rows pc = .ok (some index)) :
    index.val < rows.val.length ∧ rows.val[index.val]!.pc = pc := by
  unfold cache_graph.find_pc at success
  obtain ⟨left, searched, tailOk⟩ := MemoryWrite.bind_success _ _ _ success
  by_cases inside : left.val < rows.val.length
  · have test : left < Slice.len rows := by scalar_tac
    simp only [test, ite_true] at tailOk
    obtain ⟨entry, indexed, completed⟩ := MemoryWrite.bind_success _ _ _ tailOk
    have entrySpec := Slice.index_usize_spec rows left inside
    simp only [indexed, WP.spec_ok] at entrySpec
    have entryAt : entry = rows.val[left.val]! := by
      rw [entrySpec, List.Inhabited_getElem_eq_getElem! _ _ inside]
    split at completed
    · rename_i matched
      have leftEq : left = index := Option.some.inj (Result.ok_injective completed)
      subst left
      exact ⟨inside, by simpa only [entryAt] using matched⟩
    · have impossible := Result.ok_injective completed
      cases impossible
  · have test : ¬left < Slice.len rows := by scalar_tac
    simp only [test, ite_false] at tailOk
    have impossible := Result.ok_injective tailOk
    cases impossible

theorem low_mask_one (word : U64) (set : word.bv[0] = true) : word &&& 1#u64 = 1#u64 := by
  have bits : U64.bv (word &&& 1#u64) = U64.bv (1#u64) := by
    change word.bv &&& U64.bv (1#u64) = U64.bv (1#u64)
    apply BitVec.eq_of_getElem_eq
    intro index inside
    rw [BitVec.getElem_and inside]
    by_cases zero : index = 0
    · subst index
      simp only [set, Bool.true_and]
    · have oneFalse : (U64.bv (1#u64))[index] = false := by
        change (1#64)[index] = false
        simp only [BitVec.getElem_one inside, zero, decide_false]
      simp only [oneFalse, Bool.and_false]
  exact U64.bv_eq_imp_eq _ _ bits

theorem successful_may_at_true (sets : cache_graph.MayWriteSets) (pc register : Usize)
    (covers : ∀ row ∈ sets.rows.val, row.pc = pc → WordContains row.words.val register.val)
    (answer : Bool) (success : cache_graph.may_at sets pc (some register) = .ok answer) :
    answer = true := by
  unfold cache_graph.may_at at success
  obtain ⟨found, searched, tailOk⟩ := MemoryWrite.bind_success _ _ _ success
  cases found with
  | none =>
      exact (Result.ok_injective tailOk).symm
  | some position =>
      have foundSpec := successful_find_pc_some (alloc.vec.Vec.deref sets.rows) pc position searched
      simp only [alloc.vec.Vec.deref, Slice.from_val] at foundSpec
      obtain ⟨row, indexed, wordsOk⟩ := MemoryWrite.bind_success _ _ _ tailOk
      have rowSpec := rust_vec_index sets.rows position foundSpec.1
      simp only [indexed, WP.spec_ok] at rowSpec
      have rowAt : row = sets.rows.val[position.val]! := by
        rw [rowSpec, List.Inhabited_getElem_eq_getElem! _ _ foundSpec.1]
      have member : row ∈ sets.rows.val := by rw [rowSpec]; exact List.getElem_mem foundSpec.1
      have rowPc : row.pc = pc := by simpa only [rowAt] using foundSpec.2
      have bit := covers row member rowPc
      obtain ⟨wordIndex, divided, testOk⟩ := MemoryWrite.bind_success _ _ _ wordsOk
      have wordSpec := Usize.div_spec (x := register) (y := 64#usize) (by scalar_tac)
      simp only [divided, WP.spec_ok] at wordSpec
      have wordValue : wordIndex.val = register.val / 64 := wordSpec
      have inside : wordIndex.val < row.words.val.length := by rw [wordValue]; exact bit.bound
      have test : ¬wordIndex ≥ alloc.vec.Vec.len row.words := by scalar_tac
      simp only [test, ite_false] at testOk
      obtain ⟨word, indexedWord, shiftedOk⟩ := MemoryWrite.bind_success _ _ _ testOk
      have wordAt := rust_vec_index row.words wordIndex inside
      simp only [indexedWord, WP.spec_ok] at wordAt
      have wordEqual : word = row.words.val[register.val / 64]! := by
        rw [wordAt, List.Inhabited_getElem_eq_getElem! _ _ inside, wordValue]
      have setBit : word.bv[register.val % 64] = true := by
        simpa only [wordEqual] using (word_contains_at row.words.val register.val bit.bound).mp bit
      obtain ⟨shift, remainder, queryOk⟩ := MemoryWrite.bind_success _ _ _ shiftedOk
      have shiftSpec := Usize.rem_spec (x := register) (y := 64#usize) (by scalar_tac)
      simp only [remainder, WP.spec_ok] at shiftSpec
      have shiftValue : shift.val = register.val % 64 := shiftSpec
      obtain ⟨shifted, shiftedSource, returned⟩ := MemoryWrite.bind_success _ _ _ queryOk
      have shiftedSpec := U64.ShiftRight_spec word shift (by omega)
      simp only [shiftedSource, WP.spec_ok] at shiftedSpec
      have lowBit : shifted.bv[0] = true := by
        rw [shiftedSpec.2]
        change (word.bv >>> shift.val)[0] = true
        rw [BitVec.getElem_ushiftRight word.bv 0 shift.val (by decide : 0 < 64)]
        simpa only [shiftValue, Nat.add_zero, BitVec.getLsbD_eq_getElem,
          Nat.mod_lt _ (by decide : 0 < 64)] using setBit
      have masked := low_mask_one shifted lowBit
      simp only [lift, bind_ok, masked, decide_true] at returned
      exact (Result.ok_injective returned).symm

theorem may_write_sets_query_catches_every_path_write (graph : cache_graph.MayWriteGraph)
    (start pc register : Usize) (bit : U32) (bounded : GraphBounded graph)
    (startBound : start.val < graph.reach.val.length) (sets : cache_graph.MayWriteSets)
    (analysis : cache_graph.may_write_sets graph start bit = .ok sets)
    (written : WrittenBefore (graphView graph bit) start.val pc.val register.val)
    (answer : Bool) (query : cache_graph.may_at sets pc (some register) = .ok answer) :
    answer = true := by
  apply successful_may_at_true sets pc register _ answer query
  intro row member atPc
  exact successful_analysis_covers_path_write graph start bit bounded startBound sets analysis
    pc.val register.val written row member (congrArg UScalar.val atPc)

end MemoryWrite.MayWrite
