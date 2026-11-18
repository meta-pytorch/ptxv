/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteSlots
import MemoryWrite.SuccessfulPrimitives

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

def RowsCoherent (sets : cache_graph.WorkSets) (rows : List cache_graph.PcWrites) : Prop :=
  ∀ row ∈ rows, ∃ words, WordsAt sets row.pc.val words ∧ row.words.val = words.val

theorem words_at_page_slot (sets : cache_graph.WorkSets) (entry : cache_graph.PageIndex)
    (member : entry ∈ sets.directory.val) (slots : alloc.vec.Vec (Option (alloc.vec.Vec U64)))
    (slotsLookup : sets.pages.val[entry.storage.val]? = some slots)
    (offset : Usize) (inside : offset.val < 64) (words : alloc.vec.Vec U64)
    (wordLookup : slots.val[offset.val]? = some (some words)) :
    WordsAt sets (entry.page.val * 64 + offset.val) words := by
  have pageValue : (entry.page.val * 64 + offset.val) / 64 = entry.page.val := by omega
  have offsetValue : (entry.page.val * 64 + offset.val) % 64 = offset.val := by omega
  exact ⟨entry.storage, ⟨entry, member, pageValue.symm, rfl⟩, slots, slotsLookup,
    by simpa only [offsetValue] using wordLookup⟩

theorem successful_compact_page (sets : cache_graph.WorkSets) (invariant : DirectoryInvariant sets)
    (entry : cache_graph.PageIndex) (member : entry ∈ sets.directory.val)
    (rows result : alloc.vec.Vec cache_graph.PcWrites) (coherent : RowsCoherent sets rows.val)
    (success : cache_graph.compact_loop0_loop0 sets.pages rows entry 0#usize = .ok result) :
    RowsCoherent sets result.val := by
  unfold cache_graph.compact_loop0_loop0 at success
  have execution := MemoryWrite.successful_loop_has_execution _ _ _ success
  apply execution.preserves
    (fun state : alloc.vec.Vec cache_graph.PcWrites × Usize =>
      state.2.val ≤ 64 ∧ RowsCoherent sets state.1.val)
    (fun result => RowsCoherent sets result.val)
  · exact ⟨by scalar_tac, coherent⟩
  · rintro ⟨currentRows, offset⟩ ⟨offsetBound, currentCoherent⟩ flow bodyOk
    dsimp only at offsetBound currentCoherent
    unfold cache_graph.compact_loop0_loop0.body cache_graph.PAGE_SIZE at bodyOk
    by_cases inside : offset.val < 64
    · have test : offset < 64#usize := by scalar_tac
      simp only [test, ite_true] at bodyOk
      have storageBound := invariant.safe entry member
      obtain ⟨slots, slotsOk, lookupOk⟩ := MemoryWrite.bind_success _ _ _ bodyOk
      have slotsEq := rust_vec_index sets.pages entry.storage storageBound
      simp only [slotsOk, WP.spec_ok] at slotsEq
      have slotsAt : sets.pages.val[entry.storage.val]? = some slots := by
        rw [List.getElem?_eq_getElem storageBound, ←slotsEq]
      have slotsMember : slots ∈ sets.pages.val := by rw [slotsEq]; exact List.getElem_mem storageBound
      have slotsSize := invariant.sizes slots slotsMember
      have offsetInside : offset.val < slots.val.length := by omega
      obtain ⟨previous, previousOk, addedOk⟩ := MemoryWrite.bind_success _ _ _ lookupOk
      have previousEq := rust_vec_index slots offset offsetInside
      simp only [previousOk, WP.spec_ok] at previousEq
      obtain ⟨nextRows, added, incrementOk⟩ := MemoryWrite.bind_success _ _ _ addedOk
      have nextCoherent : RowsCoherent sets nextRows.val := by
        cases previous with
        | none =>
            have rowsEq : currentRows = nextRows := Result.ok_injective added
            simpa only [←rowsEq] using currentCoherent
        | some words =>
            obtain ⟨base, multiplied, addressOk⟩ := MemoryWrite.bind_success _ _ _ added
            have baseValue : base.val = entry.page.val * 64 := MemoryWrite.successful_mul _ _ _ multiplied
            obtain ⟨address, addressed, copiedOk⟩ := MemoryWrite.bind_success _ _ _ addressOk
            have addressValue : address.val = entry.page.val * 64 + offset.val := by
              have value := MemoryWrite.successful_add _ _ _ addressed
              simpa only [baseValue] using value
            obtain ⟨copy, copied, pushed⟩ := MemoryWrite.bind_success _ _ _ copiedOk
            have copyOk : cache_graph.copy_words (alloc.vec.Vec.deref words) = .ok copy := copied
            have copySpec := rust_copy_words (alloc.vec.Vec.deref words)
            simp only [copyOk, WP.spec_ok] at copySpec
            simp only [alloc.vec.Vec.deref, Slice.from_val] at copySpec
            have rowsEq := MemoryWrite.successful_vec_push currentRows (⟨address, copy⟩ : cache_graph.PcWrites) nextRows pushed
            intro row rowMember
            rw [rowsEq, List.mem_append, List.mem_singleton] at rowMember
            rcases rowMember with old | same
            · exact currentCoherent row old
            · subst row
              refine ⟨words, ?_, copySpec⟩
              have wordLookup : slots.val[offset.val]? = some (some words) := by
                rw [List.getElem?_eq_getElem offsetInside, ←previousEq]
              simpa only [addressValue] using words_at_page_slot sets entry member slots slotsAt offset inside words wordLookup
      obtain ⟨nextOffset, increment, completed⟩ := MemoryWrite.bind_success _ _ _ incrementOk
      have nextValue := MemoryWrite.successful_add offset (1#usize) nextOffset increment
      have flowEq : ControlFlow.cont (nextRows, nextOffset) = flow := Result.ok_injective completed
      rw [←flowEq]
      exact ⟨by change nextOffset.val ≤ 64; scalar_tac, nextCoherent⟩
    · have test : ¬offset < 64#usize := by scalar_tac
      simp only [test, ite_false] at bodyOk
      have flowEq : ControlFlow.done currentRows = flow := Result.ok_injective bodyOk
      rw [←flowEq]
      exact currentCoherent

theorem successful_compact (sets : cache_graph.WorkSets) (invariant : DirectoryInvariant sets)
    (result : cache_graph.MayWriteSets) (success : cache_graph.compact sets = .ok result) :
    RowsCoherent sets result.rows.val := by
  unfold cache_graph.compact at success
  obtain ⟨rows, rowsOk, completed⟩ := MemoryWrite.bind_success _ _ _ success
  have resultEq : (⟨rows⟩ : cache_graph.MayWriteSets) = result := Result.ok_injective completed
  rw [←resultEq]
  unfold cache_graph.compact_loop0 at rowsOk
  have execution := MemoryWrite.successful_loop_has_execution _ _ _ rowsOk
  apply execution.preserves
    (fun state : alloc.vec.Vec cache_graph.PcWrites × Usize => RowsCoherent sets state.1.val)
    (fun result => RowsCoherent sets result.val)
  · intro row member
    cases member
  · rintro ⟨currentRows, page⟩ coherent flow bodyOk
    unfold cache_graph.compact_loop0.body at bodyOk
    by_cases inside : page.val < sets.directory.val.length
    · have test : page < alloc.vec.Vec.len sets.directory := by scalar_tac
      simp only [test, ite_true] at bodyOk
      obtain ⟨entry, entryOk, pageOk⟩ := MemoryWrite.bind_success _ _ _ bodyOk
      have entryEq := rust_vec_index sets.directory page inside
      simp only [entryOk, WP.spec_ok] at entryEq
      have entryMember : entry ∈ sets.directory.val := by rw [entryEq]; exact List.getElem_mem inside
      obtain ⟨nextRows, nextRowsOk, incrementOk⟩ := MemoryWrite.bind_success _ _ _ pageOk
      have nextCoherent := successful_compact_page sets invariant entry entryMember currentRows nextRows coherent nextRowsOk
      obtain ⟨nextPage, increment, finished⟩ := MemoryWrite.bind_success _ _ _ incrementOk
      have flowEq : ControlFlow.cont (nextRows, nextPage) = flow := Result.ok_injective finished
      rw [←flowEq]
      exact nextCoherent
    · have test : ¬page < alloc.vec.Vec.len sets.directory := by scalar_tac
      simp only [test, ite_false] at bodyOk
      have flowEq : ControlFlow.done currentRows = flow := Result.ok_injective bodyOk
      rw [←flowEq]
      exact coherent

end MemoryWrite.MayWrite
