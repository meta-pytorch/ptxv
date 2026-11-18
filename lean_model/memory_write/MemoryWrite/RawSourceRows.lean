/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.SourceRows
import MemoryWrite.TotalPrimitives

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.SourceRows

abbrev RawKey := U32 × U8 × U16

def keyOfRaw (raw : RawKey) : symbolic.RegisterKey := ⟨raw.1, raw.2.1, raw.2.2⟩

def vecMap {α β : Type} (f : α → β) (values : alloc.vec.Vec α) : alloc.vec.Vec β :=
  alloc.vec.Vec.from (values.val.map f) (by simpa only [List.length_map] using values.property)

def keyRows (rows : alloc.vec.Vec (alloc.vec.Vec RawKey)) :
    alloc.vec.Vec (alloc.vec.Vec symbolic.RegisterKey) := vecMap (vecMap keyOfRaw) rows

abbrev rawReader := source_rows.RawRows.Insts.Ptxv_memory_writeSource_rowsRegisterRows

theorem raw_reader_spec (rows : alloc.vec.Vec (alloc.vec.Vec RawKey)) :
    ReaderSpec rawReader ⟨rows⟩ (keyRows rows) := by
  constructor
  · unfold rawReader source_rows.RawRows.Insts.Ptxv_memory_writeSource_rowsRegisterRows
    simp only [source_rows.RawRows.Insts.Ptxv_memory_writeSource_rowsRegisterRows.row_count]
    congr 1
    apply UScalar.eq_of_val_eq
    simp [keyRows, vecMap]
  · intro pc inside
    have bound : pc.val < rows.val.length := by
      simpa only [keyRows, vecMap, alloc.vec.Vec.from_val, List.length_map] using inside
    obtain ⟨row, read, meaning⟩ := MemoryWrite.spec_returns _ _
      (alloc.vec.Vec.index_usize_spec rows pc bound)
    simp only [source_rows.RawRows.Insts.Ptxv_memory_writeSource_rowsRegisterRows.row_len, alloc.vec.Vec.index_slice_index, read, bind_ok]
    congr 1
    apply UScalar.eq_of_val_eq
    simp [meaning, keyRows, vecMap]
  · intro pc position inside within
    have bound : pc.val < rows.val.length := by
      simpa only [keyRows, vecMap, alloc.vec.Vec.from_val, List.length_map] using inside
    obtain ⟨row, rowOk, rowMeaning⟩ := MemoryWrite.spec_returns _ _
      (alloc.vec.Vec.index_usize_spec rows pc bound)
    have positionBound : position.val < rows.val[pc.val].val.length := by
      simpa only [keyRows, vecMap, alloc.vec.Vec.from_val, List.getElem_map, List.length_map] using within
    have actualBound : position.val < row.val.length := by simpa only [rowMeaning] using positionBound
    obtain ⟨raw, rawOk, rawMeaning⟩ := MemoryWrite.spec_returns _ _
      (alloc.vec.Vec.index_usize_spec row position actualBound)
    simp only [source_rows.RawRows.Insts.Ptxv_memory_writeSource_rowsRegisterRows.key, alloc.vec.Vec.index_slice_index, rowOk, rawOk, bind_ok,
      symbolic.register_key]
    simp only [rawMeaning, rowMeaning, keyRows, vecMap, alloc.vec.Vec.from_val, List.getElem_map, keyOfRaw]
    rcases rows.val[pc.val].val[position.val] with ⟨owner, file, index⟩
    rfl

@[step]
theorem rust_from_raw (rows : alloc.vec.Vec (alloc.vec.Vec RawKey)) :
    source_rows.from_raw rows ⦃ result => result = keyRows rows ⦄ := by
  unfold source_rows.from_raw
  exact rust_collect_exact rawReader ⟨rows⟩ (keyRows rows) (raw_reader_spec rows)

end MemoryWrite.SourceRows
