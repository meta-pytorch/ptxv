/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteSetup
import MemoryWrite.MayWriteCompact

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.MayWrite

def AnalysisPost (graph : cache_graph.MayWriteGraph) (start : Usize) (bit : U32)
    (result : cache_graph.MayWriteSets) : Prop :=
  ∃ workspace : cache_graph.WorkSets,
    DirectoryInvariant workspace ∧ Closed (graphView graph bit) (workSolution workspace) ∧
    (workSolution workspace).present start.val ∧ RowsCoherent workspace result.rows.val

theorem bool_to_usize_value (value : Bool) :
    (core.convert.num.FromUsizeBool.from value).val = if value then 1 else 0 := by
  cases value <;> simp only [core.convert.num.FromUsizeBool.from, Bool.false_eq_true,
    ite_false, ite_true] <;> rfl

theorem successful_may_write_sets (graph : cache_graph.MayWriteGraph) (start : Usize) (bit : U32)
    (bounded : GraphBounded graph) (startBound : start.val < graph.reach.val.length)
    (result : cache_graph.MayWriteSets)
    (success : cache_graph.may_write_sets graph start bit = .ok result) :
    AnalysisPost graph start bit result := by
  unfold cache_graph.may_write_sets at success
  obtain ⟨quotient, divided, multipleOk⟩ := MemoryWrite.bind_success _ _ _ success
  have quotientSpec := Usize.div_spec (x := graph.register_count) (y := 64#usize) (by scalar_tac)
  simp only [divided, WP.spec_ok] at quotientSpec
  have divisorValue : (64#usize).val = 64 := by scalar_tac
  have quotientValue : quotient.val = graph.register_count.val / 64 := by
    simpa only [divisorValue] using quotientSpec
  obtain ⟨multiple, checked, wordsOk⟩ := MemoryWrite.bind_success _ _ _ multipleOk
  have multipleSpec := core.num.Usize.is_multiple_of.spec graph.register_count (64#usize)
  simp only [checked, WP.spec_ok] at multipleSpec
  have multipleValue : multiple = decide (graph.register_count.val % 64 = 0) := by
    apply Bool.eq_iff_iff.mpr
    simpa only [divisorValue, decide_eq_true_eq] using Iff.of_eq multipleSpec
  simp only [lift, bind_ok] at wordsOk
  obtain ⟨words, added, preparedOk⟩ := MemoryWrite.bind_success _ _ _ wordsOk
  have wordsValue := MemoryWrite.successful_add quotient
    (core.convert.num.FromUsizeBool.from (!multiple)) words added
  have wordCountValue : words.val = wordCount graph.register_count.val := by
    rw [wordsValue, quotientValue, bool_to_usize_value, multipleValue]
    by_cases divisible : graph.register_count.val % 64 = 0 <;> simp [wordCount, divisible]
  obtain ⟨prepared, preparedSource, analyzedOk⟩ := MemoryWrite.bind_success _ _ _ preparedOk
  have startTest : start < alloc.vec.Vec.len graph.reach := by scalar_tac
  simp only [startTest, ite_true] at preparedSource
  obtain ⟨⟨answer, initialSets⟩, initializedOk, queuedOk⟩ := MemoryWrite.bind_success _ _ _ preparedSource
  obtain ⟨⟨initialPending, initialWork⟩, pushedOk, completedSetup⟩ := MemoryWrite.bind_success _ _ _ queuedOk
  have setup := successful_setup (graphView graph bit) graph.reach.val.length start words startBound
    answer initialSets initialPending initialWork initializedOk pushedOk
  have preparedEq : (initialSets, initialWork, initialPending) = prepared := Result.ok_injective completedSetup
  subst prepared
  obtain ⟨workspace, analyzed, compacted⟩ := MemoryWrite.bind_success _ _ _ analyzedOk
  have main := successful_main_loop graph bit words initialSets initialWork initialPending bounded
    wordCountValue setup.storage setup.settled workspace analyzed
  have coherent := successful_compact workspace main.directory result compacted
  exact ⟨workspace, main.directory, main.closed, main.growth.1 start.val setup.startKnown, coherent⟩

theorem successful_analysis_covers_path_write (graph : cache_graph.MayWriteGraph) (start : Usize)
    (bit : U32) (bounded : GraphBounded graph) (startBound : start.val < graph.reach.val.length)
    (result : cache_graph.MayWriteSets)
    (success : cache_graph.may_write_sets graph start bit = .ok result)
    (pc register : Nat) (written : WrittenBefore (graphView graph bit) start.val pc register)
    (row : cache_graph.PcWrites) (member : row ∈ result.rows.val) (atPc : row.pc.val = pc) :
    WordContains row.words.val register := by
  obtain ⟨workspace, invariant, closed, initial, coherent⟩ :=
    successful_may_write_sets graph start bit bounded startBound result success
  obtain ⟨words, atWords, bits⟩ := closed_covers_path_writes (graphView graph bit) start.val
    (workSolution workspace) initial closed written
  obtain ⟨emittedWords, emittedAt, sameValues⟩ := coherent row member
  have sameWords := words_at_unique invariant (by simpa only [atPc] using emittedAt) atWords
  simpa only [sameValues, sameWords] using bits

end MemoryWrite.MayWrite
