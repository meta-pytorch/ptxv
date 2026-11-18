/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteMainTotal
import MemoryWrite.MayWriteCompactTotal
import MemoryWrite.MayWriteAnalysis
import MemoryWrite.MayWriteQuery

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace MemoryWrite.MayWrite

@[step]
theorem rust_setup_total (graph : Graph) (count : Nat) (start words : Usize)
    (startBound : start.val < count) :
    (Std.bind (cache_graph.ensure_pc emptySets start words) (fun (_, sets) =>
      Std.bind (cache_graph.push_work (alloc.vec.Vec.new Usize) 0#usize start)
        (fun (pending, work) => .ok (sets, work, pending))))
      ⦃ result => SetupPost graph count start words result.1 result.2.1 result.2.2 ∧ result.2.2.val = 1 ⦄ := by
  obtain ⟨⟨answer, sets⟩, initialized, _⟩ := MemoryWrite.spec_returns _ _
    (rust_ensure_pc_resourced emptySets start words empty_directory empty_resources)
  obtain ⟨⟨pending, work⟩, pushed, pushPost⟩ := MemoryWrite.spec_returns _ _
    (rust_push_work (alloc.vec.Vec.new Usize) 0#usize start
      (by simp [alloc.vec.Vec.new, alloc.vec.Vec.from_val]) (by scalar_tac))
  simp only [initialized, pushed, bind_ok, WP.spec_ok]
  exact ⟨successful_setup graph count start words startBound answer sets pending work initialized pushed,
    by simpa using pushPost.1⟩

def AnalysisFits (graph : cache_graph.MayWriteGraph) : Prop :=
  graph.reach.val.length * (wordCount graph.register_count.val * 64 + 1) + 1 < Usize.max

@[step]
theorem rust_may_write_sets_total (graph : cache_graph.MayWriteGraph) (start : Usize) (bit : U32)
    (bounded : GraphBounded graph) (startBound : start.val < graph.reach.val.length)
    (fits : AnalysisFits graph) :
    cache_graph.may_write_sets graph start bit ⦃ result => AnalysisPost graph start bit result ⦄ := by
  unfold cache_graph.may_write_sets
  step with Usize.div_spec (x := graph.register_count) (y := 64#usize) (by scalar_tac)
    as ⟨quotient, quotientSpec⟩
  step with core.num.Usize.is_multiple_of.spec graph.register_count (64#usize)
    as ⟨multiple, multipleSpec⟩
  have divisorValue : (64#usize).val = 64 := by scalar_tac
  have quotientValue : quotient.val = graph.register_count.val / 64 := by
    simpa only [divisorValue] using quotientSpec
  have multipleValue : multiple = decide (graph.register_count.val % 64 = 0) := by
    apply Bool.eq_iff_iff.mpr
    simpa only [divisorValue, decide_eq_true_eq] using Iff.of_eq multipleSpec
  simp only [lift, bind_ok]
  have boolNot : decide (¬multiple = true) = !multiple := by cases multiple <;> rfl
  rw [boolNot]
  have sumValue : quotient.val + (core.convert.num.FromUsizeBool.from (!multiple)).val =
      wordCount graph.register_count.val := by
    rw [quotientValue, bool_to_usize_value, multipleValue]
    by_cases divisible : graph.register_count.val % 64 = 0 <;> simp [wordCount, divisible]
  step with Usize.add_spec (x := quotient) (y := core.convert.num.FromUsizeBool.from (!multiple))
    (by rw [sumValue]; exact word_count_fits graph.register_count) as ⟨words, wordsValue⟩
  have wordCountValue : words.val = wordCount graph.register_count.val := wordsValue.trans sumValue
  have startTest : start < alloc.vec.Vec.len graph.reach := by scalar_tac
  simp only [startTest, ite_true]
  obtain ⟨⟨initialSets, initialWork, initialPending⟩, prepared, setup, pendingOne⟩ :=
    MemoryWrite.spec_returns _ _
      (rust_setup_total (graphView graph bit) graph.reach.val.length start words startBound)
  obtain ⟨⟨answer, actualSets⟩, initialized, afterInit⟩ := MemoryWrite.bind_success _ _ _ prepared
  obtain ⟨⟨actualPending, actualWork⟩, pushed, completed⟩ := MemoryWrite.bind_success _ _ _ afterInit
  have same : (actualSets, actualWork, actualPending) = (initialSets, initialWork, initialPending) :=
    Result.ok_injective completed
  have setsEq : actualSets = initialSets := congrArg Prod.fst same
  have workEq : actualWork = initialWork := congrArg (fun state => state.2.1) same
  have pendingEq : actualPending = initialPending := congrArg (fun state => state.2.2) same
  subst actualSets actualWork actualPending
  simp only [emptySets] at initialized
  simp only [initialized, pushed, bind_ok]
  have space : rank graph.reach.val.length (words.val * 64) (workSolution initialSets) initialPending.val < Usize.max := by
    unfold rank
    rw [pendingOne, wordCountValue]
    unfold AnalysisFits at fits
    omega
  step with rust_main_loop_total graph bit words initialSets initialWork initialPending bounded
    wordCountValue setup.storage setup.settled space as ⟨workspace, main⟩
  step with rust_compact_total workspace main.directory graph.reach.val.length graph.reach.property main.knownBound
    as ⟨result, coherent⟩
  exact ⟨workspace, main.directory, main.closed, main.growth.1 start.val setup.startKnown, coherent⟩


theorem may_write_sets_returns_and_catches_every_path_write
    (graph : cache_graph.MayWriteGraph) (start : Usize) (bit : U32)
    (bounded : GraphBounded graph) (startBound : start.val < graph.reach.val.length)
    (fits : AnalysisFits graph) :
    ∃ sets, cache_graph.may_write_sets graph start bit = .ok sets ∧
      ∀ (pc register : Usize), WrittenBefore (graphView graph bit) start.val pc.val register.val →
        ∀ answer, cache_graph.may_at sets pc (some register) = .ok answer → answer = true := by
  obtain ⟨sets, executed, _⟩ := MemoryWrite.spec_returns _ _
    (rust_may_write_sets_total graph start bit bounded startBound fits)
  exact ⟨sets, executed, fun pc register written answer queried =>
    may_write_sets_query_catches_every_path_write graph start pc register bit bounded startBound
      sets executed written answer queried⟩

end MemoryWrite.MayWrite
