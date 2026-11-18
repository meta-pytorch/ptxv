/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.SourceCoordinates
import FrontendProgram.SourceRegisters

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace PtxFrontendProgram

private def ThreadPrefix (manifest : Manifest) (virtuals predicates count : Nat)
    (threads : List ptxv_memory_write.execution_core.ThreadState) : Prop :=
  threads.length = count ∧ ∀ index thread, threads[index]? = some thread →
    InitialThread manifest virtuals predicates index thread

private theorem prefix_append (manifest : Manifest) (virtuals predicates count : Nat)
    (threads : List ptxv_memory_write.execution_core.ThreadState)
    (thread : ptxv_memory_write.execution_core.ThreadState)
    (inventory : ThreadPrefix manifest virtuals predicates count threads)
    (next : InitialThread manifest virtuals predicates count thread) :
    ThreadPrefix manifest virtuals predicates (count + 1) (threads ++ [thread]) := by
  refine ⟨by simpa using inventory.1, ?_⟩
  have length : threads.length = count := inventory.1
  intro index selected lookup
  by_cases inside : index < threads.length
  · apply inventory.2 index selected
    simpa only [List.getElem?_append, inside, ite_true] using lookup
  · have tail : [thread][index-threads.length]? = some selected := by
      simpa only [List.getElem?_append, inside, ite_false] using lookup
    cases difference : index-threads.length with
    | zero =>
      have atEnd : index = count := by omega
      have equal : thread = selected := by simpa only [difference, List.getElem?_cons_zero, Option.some.injEq] using tail
      simpa only [←equal, atEnd] using next
    | succ rest => simp only [difference, List.getElem?_cons_succ, List.getElem?_nil] at tail; cases tail

private theorem threads_loop (manifest : Manifest) (virtuals predicates : U16)
    (gridCount blockCount count : U64)
    (gridComputed : strict_memory_frontend_launch.dimensions_product manifest.grid = .ok (some gridCount))
    (blockComputed : strict_memory_frontend_launch.dimensions_product manifest.block = .ok (some blockCount))
    (countEq : count.val = gridCount.val * blockCount.val) (countBound : count.val ≤ 1048576)
    (threads : alloc.vec.Vec ptxv_memory_write.execution_core.ThreadState)
    (index : U64) (bound : index.val ≤ count.val)
    (inventory : ThreadPrefix manifest virtuals.val predicates.val index.val threads.val) :
    strict_memory_frontend_launch.initial_threads_loop manifest.grid manifest.block virtuals predicates
      blockCount count threads index true
      ⦃ output => output.2 = true ∧ ThreadPrefix manifest virtuals.val predicates.val count.val output.1.val ⦄ := by
  obtain ⟨gridVal, gridPositive⟩ := source_dimensions_product manifest.grid gridCount gridComputed
  obtain ⟨blockVal, blockPositive⟩ := source_dimensions_product manifest.block blockCount blockComputed
  have positiveBlock : 0 < blockCount.val := by
    rw [blockVal]
    rcases blockPositive with ⟨x,y,z⟩
    dsimp [PtxFrontend.Dimensions.size]
    positivity
  unfold strict_memory_frontend_launch.initial_threads_loop
  apply loop.spec_decr_nat
    (fun current => count.val - current.2.1.val)
    (fun current => current.2.2 = true ∧ current.2.1.val ≤ count.val ∧
      ThreadPrefix manifest virtuals.val predicates.val current.2.1.val current.1.val)
  · rintro ⟨threads,index,valid⟩ ⟨validEq,bound,inventory⟩
    have validValue : valid = true := validEq
    subst valid
    have threadLength : threads.val.length = index.val := inventory.1
    unfold strict_memory_frontend_launch.initial_threads_loop.body
    simp only [ite_true]
    by_cases inside : index < count
    · simp only [inside, ite_true]
      step as ⟨ctaIndex, ctaIndexVal⟩
      have ctaBound : ctaIndex.val < gridCount.val := by
        rw [ctaIndexVal]
        exact (Nat.div_lt_iff_lt_mul positiveBlock).2 (by scalar_tac)
      obtain ⟨ctaid, ctaComputed, ctaMeaning⟩ :=
        source_coordinates_exists ctaIndex manifest.grid gridCount gridComputed ctaBound
      rw [ctaComputed]
      simp only [bind_ok]
      step as ⟨tidIndex, tidIndexVal⟩
      have tidBound : tidIndex.val < blockCount.val := by rw [tidIndexVal]; exact Nat.mod_lt _ positiveBlock
      obtain ⟨tid, tidComputed, tidMeaning⟩ :=
        source_coordinates_exists tidIndex manifest.block blockCount blockComputed tidBound
      rw [tidComputed]
      simp only [bind_ok]
      step with source_empty_registers virtuals predicates as ⟨registers, registersMeaning⟩
      step as ⟨lane, laneVal⟩
      simp only [lift, bind_ok]
      step as ⟨nextThreads, appended⟩
      step as ⟨nextIndex, nextIndexVal⟩
      refine ⟨?_, ?_, ?_⟩
      · scalar_tac
      · rw [nextIndexVal, appended]
        apply prefix_append _ _ _ _ _ _ inventory
        simp only [InitialThread]
        refine ⟨rfl,True.intro,registersMeaning,True.intro,True.intro,?_,?_,?_⟩
        · simpa only [ctaIndexVal, blockVal] using ctaMeaning
        · simpa only [tidIndexVal, blockVal] using tidMeaning
        · scalar_tac
      · scalar_tac
    · simp only [inside, ite_false, WP.spec_ok]
      have atEnd : index.val = count.val := by scalar_tac
      exact ⟨True.intro, by simpa only [atEnd] using inventory⟩
  · exact ⟨rfl,bound,inventory⟩

private theorem reject_none {α : Type} (condition : Prop) [Decidable condition]
    (next : Result (Option α)) (result : α)
    (computed : (if condition then .ok none else next) = .ok (some result)) :
    ¬condition ∧ next = .ok (some result) := by
  by_cases h : condition
  · simp only [h,ite_true,ok.injEq] at computed
    cases computed
  · exact ⟨h,by simpa only [h,ite_false] using computed⟩

private theorem option_bind_success {α β : Type} (computation : Result (Option α))
    (next : Option α → Result (Option β)) (result : β) (missing : next none = .ok none)
    (computed : Std.bind computation next = .ok (some result)) :
    ∃ input, computation = .ok (some input) ∧ next (some input) = .ok (some result) := by
  obtain ⟨value, first, rest⟩ := source_bind_success _ _ _ computed
  cases value with
  | none => rw [missing] at rest; simp at rest
  | some input => exact ⟨input,first,rest⟩

theorem source_initial_threads (manifest : Manifest) (virtuals predicates : U16)
    (threads : alloc.vec.Vec ptxv_memory_write.execution_core.ThreadState)
    (computed : strict_memory_frontend_launch.initial_threads manifest virtuals predicates = .ok (some threads)) :
    ThreadsMeaning manifest virtuals.val predicates.val threads.val := by
  unfold strict_memory_frontend_launch.initial_threads at computed
  obtain ⟨gx, _, computed⟩ := source_bind_success _ _ _ computed
  obtain ⟨_, computed⟩ := reject_none (gx > 2147483647#u32) _ _ computed
  obtain ⟨gy, _, computed⟩ := source_bind_success _ _ _ computed
  obtain ⟨_, computed⟩ := reject_none (gy > 65535#u32) _ _ computed
  obtain ⟨gz, _, computed⟩ := source_bind_success _ _ _ computed
  obtain ⟨_, computed⟩ := reject_none (gz > 65535#u32) _ _ computed
  obtain ⟨bx, _, computed⟩ := source_bind_success _ _ _ computed
  obtain ⟨_, computed⟩ := reject_none (bx > 1024#u32) _ _ computed
  obtain ⟨byDim, _, computed⟩ := source_bind_success _ _ _ computed
  obtain ⟨_, computed⟩ := reject_none (byDim > 1024#u32) _ _ computed
  obtain ⟨bz, _, computed⟩ := source_bind_success _ _ _ computed
  obtain ⟨_, computed⟩ := reject_none (bz > 64#u32) _ _ computed
  obtain ⟨gridCount,gridComputed,computed⟩ := option_bind_success
    (strict_memory_frontend_launch.dimensions_product manifest.grid) _ threads (by rfl) computed
  dsimp only at computed
  obtain ⟨blockCount,blockComputed,computed⟩ := option_bind_success
    (strict_memory_frontend_launch.dimensions_product manifest.block) _ threads (by rfl) computed
  dsimp only at computed
  try simp only [lift,bind_ok] at computed
  obtain ⟨count,countComputed⟩ : ∃ count, U64.checked_mul gridCount blockCount = some count := by
    cases found : U64.checked_mul gridCount blockCount with
    | none => simp only [found] at computed; simp at computed
    | some value => exact ⟨value,rfl⟩
  simp only [countComputed] at computed
  have multiplication := U64.checked_mul_bv_spec gridCount blockCount
  rw [countComputed] at multiplication
  have countEq : count.val = gridCount.val * blockCount.val := multiplication.2.1
  obtain ⟨countBound,computed⟩ := reject_none (count > 1048576#u64) _ _ computed
  obtain ⟨_,computed⟩ := reject_none (blockCount > 1024#u64) _ _ computed
  try simp only [lift,bind_ok] at computed
  obtain ⟨wordSum,_,computed⟩ := source_bind_success _ _ _ computed
  obtain ⟨words,_,computed⟩ := source_bind_success _ _ _ computed
  try simp only [lift,bind_ok] at computed
  obtain ⟨budget,budgetComputed⟩ : ∃ budget, U64.checked_mul count words = some budget := by
    cases found : U64.checked_mul count words with
    | none => simp only [found] at computed; simp at computed
    | some value => exact ⟨value,rfl⟩
  simp only [budgetComputed] at computed
  obtain ⟨_,computed⟩ := reject_none (budget > 1048576#u64) _ _ computed
  obtain ⟨result,loopComputed,computed⟩ := source_bind_success _ _ _ computed
  have countNatBound : count.val ≤ 1048576 := by scalar_tac
  have emptyPrefix : ThreadPrefix manifest virtuals.val predicates.val 0
      (alloc.vec.Vec.new ptxv_memory_write.execution_core.ThreadState).val := by
    refine ⟨by simp,?_⟩
    intro index thread impossible
    simp at impossible
  have loopMeaning := threads_loop manifest virtuals predicates gridCount blockCount count
    gridComputed blockComputed countEq countNatBound
    (alloc.vec.Vec.new ptxv_memory_write.execution_core.ThreadState) 0#u64 (by simp) emptyPrefix
  rw [loopComputed] at loopMeaning
  simp only [WP.spec_ok] at loopMeaning
  rcases result with ⟨built,valid⟩
  dsimp only at loopMeaning
  change (if valid = true then Result.ok (some built) else Result.ok none) = Result.ok (some threads) at computed
  rw [loopMeaning.1] at computed
  have same : built = threads := by simpa using computed
  subst threads
  obtain ⟨gridVal,gridPositive⟩ := source_dimensions_product manifest.grid gridCount gridComputed
  obtain ⟨blockVal,blockPositive⟩ := source_dimensions_product manifest.block blockCount blockComputed
  dsimp only [ThreadsMeaning]
  refine ⟨gridPositive,blockPositive,?_,loopMeaning.2.2⟩
  calc
    built.val.length = count.val := loopMeaning.2.1
    _ = (dimensionsModel manifest.grid).size * (dimensionsModel manifest.block).size := by
      rw [countEq,gridVal,blockVal]

end PtxFrontendProgram
