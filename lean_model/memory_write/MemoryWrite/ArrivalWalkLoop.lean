/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.ArrivalWalkRestart

open Aeneas Aeneas.Std Result PtxvMemoryWrite
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace ArrivalWalk

@[step]
theorem rust_may_arrive_loop {State W : Type} (walker : arrival_walk.Walker W State) (view : W → View State)
    (settled : W → Prop) (contracts : LocalContracts walker view settled) (world : W) (budget : Usize)
    (pending : arrival_walk.Frontier State) (started : Bool)
    (initial : Invariant view world pending started) :
    arrival_walk.may_arrive_loop walker world budget pending started ⦃ possible result =>
      possible = false → settled result ∧ NoHit (view result) ⦄ := by
  unfold arrival_walk.may_arrive_loop
  apply loop.spec_decr_nat
    (fun state : W × Usize × arrival_walk.Frontier State × Bool => state.2.1.val)
    (fun state => Invariant view state.1 state.2.2.1 state.2.2.2)
  · rintro ⟨current, fuel, frontier, begun⟩ ⟨states, rep, covered⟩
    dsimp only at rep covered
    unfold arrival_walk.may_arrive_loop.body
    by_cases live : fuel > 0#usize
    · simp only [live, ite_true]
      step with Usize.sub_spec (x := fuel) (y := 1#usize) (by scalar_tac) as ⟨remaining, remainingEq⟩
      have decreases : remaining.val < fuel.val := by scalar_tac
      rcases empty_or_append states with empty | ⟨prior, state, last⟩
      · subst states
        step with rust_pop_empty frontier rep as ⟨popped, after, poppedEq, afterEq⟩
        simp only [poppedEq]
        have afterRep : Represents after [] := by rw [afterEq]; exact rep
        cases begun with
        | false =>
          simp only [Bool.false_eq_true, ite_false]
          step with rust_restart walker view settled contracts current after afterRep
            as ⟨success, result, next, restarted⟩
          cases success with
          | false => simp
          | true =>
            simp only [ite_true, WP.spec_ok]
            exact ⟨restarted rfl, decreases⟩
        | true =>
          simp only [ite_true] at covered ⊢
          step with contracts.retry current as ⟨again, result, retried⟩
          cases again with
          | false =>
            simp only [Bool.false_eq_true, false_or] at retried
            simp only [Bool.false_eq_true, ite_false, WP.spec_ok]
            intro _
            refine ⟨retried.2, ?_⟩
            rw [retried.1]
            exact Invariant.completed view current [] covered rfl
          | true =>
            simp only [ite_true]
            step with rust_restart walker view settled contracts result after afterRep
              as ⟨success, restartedWorld, next, restarted⟩
            cases success with
            | false => simp
            | true =>
              simp only [ite_true, WP.spec_ok]
              exact ⟨restarted rfl, decreases⟩
      · subst states
        have begunTrue : begun = true := by cases begun <;> simp_all
        simp only [begunTrue, ite_true] at covered ⊢
        step with rust_pop_append frontier prior state rep as ⟨popped, after, poppedEq, afterRep, _⟩
        simp only [poppedEq]
        step with contracts.fresh current state as ⟨fresh, visitedWorld, graphs, visited⟩
        rcases graphs with ⟨graphEq, startEq⟩
        cases fresh with
        | false =>
          simp only [Bool.false_eq_true, ite_false] at visited ⊢
          rcases visited with ⟨seenEq, seen⟩
          simp only [WP.spec_ok]
          refine ⟨⟨prior, afterRep, ?_⟩, decreases⟩
          simp only [ite_true, graphEq, startEq, seenEq]
          exact Covered.duplicate (view current).graph (view current).start state (view current).seen prior covered seen
        | true =>
          simp only [ite_true] at visited ⊢
          step with contracts.expand visitedWorld state as ⟨expansion, expandedWorld, sameGraph, seenEq, meaning⟩
          rcases sameGraph with ⟨expandedGraph, expandedStart⟩
          rw [graphEq] at meaning
          cases expansion with
          | Hit => simp
          | Miss =>
            simp only [ExpansionSound] at meaning
            simp only [WP.spec_ok]
            refine ⟨⟨prior, afterRep, ?_⟩, decreases⟩
            simp only [ite_true, expandedGraph, expandedStart, graphEq, startEq, seenEq, visited]
            have next := Covered.expand (view current).graph (view current).start state (view current).seen prior [] covered meaning.1 (by
              intro destination edge; exact False.elim (meaning.2 destination edge))
            simpa using next
          | Next destination =>
            simp only [ExpansionSound] at meaning
            step with rust_push after prior afterRep destination as ⟨success, next, pushed⟩
            rcases pushed with ⟨failed, _⟩ | ⟨succeeded, nextRep⟩
            · simp [failed]
            · simp only [succeeded, ite_true, WP.spec_ok]
              refine ⟨⟨prior ++ [destination], nextRep, ?_⟩, decreases⟩
              simp only [ite_true, expandedGraph, expandedStart, graphEq, startEq, seenEq, visited]
              exact Covered.expand (view current).graph (view current).start state (view current).seen prior [destination] covered meaning.1 (by
                intro next edge; simp [meaning.2 next edge])
          | Children children =>
            simp only [ExpansionSound] at meaning
            obtain ⟨childrenValues, childrenRep, miss, edges⟩ := meaning
            step with rust_push_children after children prior childrenValues afterRep childrenRep
              as ⟨success, next, values, nextRep, combined⟩
            cases success with
            | false => simp
            | true =>
              simp only [ite_true, WP.spec_ok]
              refine ⟨⟨values, nextRep, ?_⟩, decreases⟩
              rw [combined rfl]
              simp only [ite_true, expandedGraph, expandedStart, graphEq, startEq, seenEq, visited]
              exact Covered.expand (view current).graph (view current).start state (view current).seen prior childrenValues.reverse covered miss (by
                intro next edge; simpa using edges next edge)
    · simp [live]
  · exact initial

@[step]
theorem rust_may_arrive {State W : Type} (walker : arrival_walk.Walker W State) (view : W → View State)
    (settled : W → Prop) (contracts : LocalContracts walker view settled) (world : W) (budget : Usize) :
    arrival_walk.may_arrive walker world budget ⦃ possible result =>
      possible = false → settled result ∧ NoHit (view result) ⦄ := by
  unfold arrival_walk.may_arrive
  step with rust_new (State := State) as ⟨pending, rep⟩
  exact rust_may_arrive_loop walker view settled contracts world budget pending false ⟨[], rep, rfl⟩

end ArrivalWalk
