/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.SourceCertificate
import MemorySafety.SourceVectors

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option debug.skipKernelTC false
set_option Elab.async false

namespace MemorySafety.Source

theorem source_gather_append (accumulated next output : alloc.vec.Vec memory_safety.Move)
    (index : Usize)
    (computed : memory_safety.moves_loop0_loop0 accumulated next index = .ok output) :
    output.val = accumulated.val ++ next.val.drop index.val := by
  generalize fuelEq : next.val.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing accumulated index with
  | h fuel ih =>
    have unroll : memory_safety.moves_loop0_loop0 accumulated next index = (do
        let result ← memory_safety.moves_loop0_loop0.body next accumulated index
        match result with
        | .cont pair => memory_safety.moves_loop0_loop0 pair.1 next pair.2
        | .done output => .ok output) := by
      unfold memory_safety.moves_loop0_loop0
      rw [loop.eq_def]
      congr 1
      funext result
      cases result with
      | cont pair => cases pair; rfl
      | done output => rfl
    rw [unroll] at computed
    by_cases inside : index.val < next.val.length
    · have bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := alloc.vec.Vec.len_ineq next
        scalar_tac
      let after : Usize := Usize.ofNatCore (index.val + 1) bound
      have afterVal : after.val = index.val + 1 := by simp [after]
      have increment := usize_increment index bound
      have lookup : alloc.vec.Vec.index_usize next index = .ok next.val[index.val] := by
        simp [alloc.vec.Vec.index_usize, List.getElem?_eq_getElem inside]
      have test : index < alloc.vec.Vec.len next := by scalar_tac
      have body : memory_safety.moves_loop0_loop0.body next accumulated index = (do
          let updated ← alloc.vec.Vec.push accumulated next.val[index.val]
          .ok (.cont (updated, after))) := by
        simp only [memory_safety.moves_loop0_loop0.body, test, ite_true,
          alloc.vec.Vec.index_slice_index, lookup, bind_ok, source_move_clone]
        congr 1
        funext updated
        rw [increment]
        simp only [bind_ok]
        rfl
      rw [body] at computed
      simp only [Std.bind_assoc, bind_ok] at computed
      obtain ⟨updated, pushed, finished⟩ := source_bind_success _ _ _ computed
      have tail := ih (next.val.length - after.val) (by omega) updated after finished rfl
      rw [tail, vec_push_success accumulated updated next.val[index.val] pushed]
      rw [List.drop_eq_getElem_cons inside]
      simp only [afterVal, List.append_assoc, List.singleton_append]
      rfl
    · have test : ¬index < alloc.vec.Vec.len next := by scalar_tac
      simp only [memory_safety.moves_loop0_loop0.body, test, ite_false, bind_ok, ok.injEq] at computed
      rw [← computed]
      simp [List.drop_eq_nil_of_le (show next.val.length ≤ index.val by omega)]

def ThreadAgreement (program : memory_safety.Program) (state : memory_safety.State) : Prop :=
  ∀ thread choices, memory_safety.thread_moves program state thread = .ok (some choices) →
    threadMoves (programModel program) (stateModel state) thread.val = some (choices.val.map moveModel)

theorem source_gather_loop_agrees (program : memory_safety.Program) (state : memory_safety.State)
    (agreement : ThreadAgreement program state)
    (accumulated output : alloc.vec.Vec memory_safety.Move) (known : Bool) (index : Usize)
    (computed : memory_safety.moves_loop0 program.allocations program.threads state.pcs state.live
      state.writes state.history state.bad accumulated known index = .ok (output, true)) :
    known = true ∧ ∃ rest,
      gatherMoves (programModel program) (stateModel state)
        ((List.range program.threads.val.length).drop index.val) = some rest ∧
      output.val.map moveModel = accumulated.val.map moveModel ++ rest := by
  generalize fuelEq : program.threads.val.length - index.val = fuel
  induction fuel using Nat.strong_induction_on generalizing accumulated known index with
  | h fuel ih =>
    have unroll : memory_safety.moves_loop0 program.allocations program.threads state.pcs state.live
        state.writes state.history state.bad accumulated known index = (do
        let result ← memory_safety.moves_loop0.body program.allocations program.threads state.pcs state.live
          state.writes state.history state.bad accumulated known index
        match result with
        | .cont (updated, nextKnown, nextIndex) =>
          memory_safety.moves_loop0 program.allocations program.threads state.pcs state.live
            state.writes state.history state.bad updated nextKnown nextIndex
        | .done output => .ok output) := by
      unfold memory_safety.moves_loop0
      rw [loop.eq_def]
      congr 1
      funext result
      cases result with
      | cont triple => rcases triple with ⟨updated, nextKnown, nextIndex⟩; rfl
      | done output => rfl
    rw [unroll] at computed
    by_cases inside : index.val < program.threads.val.length
    · have bound : index.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := alloc.vec.Vec.len_ineq program.threads
        scalar_tac
      let nextIndex : Usize := Usize.ofNatCore (index.val + 1) bound
      have nextVal : nextIndex.val = index.val + 1 := by simp [nextIndex]
      have increment := usize_increment index bound
      have test : index < alloc.vec.Vec.len program.threads := by scalar_tac
      cases known with
      | false => simp [memory_safety.moves_loop0.body, test] at computed
      | true =>
        have body : memory_safety.moves_loop0.body program.allocations program.threads state.pcs state.live
            state.writes state.history state.bad accumulated true index = (do
            let choices ← memory_safety.thread_moves program state index
            match choices with
            | none => .ok (.cont (accumulated, false, nextIndex))
            | some choices => do
              let updated ← memory_safety.moves_loop0_loop0 accumulated choices 0#usize
              .ok (.cont (updated, true, nextIndex))) := by
          simp only [memory_safety.moves_loop0.body, test, ite_true]
          change Aeneas.Std.bind (memory_safety.thread_moves program state index) _ = _
          congr 1
          funext choices
          cases choices with
          | none => rw [increment]; simp only [bind_ok]; rfl
          | some choices =>
            dsimp only
            congr 1
            funext updated
            rw [increment]
            simp only [bind_ok]
            rfl
        rw [body] at computed
        simp only [Std.bind_assoc] at computed
        obtain ⟨choices, enumerated, remaining⟩ := source_bind_success _ _ _ computed
        cases choices with
        | none =>
          dsimp only at remaining
          simp only [bind_ok] at remaining
          change memory_safety.moves_loop0 program.allocations program.threads state.pcs state.live state.writes
            state.history state.bad accumulated false nextIndex = .ok (output, true) at remaining
          have impossible := (ih (program.threads.val.length - nextIndex.val) (by omega)
            accumulated false nextIndex remaining rfl).1
          cases impossible
        | some choices =>
          dsimp only at remaining
          simp only [Std.bind_assoc, bind_ok] at remaining
          change (do
            let updated ← memory_safety.moves_loop0_loop0 accumulated choices 0#usize
            memory_safety.moves_loop0 program.allocations program.threads state.pcs state.live state.writes
              state.history state.bad updated true nextIndex) = .ok (output, true) at remaining
          obtain ⟨updated, appended, continued⟩ := source_bind_success _ _ _ remaining
          have appendedEq : updated.val = accumulated.val ++ choices.val := by
            simpa using source_gather_append accumulated choices updated 0#usize appended
          obtain ⟨_, rest, tailModel, outputEq⟩ := ih
            (program.threads.val.length - nextIndex.val) (by omega) updated true nextIndex continued rfl
          rw [nextVal] at tailModel
          refine ⟨rfl, choices.val.map moveModel ++ rest, ?_, ?_⟩
          · have rangeInside : index.val < (List.range program.threads.val.length).length := by simpa using inside
            rw [List.drop_eq_getElem_cons rangeInside]
            simp only [List.getElem_range, gatherMoves]
            rw [agreement index choices enumerated, tailModel]
            rfl
          · rw [outputEq, appendedEq, List.map_append, List.append_assoc]
    · have test : ¬index < alloc.vec.Vec.len program.threads := by scalar_tac
      have same : accumulated = output ∧ known = true := by
        simpa [memory_safety.moves_loop0.body, test] using computed
      rcases same with ⟨rfl, rfl⟩
      refine ⟨rfl, [], ?_, by simp⟩
      rw [List.drop_eq_nil_of_le (by simpa using (show program.threads.val.length ≤ index.val by omega))]
      rfl

theorem source_moves_agrees_of_thread (program : memory_safety.Program) (state : memory_safety.State)
    (agreement : ThreadAgreement program state) (choices : alloc.vec.Vec memory_safety.Move)
    (computed : memory_safety.moves program state = .ok (some choices)) :
    moves (programModel program) (stateModel state) = some (choices.val.map moveModel) := by
  by_cases pcsShape : state.pcs.val.length = program.threads.val.length
  · have sourcePcs : alloc.vec.Vec.len state.pcs = alloc.vec.Vec.len program.threads := by scalar_tac
    by_cases liveShape : state.live.val.length = program.allocations.val.length
    · have sourceLive : alloc.vec.Vec.len state.live = alloc.vec.Vec.len program.allocations := by scalar_tac
      have shape : ¬((stateModel state).pcs.length ≠ (programModel program).threads.length ∨
          (stateModel state).live.length ≠ (programModel program).allocations.length) := by
        simp [stateModel, programModel, pcsShape, liveShape]
      simp only [memory_safety.moves, sourcePcs, sourceLive, bne_self_eq_false,
        Bool.false_eq_true, ite_false] at computed
      rw [moves, ite_eq_right shape]
      simp only [stateModel_bad]
      by_cases bad : state.bad = true
      · simp only [bad, ite_true, ok.injEq, Option.some.injEq] at computed
        rw [←computed]
        simp only [bad, ite_true]
        rfl
      · simp only [bad] at computed
        obtain ⟨pair, gathered, remaining⟩ := source_bind_success _ _ _ computed
        rcases pair with ⟨output, known⟩
        cases known with
        | false => simp at remaining
        | true =>
          have equal : output = choices := by simpa using remaining
          subst output
          obtain ⟨_, rest, gatherEq, outputEq⟩ := source_gather_loop_agrees program state agreement
            (alloc.vec.Vec.new _) choices true 0#usize gathered
          have outputModel : choices.val.map moveModel = rest := by simpa [alloc.vec.Vec.new] using outputEq
          simp only [bad]
          simpa [outputModel, programModel] using gatherEq
    · have sourceLive : alloc.vec.Vec.len state.live ≠ alloc.vec.Vec.len program.allocations := by scalar_tac
      simp [memory_safety.moves, sourcePcs, sourceLive] at computed
  · have sourcePcs : alloc.vec.Vec.len state.pcs ≠ alloc.vec.Vec.len program.threads := by scalar_tac
    simp [memory_safety.moves, sourcePcs] at computed

end MemorySafety.Source
