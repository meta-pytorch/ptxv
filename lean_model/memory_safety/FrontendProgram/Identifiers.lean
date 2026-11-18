/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Names

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option backward.split false
set_option backward.isDefEq.respectTransparency false

namespace PtxFrontendProgram

local instance (byte : U8) : Decidable (InitialCharacter byte) := by
  unfold InitialCharacter
  infer_instance

local instance (byte : U8) : Decidable (FollowingCharacter byte) := by
  unfold FollowingCharacter
  infer_instance

private def followingResult (byte : U8) : Result Bool :=
if byte >= 97#u8
                       then if byte <= 122#u8
                            then ok true
                            else if byte >= 65#u8
                                 then if byte <= 90#u8
                                      then ok true
                                      else if byte >= 48#u8
                                           then if byte <= 57#u8
                                                then ok true
                                                else if byte = 95#u8
                                                     then ok true
                                                     else ok (byte = 36#u8)
                                           else if byte = 95#u8
                                                then ok true
                                                else ok (byte = 36#u8)
                                 else if byte >= 48#u8
                                      then if byte <= 57#u8
                                           then ok true
                                           else if byte = 95#u8
                                                then ok true
                                                else ok (byte = 36#u8)
                                      else if byte = 95#u8
                                           then ok true
                                           else ok (byte = 36#u8)
                       else if byte >= 65#u8
                            then if byte <= 90#u8
                                 then ok true
                                 else if byte >= 48#u8
                                      then if byte <= 57#u8
                                           then ok true
                                           else if byte = 95#u8
                                                then ok true
                                                else ok (byte = 36#u8)
                                      else if byte = 95#u8
                                           then ok true
                                           else ok (byte = 36#u8)
                            else if byte >= 48#u8
                                 then if byte <= 57#u8
                                      then ok true
                                      else if byte = 95#u8
                                           then ok true
                                           else ok (byte = 36#u8)
                                 else if byte = 95#u8
                                      then ok true
                                      else ok (byte = 36#u8)

private def initialResult (byte : U8) : Result Bool :=
if byte >= 97#u8
                                    then if byte <= 122#u8
                                         then ok true
                                         else if byte >= 65#u8
                                              then if byte <= 90#u8
                                                   then ok true
                                                   else if byte = 95#u8
                                                        then ok true
                                                        else ok (byte = 36#u8)
                                              else if byte = 95#u8
                                                   then ok true
                                                   else ok (byte = 36#u8)
                                    else if byte >= 65#u8
                                         then if byte <= 90#u8
                                              then ok true
                                              else if byte = 95#u8
                                                   then ok true
                                                   else ok (byte = 36#u8)
                                         else if byte = 95#u8
                                              then ok true
                                              else ok (byte = 36#u8)

private theorem following_result (byte : U8) :
    followingResult byte = .ok (decide (FollowingCharacter byte)) := by
  generalize expectedEq : decide (FollowingCharacter byte) = expected
  unfold followingResult
  repeat' split
  all_goals apply congrArg (fun value : Bool => (ok value : Result Bool))
  all_goals rw [←expectedEq]
  all_goals apply Bool.eq_iff_iff.mpr
  all_goals simp_all [FollowingCharacter,InitialCharacter]
  all_goals scalar_tac

private theorem initial_result (byte : U8) :
    initialResult byte = .ok (decide (InitialCharacter byte)) := by
  generalize expectedEq : decide (InitialCharacter byte) = expected
  unfold initialResult
  repeat' split
  all_goals apply congrArg (fun value : Bool => (ok value : Result Bool))
  all_goals rw [←expectedEq]
  all_goals apply Bool.eq_iff_iff.mpr
  all_goals simp_all [InitialCharacter]
  all_goals scalar_tac

private theorem identifier_body (source : Slice U8) (finish position : Usize) (valid : Bool) :
    strict_memory_frontend_parse.identifier_loop0.body source finish position valid =
      if valid then if position < finish then do
        let byte ← Slice.index_usize source position
        let after ← position + 1#usize
        .ok (.cont (after,decide (FollowingCharacter byte)))
      else .ok (.done valid) else .ok (.done valid) := by
  unfold strict_memory_frontend_parse.identifier_loop0.body
  cases valid <;> simp only [Bool.false_eq_true,ite_false,ite_true]
  split
  · congr 1
    funext byte
    change (do let valid ← followingResult byte; let after ← position + 1#usize; .ok (ControlFlow.cont (after,valid) : ControlFlow (Usize × Bool) Bool)) = _
    rw [following_result]
    simp only [Bind.bind,bind_ok]
  · rfl

private theorem identifier_loop1_eq (source : Slice U8) (finish position : Usize) (valid : Bool) :
    strict_memory_frontend_parse.identifier_loop1 source finish position valid =
      strict_memory_frontend_parse.identifier_loop0 source finish position valid := by rfl

private theorem identifier_loop_false (source : Slice U8) (finish position : Usize) :
    strict_memory_frontend_parse.identifier_loop0 source finish position false = .ok false := by
  unfold strict_memory_frontend_parse.identifier_loop0
  rw [loop.eq_def]
  simp [strict_memory_frontend_parse.identifier_loop0.body,bind_ok]

theorem source_identifier_tail (source : Slice U8) (finish position : Usize) (valid : Bool)
    (ordered : position.val≤finish.val) (bound : finish.val≤source.val.length)
    (computed : strict_memory_frontend_parse.identifier_loop0 source finish position valid=.ok true) :
    valid=true ∧ ∀ slot,position.val≤slot → slot<finish.val →
      ∃ byte,source.val[slot]?=some byte ∧ FollowingCharacter byte := by
  cases valid with
  | false => rw [identifier_loop_false] at computed; simp at computed
  | true =>
    refine ⟨rfl,?_⟩
    generalize fuelEq : finish.val-position.val=fuel
    induction fuel using Nat.strong_induction_on generalizing position with
    | h fuel ih =>
      have unroll : strict_memory_frontend_parse.identifier_loop0 source finish position true = (do
          let outcome ← strict_memory_frontend_parse.identifier_loop0.body source finish position true
          match outcome with
          | .cont pair => strict_memory_frontend_parse.identifier_loop0 source finish pair.1 pair.2
          | .done value => .ok value) := by
        unfold strict_memory_frontend_parse.identifier_loop0
        rw [loop.eq_def]
        congr 1
        funext outcome
        cases outcome with
        | cont pair => cases pair; rfl
        | done value => rfl
      rw [unroll,identifier_body] at computed
      simp only [ite_true] at computed
      by_cases inside : position.val<finish.val
      · have test : position<finish := by scalar_tac
        have sourceInside : position.val<source.val.length := by omega
        let byte := source.val[position.val]
        have read : Slice.index_usize source position=.ok byte := by
          simp [Slice.index_usize,List.getElem?_eq_getElem sourceInside,byte]
        have incrementBound : position.val+1<2^UScalarTy.Usize.numBits := by
          have := Slice.length_ineq source
          scalar_tac
        let after := Usize.ofNatCore (position.val+1) incrementBound
        have afterVal : after.val=position.val+1 := by simp [after]
        have increment := source_usize_increment position incrementBound
        simp only [test,ite_true,read,increment,bind_ok] at computed
        by_cases follows : FollowingCharacter byte
        · simp only [follows,decide_true] at computed
          have tail := ih (finish.val-after.val) (by omega) after (by omega) computed rfl
          intro slot low high
          by_cases first : slot=position.val
          · subst slot
            exact ⟨byte,List.getElem?_eq_getElem sourceInside,follows⟩
          · exact tail slot (by omega) high
        · simp only [follows,decide_false,identifier_loop_false] at computed
          simp at computed
      · intro slot low high
        omega

theorem source_slice_index {α : Type} (source : Slice α) (index : Usize) (value : α)
    (computed : Slice.index_usize source index=.ok value) : source.val[index.val]?=some value := by
  unfold Slice.index_usize at computed
  cases found : source.val[index.val]? with
  | none => simp [found] at computed
  | some input => simpa [found] using computed

theorem source_identifier (source : Slice U8) (token : SourceToken) (register : Bool)
    (computed : strict_memory_frontend_parse.identifier source token register=.ok true) :
    Identifier source.val token register := by
  unfold strict_memory_frontend_parse.identifier at computed
  cases wordEq : token.word with
  | false => simp only [wordEq,Bool.false_eq_true,ite_false] at computed; simp at computed
  | true =>
    simp only [wordEq,ite_true] at computed
    by_cases empty : token.start≥token.end
    · simp only [empty,ite_true] at computed; simp at computed
    · simp only [empty,ite_false] at computed
      by_cases outside : token.end>Slice.len source
      · simp only [outside,ite_true] at computed; simp at computed
      · simp only [outside,ite_false] at computed
        have positive : token.start.val<token.end.val := by scalar_tac
        have bound : token.end.val≤source.val.length := by scalar_tac
        have startIncrement : token.start.val+1<2^UScalarTy.Usize.numBits := by
          have := Slice.length_ineq source
          scalar_tac
        let next := Usize.ofNatCore (token.start.val+1) startIncrement
        have nextVal : next.val=token.start.val+1 := by simp [next]
        have increment := source_usize_increment token.start startIncrement
        cases register with
        | false =>
          simp only [Bool.false_eq_true,ite_false] at computed
          have different : token.start≠token.end := by scalar_tac
          simp only [different,ite_false] at computed
          obtain ⟨first,firstRead,computed⟩ := source_bind_success _ _ _ computed
          change (do let valid ← initialResult first; let after ← token.start+1#usize
                     strict_memory_frontend_parse.identifier_loop1 source token.end after valid)=.ok true at computed
          rw [initial_result] at computed
          simp only [Bind.bind,bind_ok,increment,identifier_loop1_eq] at computed
          have tail := source_identifier_tail source token.end next _ (by omega) bound computed
          refine ⟨wordEq,positive,bound,?_,?_,?_,?_⟩
          · simpa using positive
          · simp
          · exact ⟨first,by simpa using source_slice_index source token.start first firstRead,
              of_decide_eq_true tail.1⟩
          · simpa [nextVal] using tail.2
        | true =>
          simp only [ite_true] at computed
          obtain ⟨leadByte,prefixRead,computed⟩ := source_bind_success _ _ _ computed
          by_cases wrong : leadByte≠37#u8
          · have test : (leadByte !=37#u8)=true := by scalar_tac
            simp only [test,ite_true] at computed
            simp at computed
          · have prefixEq : leadByte=37#u8 := by scalar_tac
            have test : (leadByte !=37#u8)=false := by simp [prefixEq]
            simp only [test,Bool.false_eq_true,ite_false,increment,bind_ok] at computed
            change (if next=token.end then (ok false : Result Bool) else _) = ok true at computed
            by_cases ended : next=token.end
            · simp only [ended,ite_true] at computed; simp at computed
            · simp only [ended,ite_false] at computed
              have remaining : next.val<token.end.val := by scalar_tac
              have afterBound : next.val+1<2^UScalarTy.Usize.numBits := by
                have := Slice.length_ineq source
                scalar_tac
              let after := Usize.ofNatCore (next.val+1) afterBound
              have afterVal : after.val=next.val+1 := by simp [after]
              obtain ⟨first,firstRead,computed⟩ := source_bind_success _ _ _ computed
              change (do let valid ← initialResult first; let after ← next+1#usize
                         strict_memory_frontend_parse.identifier_loop0 source token.end after valid)=.ok true at computed
              rw [initial_result] at computed
              simp only [Bind.bind,bind_ok,source_usize_increment next afterBound] at computed
              have tail := source_identifier_tail source token.end after _ (by omega) bound computed
              refine ⟨wordEq,positive,bound,?_,?_,?_,?_⟩
              · simpa [nextVal] using remaining
              · intro _
                exact prefixEq ▸ source_slice_index source token.start leadByte prefixRead
              · exact ⟨first,by simpa [nextVal] using source_slice_index source next first firstRead,
                  of_decide_eq_true tail.1⟩
              · simpa [nextVal,afterVal] using tail.2

end PtxFrontendProgram
