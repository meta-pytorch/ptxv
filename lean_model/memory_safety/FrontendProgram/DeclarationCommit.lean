/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.DeclarationFresh
import FrontendProgram.SyntaxProperties
import FrontendProgram.Header

open Aeneas Aeneas.Std Result PtxvMemorySafety
open MemorySafety.Source
set_option autoImplicit false
set_option debug.skipKernelTC false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192
set_option Elab.async false
set_option backward.split false
set_option backward.isDefEq.respectTransparency false

namespace PtxFrontendProgram

private def declarationCommit (source : Slice U8) (scope : Scope) (candidate : Declaration) (next : Usize) :
    Result (Option (Declaration × Usize)) :=
  match U16.checked_add candidate.first candidate.count with
  | none => .ok none
  | some _ => do
    let fresh ← strict_memory_frontend_parse.parse_declaration_loop0 source scope.declarations candidate.bits candidate.predicate
      candidate.«name» candidate.count candidate.indexed candidate.first 0#usize true
    if fresh then .ok (some (candidate,next)) else .ok none

private theorem source_declaration_commit (source : Slice U8) (scope : Scope) (candidate selected : Declaration) (next finish : Usize)
    (valid : ∀ declaration∈scope.declarations.val,Identifier source.val declaration.«name» true)
    (candidateValid : Identifier source.val candidate.«name» true)
    (computed : declarationCommit source scope candidate next=.ok (some (selected,finish))) :
    selected=candidate ∧ finish=next ∧ candidate.first.val+candidate.count.val<65536 ∧
    ∀ declaration∈scope.declarations.val,DeclarationSeparate source.val declaration candidate := by
  unfold declarationCommit at computed
  cases sum : U16.checked_add candidate.first candidate.count with
  | none => simp only [sum] at computed; simp at computed
  | some total =>
    simp only [sum] at computed
    obtain ⟨fresh,freshComputed,computed⟩ := source_bind_success _ _ _ computed
    cases fresh with
    | false => simp at computed
    | true =>
      simp only [ite_true,ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
      obtain ⟨same,nextEq⟩ := computed
      have law := U16.checked_add_bv_spec candidate.first candidate.count
      rw [sum] at law
      refine ⟨same.symm,nextEq.symm,by scalar_tac,?_⟩
      intro declaration member
      obtain ⟨position,found⟩ := List.mem_iff_getElem?.mp member
      exact source_declaration_fresh_loop source scope.declarations candidate 0#usize valid candidateValid freshComputed
        position declaration (by simp) found

private theorem declaration_take (source : Slice U8) (tokens : Slice SourceToken) (position after : Usize)
    (expected : Slice U8) (text : String) (literal : expected.val=utf8 text)
    (computed : strict_memory_frontend_parse.take source tokens position expected=.ok (some after)) :
    after.val=position.val+1 ∧ TokenAt source.val tokens.val position.val text := by
  obtain ⟨increment,token,found,meaning⟩ := source_take source tokens position after expected computed
  exact ⟨increment,token,found,by simpa only [TokenSpelling,←literal] using meaning⟩

theorem source_parse_declaration (source : Slice U8) (tokens : Slice SourceToken) (position finish : Usize)
    (scope : Scope) (declaration : Declaration)
    (valid : ∀ old∈scope.declarations.val,Identifier source.val old.«name» true)
    (computed : strict_memory_frontend_parse.parse_declaration source tokens position scope=.ok (some (declaration,finish))) :
    DeclarationSyntax source.val tokens.val position.val finish.val declaration ∧ 0<declaration.count.val ∧
    declaration.first=(if declaration.predicate then scope.predicate_count else scope.virtual_count) ∧
    declaration.first.val+declaration.count.val<65536 ∧
    ∀ old∈scope.declarations.val,DeclarationSeparate source.val old declaration := by
  unfold strict_memory_frontend_parse.parse_declaration at computed
  simp only [lift,bind_ok] at computed
  obtain ⟨regResult,regComputed,computed⟩ := source_bind_success _ _ _ computed
  cases regResult with
  | none => simp at computed
  | some typePosition =>
    obtain ⟨namePosition,nameAdded,computed⟩ := source_bind_success _ _ _ computed
    split at computed
    · simp at computed
    · obtain ⟨typeToken,typeComputed,computed⟩ := source_bind_success _ _ _ computed
      obtain ⟨kind,kindComputed,computed⟩ := source_bind_success _ _ _ computed
      cases kind with
      | none => simp at computed
      | some pair =>
        rcases pair with ⟨bits,predicate⟩
        obtain ⟨nameToken,nameComputed,computed⟩ := source_bind_success _ _ _ computed
        obtain ⟨identifier,identifierComputed,computed⟩ := source_bind_success _ _ _ computed
        cases identifier with
        | false => simp at computed
        | true =>
          simp only [ite_true] at computed
          obtain ⟨after,afterAdded,computed⟩ := source_bind_success _ _ _ computed
          obtain ⟨indexed,indexedComputed,computed⟩ := source_bind_success _ _ _ computed
          have nameValid := source_identifier source nameToken true identifierComputed
          have typeMeaning := source_register_type source typeToken bits predicate kindComputed
          have regMeaning := declaration_take source tokens position typePosition _ ".reg" (by decide) regComputed
          have nameEq := source_usize_add_value typePosition 1#usize namePosition nameAdded
          have afterEq := source_usize_add_value typePosition 2#usize after afterAdded
          cases indexed with
          | false =>
            simp only [Bool.false_eq_true,ite_false] at computed
            obtain ⟨semicolonResult,semicolonComputed,computed⟩ := source_bind_success _ _ _ computed
            cases semicolonResult with
            | none => simp at computed
            | some next =>
              obtain ⟨first,firstComputed,computed⟩ := source_bind_success _ _ _ computed
              let candidate : Declaration := ⟨nameToken,1#u16,first,bits,predicate,false⟩
              change declarationCommit source scope candidate next=.ok (some (declaration,finish)) at computed
              obtain ⟨rfl,rfl,sumBound,fresh⟩ := source_declaration_commit source scope candidate declaration next finish valid nameValid computed
              have firstEq : first=(if predicate then scope.predicate_count else scope.virtual_count) := by
                cases predicate <;> simpa only [Bool.false_eq_true,ite_false,ite_true,ok.injEq] using firstComputed.symm
              have semicolon := declaration_take source tokens after finish _ ";" (by decide) semicolonComputed
              have finishEq : finish.val=position.val+4 := by scalar_tac
              refine ⟨?_,by change 0 < 1; omega,firstEq,sumBound,fresh⟩
              rw [finishEq]
              apply DeclarationSyntax.single candidate
              · exact regMeaning.2
              · simpa [regMeaning.1] using source_slice_index tokens typePosition typeToken typeComputed
              · exact typeMeaning
              · have nameAt : namePosition.val=position.val+2 := by scalar_tac
                simpa [nameAt] using source_slice_index tokens namePosition nameToken nameComputed
              · exact nameValid
              · rfl
              · rfl
              · rfl
              · have semicolonAt : after.val=position.val+3 := by scalar_tac
                simpa [semicolonAt] using semicolon.2
          | true =>
            simp only [ite_true] at computed
            obtain ⟨numberPosition,numberAdded,computed⟩ := source_bind_success _ _ _ computed
            obtain ⟨numberResult,numberComputed,computed⟩ := source_bind_success _ _ _ computed
            cases numberResult with
            | none => simp at computed
            | some pair =>
              rcases pair with ⟨number,next⟩
              by_cases zero : number=0#u64
              · simp [zero] at computed
              · simp [zero] at computed
                split at computed
                · simp at computed
                · obtain ⟨closeResult,closeComputed,computed⟩ := source_bind_success _ _ _ computed
                  cases closeResult with
                  | none => simp at computed
                  | some close =>
                    obtain ⟨semicolonResult,semicolonComputed,computed⟩ := source_bind_success _ _ _ computed
                    cases semicolonResult with
                    | none => simp at computed
                    | some finishPosition =>
                      obtain ⟨first,firstComputed,computed⟩ := source_bind_success _ _ _ computed
                      let count := UScalar.cast .U16 number
                      let candidate : Declaration := ⟨nameToken,count,first,bits,predicate,true⟩
                      change declarationCommit source scope candidate finishPosition=.ok (some (declaration,finish)) at computed
                      obtain ⟨rfl,rfl,sumBound,fresh⟩ := source_declaration_commit source scope candidate declaration finishPosition finish valid nameValid computed
                      have firstEq : first=(if predicate then scope.predicate_count else scope.virtual_count) := by
                        cases predicate <;> simpa only [Bool.false_eq_true,ite_false,ite_true,ok.injEq] using firstComputed.symm
                      have numberBound : number.val<65536 := by
                        have maximum : U16.rMax=65535 := rfl
                        omega
                      have countEq : count.val=number.val := by simp [count,UScalar.cast_val_eq,UScalar.size_def,Nat.mod_eq_of_lt numberBound]
                      have positive : 0<number.val := by scalar_tac
                      have numberAt := source_usize_add_value after 1#usize numberPosition numberAdded
                      obtain ⟨nextEq,numberMeaning⟩ := source_unsigned source tokens numberPosition next number numberComputed
                      have closeMeaning := declaration_take source tokens next close _ ">" (by decide) closeComputed
                      have semicolon := declaration_take source tokens close finish _ ";" (by decide) semicolonComputed
                      have finishEq : finish.val=position.val+7 := by scalar_tac
                      refine ⟨?_,by simpa [candidate,countEq] using positive,firstEq,sumBound,fresh⟩
                      rw [finishEq]
                      apply DeclarationSyntax.indexed candidate
                      · exact regMeaning.2
                      · simpa [regMeaning.1] using source_slice_index tokens typePosition typeToken typeComputed
                      · exact typeMeaning
                      · have nameAt : namePosition.val=position.val+2 := by scalar_tac
                        simpa [nameAt] using source_slice_index tokens namePosition nameToken nameComputed
                      · exact nameValid
                      · rfl
                      · obtain ⟨sign,found,meaning⟩ := source_matches source tokens after _ indexedComputed
                        have literal : utf8 "<"=[60#u8] := by decide
                        have afterAt : after.val=position.val+3 := by scalar_tac
                        exact ⟨sign,by simpa [afterAt] using found,by simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning⟩
                      · have numberPositionEq : numberPosition.val=position.val+4 := by scalar_tac
                        simpa [numberPositionEq] using numberMeaning
                      · exact positive
                      · exact countEq
                      · rfl
                      · have closeAt : next.val=position.val+5 := by scalar_tac
                        simpa [closeAt] using closeMeaning.2
                      · have semicolonAt : close.val=position.val+6 := by scalar_tac
                        simpa [semicolonAt] using semicolon.2

end PtxFrontendProgram
