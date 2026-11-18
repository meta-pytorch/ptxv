/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.TokenHelpers

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

private def suffixResult (source : Slice U8) (start finish : Usize) (count first : U16) : Result (Option U16) := do
  let parsed ← strict_memory_frontend.parse_digits source start finish 10#u64
  match parsed with
  | none => ok none
  | some value =>
    let count64 ← lift (UScalar.cast .U64 count)
    if value<count64 then
      let number ← lift (UScalar.cast .U16 value)
      ok (U16.checked_add first number)
    else ok none

private theorem source_suffix (source : Slice U8) (start finish : Usize) (count first index : U16)
    (computed : suffixResult source start finish count first=.ok (some index)) :
    ∃ number,number<count.val ∧ DigitSequence source.val 10 start.val finish.val 0 number ∧
      index.val=first.val+number := by
  unfold suffixResult at computed
  obtain ⟨parsed,parsedComputed,computed⟩ := source_bind_success _ _ _ computed
  cases parsed with
  | none => simp at computed
  | some value =>
    simp only [lift,bind_ok] at computed
    split at computed
    · simp only [ok.injEq] at computed
      have smaller : value.val<count.val := by scalar_tac
      have addition := U16.checked_add_bv_spec first (UScalar.cast .U16 value)
      rw [computed] at addition
      have suffix := source_parse_digits source start finish 10#u64 value parsedComputed
      refine ⟨value.val,smaller,by simpa using suffix.2.2.2,?_⟩
      rw [addition.2.1]
      have cast : (UScalar.cast .U16 value).val=value.val := by scalar_tac
      rw [cast]
    · simp at computed

private theorem prefix_name (source : List U8) (token : SourceToken) (finish : Usize)
    (inside : finish.val≤token.end.val) (ordered : token.start.val≤finish.val) :
    nameBytes source {token with «end»:=finish,word:=true}=
      (nameBytes source token).take (finish.val-token.start.val) := by
  simp only [nameBytes,List.take_take]
  rw [Nat.min_eq_left (by omega)]

theorem source_declaration_register (source : Slice U8) (declaration : Declaration) (token : SourceToken) (index : U16)
    (positiveCount : 0<declaration.count.val)
    (computed : strict_memory_frontend_parse.declaration_register source declaration token=.ok (some index)) :
    ∃ number,number<declaration.count.val ∧ RegisterName source.val declaration token number ∧
      index.val=declaration.first.val+number := by
  unfold strict_memory_frontend_parse.declaration_register at computed
  obtain ⟨valid,identifierComputed,computed⟩ := source_bind_success _ _ _ computed
  cases valid with
  | false => simp at computed
  | true =>
    simp only [ite_true] at computed
    cases indexed : declaration.indexed with
    | false =>
      simp only [indexed,Bool.false_eq_true,ite_false] at computed
      obtain ⟨same,sameComputed,computed⟩ := source_bind_success _ _ _ computed
      cases same with
      | false => simp at computed
      | true =>
        simp only [ite_true,ok.injEq,Option.some.injEq] at computed
        cases computed
        refine ⟨0,positiveCount,?_,by simp⟩
        simpa only [RegisterName,indexed,Bool.false_eq_true,ite_false] using
          And.intro (source_names_equal source declaration.name token sameComputed) (Eq.refl (0:Nat))
    | true =>
      simp only [indexed,ite_true] at computed
      by_cases outside : declaration.name.end>Slice.len source
      · simp only [outside,ite_true] at computed; simp at computed
      · simp only [outside,ite_false] at computed
        by_cases empty : declaration.name.start≥declaration.name.end
        · simp only [empty,ite_true] at computed; simp at computed
        · simp only [empty,ite_false] at computed
          have nameMeaning := source_identifier source token true identifierComputed
          have namePositive := nameMeaning.2.1
          have nameBound := nameMeaning.2.2.1
          have declPositive : declaration.name.start.val<declaration.name.end.val := by scalar_tac
          have declBound : declaration.name.end.val≤source.val.length := by scalar_tac
          let width := declaration.name.end.val-declaration.name.start.val
          have subName := source_usize_sub token.end token.start (by omega)
          have subDecl := source_usize_sub declaration.name.end declaration.name.start (by omega)
          simp only [subName,subDecl,bind_ok] at computed
          split at computed
          · simp at computed
          · have longer : width<token.end.val-token.start.val := by scalar_tac
            have sumBound : token.start.val+width<2^UScalarTy.Usize.numBits := by
              have := Slice.length_ineq source
              scalar_tac
            let middle := Usize.ofNatCore (token.start.val+width) sumBound
            have middleVal : middle.val=token.start.val+width := by simp [middle]
            have added := source_usize_add token.start
              (Usize.ofNatCore width (by scalar_tac)) sumBound
            change (token.start + Usize.ofNatCore width (by scalar_tac) : Result Usize)=ok middle at added
            rw [added] at computed
            simp only [bind_ok] at computed
            obtain ⟨same,sameComputed,computed⟩ := source_bind_success _ _ _ computed
            cases same with
            | false => simp at computed
            | true =>
              simp only [ite_true] at computed
              have prefixMeaning := source_names_equal source declaration.name
                {token with «end»:=middle,word:=true} sameComputed
              have suffixBound : middle.val<token.end.val := by omega
              have remaining := source_usize_sub token.end middle (by omega)
              simp only [remaining,bind_ok] at computed
              have finishProof (suffixComputed : suffixResult source middle token.end declaration.count declaration.first=.ok (some index))
                  (leading : token.end.val=middle.val+1 ∨ ∃ byte,source.val[middle.val]?=some byte ∧ byte.val≠48) :
                  ∃ number,number<declaration.count.val ∧ RegisterName source.val declaration token number ∧
                    index.val=declaration.first.val+number := by
                obtain ⟨number,smaller,digits,indexEq⟩ := source_suffix source middle token.end declaration.count declaration.first index suffixComputed
                refine ⟨number,smaller,?_,indexEq⟩
                simp only [RegisterName,indexed,ite_true]
                refine ⟨declPositive,declBound,by omega,nameBound,?_,?_,?_⟩
                · have prefixLaw := prefix_name source.val token middle (by omega) (by omega)
                  rw [middleVal,Nat.add_sub_cancel_left] at prefixLaw
                  rw [←prefixLaw]
                  exact prefixMeaning.2.2.2.2.symm
                · simpa only [middleVal] using digits
                · simpa only [middleVal] using leading
              split at computed
              · obtain ⟨byte,read,computed⟩ := source_bind_success _ _ _ computed
                by_cases zero : byte=48#u8
                · simp only [zero,ite_true] at computed; simp at computed
                · simp only [zero,ite_false] at computed
                  apply finishProof computed
                  exact Or.inr ⟨byte,source_slice_index source middle byte read,by scalar_tac⟩
              · apply finishProof computed
                exact Or.inl (by scalar_tac)

end PtxFrontendProgram
