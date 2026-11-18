/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.GlobalInitializers
import Init.Data.Nat.Power2.Bitwise

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

theorem source_some_bind {α β : Type} (operation : Result (Option α))
    (continuation : α → Result (Option β)) (output : β)
    (computed : (do let value ← operation; match value with
      | none => .ok none
      | some selected => continuation selected) = .ok (some output)) :
    ∃ selected, operation = .ok (some selected) ∧ continuation selected = .ok (some output) := by
  obtain ⟨value,first,rest⟩ := source_bind_success _ _ _ computed
  cases value with
  | none => simp at rest
  | some selected => exact ⟨selected,first,rest⟩

theorem source_taken_token (source : Slice U8) (tokens : Slice SourceToken)
    (position after : Usize) (expected : Slice U8) (spelling : String)
    (computed : strict_memory_frontend_parse.take source tokens position expected=.ok (some after))
    (literal : expected.val=utf8 spelling) : TokenAt source.val tokens.val position.val spelling := by
  obtain ⟨_,token,found,meaning⟩ := source_take source tokens position after expected computed
  exact ⟨token,found,by simpa only [TokenSpelling,literal] using meaning⟩

theorem global_type_width (source : List U8) (token : SourceToken) (width : U8)
    (meaning : GlobalElementType source token width.val) : InitializerWidth width := by
  rcases meaning with ⟨text,member,_⟩
  simp only [List.mem_cons,List.not_mem_nil,or_false,Prod.mk.injEq] at member
  rcases member with h|h|h|h|h|h|h|h <;> rcases h with ⟨_,h⟩ <;>
    unfold InitializerWidth <;> omega

theorem source_parse_global (source : Slice U8) (tokens : Slice SourceToken)
    (position finish : Usize) (scope : strict_memory_frontend_parse.Scope)
    (global : Global)
    (computed : strict_memory_frontend_parse.parse_global source tokens position scope = .ok (some (global,finish))) :
    GlobalSyntax source.val tokens.val position.val finish.val global ∧
      strict_memory_frontend_parse.lookup_global source scope global.name = .ok none := by
  unfold strict_memory_frontend_parse.parse_global at computed
  try simp only [lift,bind_ok] at computed
  obtain ⟨option,globalRun,computed⟩ := source_bind_success _ _ _ computed
  cases option with
  | none => simp at computed
  | some alignPosition =>
    try dsimp (config := { zeta := true }) only [Prod.casesOn] at computed
    try simp only [lift,bind_ok] at computed
    obtain ⟨option,alignRun,computed⟩ := source_bind_success _ _ _ computed
    cases option with
    | none => simp at computed
    | some alignmentPosition =>
      try dsimp (config := { zeta := true }) only [Prod.casesOn] at computed
      obtain ⟨option,alignmentRun,computed⟩ := source_bind_success _ _ _ computed
      cases option with
      | none => simp at computed
      | some alignmentPair =>
        try dsimp (config := { zeta := true }) only [Prod.casesOn] at computed
        rcases alignmentPair with ⟨alignment,typePosition⟩
        change (if alignment > 4294967295#u64 then _ else _) = Result.ok (some (global,finish)) at computed
        by_cases tooLarge : alignment > 4294967295#u64
        · simp only [tooLarge,ite_true] at computed
          simp at computed
        simp only [tooLarge,ite_false] at computed
        change (if alignment = 0#u64 then _ else _) = Result.ok (some (global,finish)) at computed
        split at computed
        · simp at computed
        · rename_i nonzero
          obtain ⟨previous,subtractRun,computed⟩ := source_bind_success _ _ _ computed
          try simp only [lift,bind_ok] at computed
          split at computed
          · simp at computed
          · rename_i powerTest
            obtain ⟨namePosition,namePositionRun,computed⟩ := source_bind_success _ _ _ computed
            split at computed
            · simp at computed
            · rename_i inBounds
              obtain ⟨typeToken,typeRead,computed⟩ := source_bind_success _ _ _ computed
              obtain ⟨option,widthRun,computed⟩ := source_bind_success _ _ _ computed
              cases option with
              | none => simp at computed
              | some width =>
                try dsimp (config := { zeta := true }) only [Prod.casesOn] at computed
                try simp only [lift,bind_ok] at computed
                split at computed
                · simp at computed
                · rename_i wideEnough
                  obtain ⟨symbolToken,nameRead,computed⟩ := source_bind_success _ _ _ computed
                  obtain ⟨validName,nameRun,computed⟩ := source_bind_success _ _ _ computed
                  cases validName with
                  | false => simp at computed
                  | true =>
                    simp only [ite_true] at computed
                    obtain ⟨existing,lookupRun,computed⟩ := source_bind_success _ _ _ computed
                    cases existing with
                    | some value => simp [core.option.Option.is_some] at computed
                    | none =>
                      simp only [core.option.Option.is_some, Bool.false_eq_true,ite_false] at computed
                      obtain ⟨bracketPosition,bracketPositionRun,computed⟩ := source_bind_success _ _ _ computed
                      try simp only [lift,bind_ok] at computed
                      obtain ⟨option,bracketRun,computed⟩ := source_bind_success _ _ _ computed
                      cases option with
                      | none => simp at computed
                      | some countPosition =>
                        try dsimp (config := { zeta := true }) only [Prod.casesOn] at computed
                        obtain ⟨option,countRun,computed⟩ := source_bind_success _ _ _ computed
                        cases option with
                        | none => simp at computed
                        | some countPair =>
                          try dsimp (config := { zeta := true }) only [Prod.casesOn] at computed
                          rcases countPair with ⟨count,closePosition⟩
                          change (if count=0#u64 then _ else _) = Result.ok (some (global,finish)) at computed
                          split at computed
                          · simp at computed
                          · rename_i positive
                            try simp only [lift,bind_ok] at computed
                            obtain ⟨maximum,divideRun,computed⟩ := source_bind_success _ _ _ computed
                            split at computed
                            · simp at computed
                            · rename_i fits
                              try simp only [lift,bind_ok] at computed
                              obtain ⟨option,closeRun,computed⟩ := source_bind_success _ _ _ computed
                              cases option with
                              | none => simp at computed
                              | some equalPosition =>
                                try dsimp (config := { zeta := true }) only [Prod.casesOn] at computed
                                try simp only [lift,bind_ok] at computed
                                obtain ⟨option,equalRun,computed⟩ := source_bind_success _ _ _ computed
                                cases option with
                                | none => simp at computed
                                | some openPosition =>
                                  try dsimp (config := { zeta := true }) only [Prod.casesOn] at computed
                                  try simp only [lift,bind_ok] at computed
                                  obtain ⟨option,openRun,computed⟩ := source_bind_success _ _ _ computed
                                  cases option with
                                  | none => simp at computed
                                  | some valuesPosition =>
                                    try dsimp (config := { zeta := true }) only [Prod.casesOn] at computed
                                    obtain ⟨initialized,initializerRun,computed⟩ := source_bind_success _ _ _ computed
                                    rcases initialized with ⟨endPosition,bytes,valid⟩
                                    change (if valid then _ else _) = Result.ok (some (global,finish)) at computed
                                    cases valid with
                                    | false => simp at computed
                                    | true =>
                                      simp only [ite_true,lift,bind_ok] at computed
                                      obtain ⟨option,endRun,computed⟩ := source_bind_success _ _ _ computed
                                      cases option with
                                      | none => simp at computed
                                      | some semicolonPosition =>
                                        try dsimp (config := { zeta := true }) only [Prod.casesOn] at computed
                                        try simp only [lift,bind_ok] at computed
                                        obtain ⟨option,semicolonRun,computed⟩ := source_bind_success _ _ _ computed
                                        cases option with
                                        | none => simp at computed
                                        | some finalPosition =>
                                          try dsimp (config := { zeta := true }) only [Prod.casesOn] at computed
                                          simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
                                          rcases computed with ⟨rfl,rfl⟩
                                          have globalToken := source_taken_token source tokens position alignPosition _ ".global" globalRun (by decide)
                                          have alignToken := source_taken_token source tokens alignPosition alignmentPosition _ ".align" alignRun (by decide)
                                          have bracketToken := source_taken_token source tokens bracketPosition countPosition _ "[" bracketRun (by decide)
                                          have closeToken := source_taken_token source tokens closePosition equalPosition _ "]" closeRun (by decide)
                                          have equalToken := source_taken_token source tokens equalPosition openPosition _ "=" equalRun (by decide)
                                          have openToken := source_taken_token source tokens openPosition valuesPosition _ "{" openRun (by decide)
                                          have endToken := source_taken_token source tokens endPosition semicolonPosition _ "}" endRun (by decide)
                                          have semicolonToken := source_taken_token source tokens semicolonPosition finalPosition _ ";" semicolonRun (by decide)
                                          have globalMeaning := source_take source tokens position alignPosition _ globalRun
                                          have alignMeaning := source_take source tokens alignPosition alignmentPosition _ alignRun
                                          have alignmentMeaning := source_unsigned source tokens alignmentPosition typePosition alignment alignmentRun
                                          have typeMeaning := source_global_type source typeToken width widthRun
                                          have supported := global_type_width source.val typeToken width typeMeaning
                                          have nameIndex := source_slice_index tokens namePosition symbolToken nameRead
                                          have typeIndex := source_slice_index tokens typePosition typeToken typeRead
                                          have nameMeaning := source_identifier source symbolToken false nameRun
                                          have bracketMeaning := source_take source tokens bracketPosition countPosition _ bracketRun
                                          have countMeaning := source_unsigned source tokens countPosition closePosition count countRun
                                          have closeMeaning := source_take source tokens closePosition equalPosition _ closeRun
                                          have equalMeaning := source_take source tokens equalPosition openPosition _ equalRun
                                          have openMeaning := source_take source tokens openPosition valuesPosition _ openRun
                                          obtain ⟨values,initializerGrammar,valueLength,bytesMeaning⟩ := source_initializer_values source tokens width count valuesPosition endPosition
                                            (alloc.vec.Vec.new U8) bytes 0#u64 supported (by scalar_tac) initializerRun
                                          have endMeaning := source_take source tokens endPosition semicolonPosition _ endRun
                                          have semicolonMeaning := source_take source tokens semicolonPosition finalPosition _ semicolonRun
                                          have namePositionSpec := UScalar.add_equiv typePosition 1#usize
                                          rw [namePositionRun] at namePositionSpec
                                          have namePositionEq : namePosition.val=typePosition.val+1 := by simpa using namePositionSpec.2.1
                                          have bracketPositionSpec := UScalar.add_equiv typePosition 2#usize
                                          rw [bracketPositionRun] at bracketPositionSpec
                                          have bracketPositionEq : bracketPosition.val=typePosition.val+2 := by simpa using bracketPositionSpec.2.1
                                          have subSpec := UScalar.sub_spec (x:=alignment) (y:=1#u64) (by scalar_tac)
                                          rw [subtractRun] at subSpec
                                          have previousEq : previous.val=alignment.val-1 := by simpa [WP.spec_ok] using (show previous.val=alignment.val-1 ∧ 1≤alignment.val from by simpa [WP.spec_ok] using subSpec).1
                                          have andZero : alignment.val &&& (alignment.val-1)=0 := by
                                            have equal : alignment &&& previous = 0#u64 := by simpa only [bne_iff_ne,ne_eq,not_not] using powerTest
                                            have val := congrArg UScalar.val equal
                                            simpa [UScalar.val_and,previousEq] using val
                                          have power : ∃ exponent,alignment.val=2^exponent := (Nat.and_sub_one_eq_zero_iff_isPowerOfTwo (by scalar_tac)).mp andZero
                                          have castWidth : (UScalar.cast .U64 width).val=width.val := by
                                            rw [UScalar.cast_val_eq]
                                            exact Nat.mod_eq_of_lt (by scalar_tac)
                                          refine ⟨?_,lookupRun⟩
                                          apply GlobalSyntax.global (width:=width.val) (count:=count.val) (values:=values)
                                            (typeToken:=typeToken) («name»:=symbolToken) (alignment:=alignment.val) (middle:=endPosition.val)
                                          all_goals first
                                            | assumption
                                            | rfl
                                            | (simp only [alloc.vec.Vec.new,List.nil_append] at bytesMeaning; exact bytesMeaning)
                                            | (simp only [UScalar.ofNatCore_val_eq] at valueLength; omega)
                                            | (convert globalToken using 1 <;> omega)
                                            | (convert alignToken using 1 <;> omega)
                                            | (convert bracketToken using 1 <;> omega)
                                            | (convert closeToken using 1 <;> omega)
                                            | (convert equalToken using 1 <;> omega)
                                            | (convert openToken using 1 <;> omega)
                                            | (convert endToken using 1 <;> omega)
                                            | (convert semicolonToken using 1 <;> omega)
                                            | (convert alignmentMeaning.2 using 1 <;> omega)
                                            | (convert countMeaning.2 using 1 <;> omega)
                                            | (convert typeIndex using 1 <;> omega)
                                            | (convert nameIndex using 1 <;> omega)
                                            | (convert initializerGrammar using 1 <;> omega)
                                            | scalar_tac


end PtxFrontendProgram
