/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.ParseOperands
import FrontendProgram.Destinations

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

theorem source_parse_move (source : Slice U8) (tokens : Slice SourceToken) (position finish : Usize)
    (scope : Scope) (bits : U8) (instruction : Instruction) (start : Nat)
    (counts : PositiveCounts scope) (positionEq : position.val=start+1)
    (opcodeMeaning : OpcodeAt source.val tokens.val start (.Move bits))
    (computed : strict_memory_frontend_parse.parse_move source tokens position scope bits=.ok (some (instruction,finish))) :
    InstructionBody source.val scope tokens.val start finish.val instruction := by
  unfold strict_memory_frontend_parse.parse_move at computed
  obtain ⟨destinationResult,destinationComputed,computed⟩ := source_bind_success _ _ _ computed
  cases destinationResult with
  | none => simp at computed
  | some pair =>
    rcases pair with ⟨destination,after⟩
    obtain ⟨sourceResult,sourceComputed,computed⟩ := source_bind_success _ _ _ computed
    cases sourceResult with
    | none => simp at computed
    | some pair =>
      rcases pair with ⟨operand,next⟩
      change ok (some (_,next))=ok (some (instruction,finish)) at computed
      simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
      rcases computed with ⟨rfl,rfl⟩
      obtain ⟨destinationMeaning,comma,afterEq⟩ := source_parse_destination source tokens position after scope bits false destination counts destinationComputed
      have sourceMeaning := source_parse_operand source tokens after scope bits operand next counts sourceComputed
      apply InstructionBody.move bits destination operand opcodeMeaning
      · simpa [positionEq] using destinationMeaning
      · have commaPosition : position.val+1=start+2 := by omega
        simpa [commaPosition] using comma
      · have sourcePosition : after.val=start+3 := by omega
        simpa [sourcePosition] using sourceMeaning

theorem source_parse_convert (source : Slice U8) (tokens : Slice SourceToken) (position finish : Usize)
    (scope : Scope) (fromBits toBits : U8) (instruction : Instruction) (start : Nat)
    (counts : PositiveCounts scope) (positionEq : position.val=start+1)
    (opcodeMeaning : OpcodeAt source.val tokens.val start (.Convert fromBits toBits))
    (computed : strict_memory_frontend_parse.parse_convert source tokens position scope fromBits toBits=.ok (some (instruction,finish))) :
    InstructionBody source.val scope tokens.val start finish.val instruction := by
  unfold strict_memory_frontend_parse.parse_convert at computed
  obtain ⟨destinationResult,destinationComputed,computed⟩ := source_bind_success _ _ _ computed
  cases destinationResult with
  | none => simp at computed
  | some pair =>
    rcases pair with ⟨destination,after⟩
    obtain ⟨sourceResult,sourceComputed,computed⟩ := source_bind_success _ _ _ computed
    cases sourceResult with
    | none => simp at computed
    | some pair =>
      rcases pair with ⟨operand,next⟩
      change ok (some (_,next))=ok (some (instruction,finish)) at computed
      simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
      rcases computed with ⟨rfl,rfl⟩
      obtain ⟨destinationMeaning,comma,afterEq⟩ := source_parse_destination source tokens position after scope toBits false destination counts destinationComputed
      have sourceMeaning := source_parse_operand source tokens after scope fromBits operand next counts sourceComputed
      apply InstructionBody.convert fromBits toBits destination operand opcodeMeaning
      · simpa [positionEq] using destinationMeaning
      · have commaPosition : position.val+1=start+2 := by omega
        simpa [commaPosition] using comma
      · have sourcePosition : after.val=start+3 := by omega
        simpa [sourcePosition] using sourceMeaning

theorem source_parse_binary (source : Slice U8) (tokens : Slice SourceToken) (position finish : Usize)
    (scope : Scope) (bits : U8) (operation : ptxv_memory_write.value_flow.BinaryOperation) (instruction : Instruction) (start : Nat)
    (counts : PositiveCounts scope) (positionEq : position.val=start+1)
    (opcodeMeaning : OpcodeAt source.val tokens.val start (.Binary bits operation))
    (computed : strict_memory_frontend_parse.parse_binary_instruction source tokens position scope bits operation=.ok (some (instruction,finish))) :
    InstructionBody source.val scope tokens.val start finish.val instruction := by
  unfold strict_memory_frontend_parse.parse_binary_instruction at computed
  obtain ⟨destinationResult,destinationComputed,computed⟩ := source_bind_success _ _ _ computed
  cases destinationResult with
  | none => simp at computed
  | some pair =>
    rcases pair with ⟨destination,after⟩
    obtain ⟨leftResult,leftComputed,computed⟩ := source_bind_success _ _ _ computed
    cases leftResult with
    | none => simp at computed
    | some pair =>
      rcases pair with ⟨left,middle⟩
      simp only [lift,bind_ok] at computed
      obtain ⟨commaResult,commaComputed,computed⟩ := source_bind_success _ _ _ computed
      cases commaResult with
      | none => simp at computed
      | some rightPosition =>
        obtain ⟨rightResult,rightComputed,computed⟩ := source_bind_success _ _ _ computed
        cases rightResult with
        | none => simp at computed
        | some pair =>
          rcases pair with ⟨right,next⟩
          change ok (some (_,next))=ok (some (instruction,finish)) at computed
          simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
          rcases computed with ⟨rfl,rfl⟩
          obtain ⟨destinationMeaning,destinationComma,afterEq⟩ := source_parse_destination source tokens position after scope bits false destination counts destinationComputed
          have leftMeaning := source_parse_operand source tokens after scope bits left middle counts leftComputed
          have rightMeaning := source_parse_operand source tokens rightPosition scope bits right next counts rightComputed
          obtain ⟨rightEq,commaToken,commaFound,commaMeaning⟩ := source_take source tokens middle rightPosition _ commaComputed
          have comma : TokenAt source.val tokens.val middle.val "," := by
            have literal : utf8 ","=[44#u8] := by decide
            exact ⟨commaToken,commaFound,by simpa [TokenSpelling,literal,Array.to_slice,Array.make] using commaMeaning⟩
          apply InstructionBody.binary bits destination operation left right opcodeMeaning
          · simpa [positionEq] using destinationMeaning
          · have commaPosition : position.val+1=start+2 := by omega
            simpa [commaPosition] using destinationComma
          · have sourcePosition : after.val=start+3 := by omega
            simpa [sourcePosition] using leftMeaning
          · exact comma
          · simpa [rightEq] using rightMeaning

theorem source_parse_wide (source : Slice U8) (tokens : Slice SourceToken) (position finish : Usize)
    (scope : Scope)  (instruction : Instruction) (start : Nat)
    (counts : PositiveCounts scope) (positionEq : position.val=start+1)
    (opcodeMeaning : OpcodeAt source.val tokens.val start (.MultiplyWide))
    (computed : strict_memory_frontend_parse.parse_wide source tokens position scope =.ok (some (instruction,finish))) :
    InstructionBody source.val scope tokens.val start finish.val instruction := by
  unfold strict_memory_frontend_parse.parse_wide at computed
  obtain ⟨destinationResult,destinationComputed,computed⟩ := source_bind_success _ _ _ computed
  cases destinationResult with
  | none => simp at computed
  | some pair =>
    rcases pair with ⟨destination,after⟩
    obtain ⟨leftResult,leftComputed,computed⟩ := source_bind_success _ _ _ computed
    cases leftResult with
    | none => simp at computed
    | some pair =>
      rcases pair with ⟨left,middle⟩
      simp only [lift,bind_ok] at computed
      obtain ⟨commaResult,commaComputed,computed⟩ := source_bind_success _ _ _ computed
      cases commaResult with
      | none => simp at computed
      | some rightPosition =>
        obtain ⟨rightResult,rightComputed,computed⟩ := source_bind_success _ _ _ computed
        cases rightResult with
        | none => simp at computed
        | some pair =>
          rcases pair with ⟨right,next⟩
          change ok (some (_,next))=ok (some (instruction,finish)) at computed
          simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
          rcases computed with ⟨rfl,rfl⟩
          obtain ⟨destinationMeaning,destinationComma,afterEq⟩ := source_parse_destination source tokens position after scope 64#u8 false destination counts destinationComputed
          have leftMeaning := source_parse_operand source tokens after scope 32#u8 left middle counts leftComputed
          have rightMeaning := source_parse_operand source tokens rightPosition scope 32#u8 right next counts rightComputed
          obtain ⟨rightEq,commaToken,commaFound,commaMeaning⟩ := source_take source tokens middle rightPosition _ commaComputed
          have comma : TokenAt source.val tokens.val middle.val "," := by
            have literal : utf8 ","=[44#u8] := by decide
            exact ⟨commaToken,commaFound,by simpa [TokenSpelling,literal,Array.to_slice,Array.make] using commaMeaning⟩
          apply InstructionBody.wide destination left right opcodeMeaning
          · simpa [positionEq] using destinationMeaning
          · have commaPosition : position.val+1=start+2 := by omega
            simpa [commaPosition] using destinationComma
          · have sourcePosition : after.val=start+3 := by omega
            simpa [sourcePosition] using leftMeaning
          · exact comma
          · simpa [rightEq] using rightMeaning

theorem source_parse_comparison (source : Slice U8) (tokens : Slice SourceToken) (position finish : Usize)
    (scope : Scope) (bits : U8) (comparison : ptxv_memory_write.value_flow.Comparison) (instruction : Instruction) (start : Nat)
    (counts : PositiveCounts scope) (positionEq : position.val=start+1)
    (opcodeMeaning : OpcodeAt source.val tokens.val start (.Compare bits comparison))
    (computed : strict_memory_frontend_parse.parse_comparison_instruction source tokens position scope bits comparison=.ok (some (instruction,finish))) :
    InstructionBody source.val scope tokens.val start finish.val instruction := by
  unfold strict_memory_frontend_parse.parse_comparison_instruction at computed
  obtain ⟨destinationResult,destinationComputed,computed⟩ := source_bind_success _ _ _ computed
  cases destinationResult with
  | none => simp at computed
  | some pair =>
    rcases pair with ⟨destination,after⟩
    obtain ⟨leftResult,leftComputed,computed⟩ := source_bind_success _ _ _ computed
    cases leftResult with
    | none => simp at computed
    | some pair =>
      rcases pair with ⟨left,middle⟩
      simp only [lift,bind_ok] at computed
      obtain ⟨commaResult,commaComputed,computed⟩ := source_bind_success _ _ _ computed
      cases commaResult with
      | none => simp at computed
      | some rightPosition =>
        obtain ⟨rightResult,rightComputed,computed⟩ := source_bind_success _ _ _ computed
        cases rightResult with
        | none => simp at computed
        | some pair =>
          rcases pair with ⟨right,next⟩
          change ok (some (_,next))=ok (some (instruction,finish)) at computed
          simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
          rcases computed with ⟨rfl,rfl⟩
          obtain ⟨destinationMeaning,destinationComma,afterEq⟩ := source_parse_destination source tokens position after scope 1#u8 true destination counts destinationComputed
          have leftMeaning := source_parse_operand source tokens after scope bits left middle counts leftComputed
          have rightMeaning := source_parse_operand source tokens rightPosition scope bits right next counts rightComputed
          obtain ⟨rightEq,commaToken,commaFound,commaMeaning⟩ := source_take source tokens middle rightPosition _ commaComputed
          have comma : TokenAt source.val tokens.val middle.val "," := by
            have literal : utf8 ","=[44#u8] := by decide
            exact ⟨commaToken,commaFound,by simpa [TokenSpelling,literal,Array.to_slice,Array.make] using commaMeaning⟩
          apply InstructionBody.compare bits destination comparison left right opcodeMeaning
          · simpa [positionEq] using destinationMeaning
          · have commaPosition : position.val+1=start+2 := by omega
            simpa [commaPosition] using destinationComma
          · have sourcePosition : after.val=start+3 := by omega
            simpa [sourcePosition] using leftMeaning
          · exact comma
          · simpa [rightEq] using rightMeaning

end PtxFrontendProgram
