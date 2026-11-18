/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.OperandRestrictions
import FrontendProgram.MemoryInstructions
import FrontendProgram.Opcodes
import FrontendProgram.Guards
import FrontendProgram.LabelReferences

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

theorem source_parse_body (source : Slice U8) (tokens : Slice SourceToken) (position finish : Usize)
    (scope : Scope) (opcode : Opcode) (instruction : Instruction) (start : Nat)
    (counts : PositiveCounts scope) (positionEq : position.val=start+1)
    (opcodeMeaning : OpcodeAt source.val tokens.val start opcode)
    (computed : strict_memory_frontend_parse.parse_body source tokens position scope opcode=.ok (some (instruction,finish))) :
    InstructionBody source.val scope tokens.val start finish.val instruction := by
  cases opcode with
  | Exit =>
    simp only [strict_memory_frontend_parse.parse_body,ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
    obtain ⟨same,after⟩ := computed
    cases same
    rw [←after,positionEq]
    exact InstructionBody.exit start opcodeMeaning
  | Nop =>
    simp only [strict_memory_frontend_parse.parse_body,ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
    obtain ⟨same,after⟩ := computed
    cases same
    rw [←after,positionEq]
    exact InstructionBody.nop start opcodeMeaning
  | Branch =>
    simp only [strict_memory_frontend_parse.parse_body] at computed
    split at computed
    · simp at computed
    · obtain ⟨token,tokenComputed,computed⟩ := source_bind_success _ _ _ computed
      obtain ⟨targetResult,targetComputed,computed⟩ := source_bind_success _ _ _ computed
      cases targetResult with
      | none => simp at computed
      | some target =>
        obtain ⟨next,nextComputed,computed⟩ := source_bind_success _ _ _ computed
        simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
        rcases computed with ⟨rfl,rfl⟩
        have nextEq := source_usize_add_value position 1#usize next nextComputed
        have nextAt : next.val=start+2 := by scalar_tac
        rw [nextAt]
        apply InstructionBody.branch target opcodeMeaning
        · simpa [positionEq] using source_slice_index tokens position token tokenComputed
        · exact source_lookup_label source scope token target targetComputed
  | Move bits =>
    exact source_parse_move source tokens position finish scope bits instruction start counts positionEq opcodeMeaning computed
  | Binary bits operation =>
    exact source_parse_binary source tokens position finish scope bits operation instruction start counts positionEq opcodeMeaning computed
  | MultiplyWide =>
    exact source_parse_wide source tokens position finish scope instruction start counts positionEq opcodeMeaning computed
  | Convert fromBits toBits =>
    exact source_parse_convert source tokens position finish scope fromBits toBits instruction start counts positionEq opcodeMeaning computed
  | Compare bits comparison =>
    exact source_parse_comparison source tokens position finish scope bits comparison instruction start counts positionEq opcodeMeaning computed
  | Load bytes signed =>
    exact source_parse_load source tokens position finish scope bytes signed instruction start counts positionEq opcodeMeaning computed
  | Store bytes =>
    exact source_parse_store source tokens position finish scope bytes instruction start counts positionEq opcodeMeaning computed

theorem source_parse_instruction (source : Slice U8) (tokens : Slice SourceToken) (position finish : Usize)
    (scope : Scope) (instruction : GuardedInstruction) (counts : PositiveCounts scope)
    (computed : strict_memory_frontend_parse.parse_instruction source tokens position scope=.ok (some (instruction,finish))) :
    InstructionSyntax source.val scope tokens.val position.val finish.val instruction := by
  unfold strict_memory_frontend_parse.parse_instruction at computed
  obtain ⟨guardResult,guardComputed,computed⟩ := source_bind_success _ _ _ computed
  cases guardResult with
  | none => simp at computed
  | some pair =>
    rcases pair with ⟨guard,opcodePosition⟩
    change (if opcodePosition≥Slice.len tokens then (ok none : Result (Option (GuardedInstruction × Usize))) else _)=_ at computed
    split at computed
    · simp at computed
    · obtain ⟨token,tokenComputed,computed⟩ := source_bind_success _ _ _ computed
      obtain ⟨opcodeResult,opcodeComputed,computed⟩ := source_bind_success _ _ _ computed
      cases opcodeResult with
      | none => simp at computed
      | some opcode =>
        obtain ⟨bodyPosition,bodyAdded,computed⟩ := source_bind_success _ _ _ computed
        obtain ⟨bodyResult,bodyComputed,computed⟩ := source_bind_success _ _ _ computed
        cases bodyResult with
        | none => simp at computed
        | some pair =>
          rcases pair with ⟨body,semicolonPosition⟩
          obtain ⟨supported,supportedComputed,computed⟩ := source_bind_success _ _ _ computed
          have operandMeaning : OperandRestrictions body := by
            cases supported with
            | false => simp at computed
            | true => exact source_instruction_operands body supportedComputed
          have supportedTrue : supported = true := by
            cases supported <;> simp_all
          simp only [supportedTrue,Bool.not_true,Bool.false_eq_true,ite_false,lift,bind_ok] at computed
          obtain ⟨semicolonResult,semicolonComputed,computed⟩ := source_bind_success _ _ _ computed
          cases semicolonResult with
          | none => simp at computed
          | some next =>
            simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
            rcases computed with ⟨rfl,rfl⟩
            have guardMeaning := source_parse_guard source tokens position scope guard opcodePosition counts guardComputed
            obtain ⟨text,listed,spelling⟩ := source_classify source token opcode opcodeComputed
            have opcodeMeaning : OpcodeAt source.val tokens.val opcodePosition.val opcode :=
              ⟨text,listed,token,source_slice_index tokens opcodePosition token tokenComputed,spelling⟩
            have bodyEq := source_usize_add_value opcodePosition 1#usize bodyPosition bodyAdded
            have bodyMeaning := source_parse_body source tokens bodyPosition semicolonPosition scope opcode body opcodePosition.val counts (by simpa using bodyEq) opcodeMeaning bodyComputed
            obtain ⟨finishEq,semicolon,found,meaning⟩ := source_take source tokens semicolonPosition next _ semicolonComputed
            apply InstructionSyntax.instruction guard body guardMeaning bodyMeaning operandMeaning
            · have literal : utf8 ";"=[59#u8] := by decide
              exact ⟨semicolon,found,by simpa [TokenSpelling,literal,Array.to_slice,Array.make] using meaning⟩
            · exact finishEq

end PtxFrontendProgram
