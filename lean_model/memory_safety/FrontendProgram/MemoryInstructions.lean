/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.AddressTails
import FrontendProgram.ArithmeticInstructions

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

theorem source_parse_store (source : Slice U8) (tokens : Slice SourceToken) (position finish : Usize)
    (scope : Scope) (bytes : U8) (instruction : Instruction) (start : Nat)
    (counts : PositiveCounts scope) (positionEq : position.val=start+1)
    (opcodeMeaning : OpcodeAt source.val tokens.val start (.Store bytes))
    (computed : strict_memory_frontend_parse.parse_store source tokens position scope bytes=.ok (some (instruction,finish))) :
    InstructionBody source.val scope tokens.val start finish.val instruction := by
  unfold strict_memory_frontend_parse.parse_store at computed
  obtain ⟨addressResult,addressComputed,computed⟩ := source_bind_success _ _ _ computed
  cases addressResult with
  | none => simp at computed
  | some pair =>
    rcases pair with ⟨address,middle⟩
    simp only [lift,bind_ok] at computed
    obtain ⟨commaResult,commaComputed,computed⟩ := source_bind_success _ _ _ computed
    cases commaResult with
    | none => simp at computed
    | some operandPosition =>
      obtain ⟨bits,bitsComputed,computed⟩ := source_bind_success _ _ _ computed
      obtain ⟨operandResult,operandComputed,computed⟩ := source_bind_success _ _ _ computed
      cases operandResult with
      | none => simp at computed
      | some pair =>
        rcases pair with ⟨operand,next⟩
        change ok (some ((.Store bytes address operand : Instruction),next))=ok (some (instruction,finish)) at computed
        simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
        rcases computed with ⟨rfl,rfl⟩
        have bitsVal : bits.val=(if bytes.val=8 then 64 else 32) := by
          by_cases byte8 : bytes=8#u8
          · simp only [byte8,ite_true,ok.injEq] at bitsComputed
            cases bitsComputed
            simp [byte8]
          · have byteVal8 : bytes.val≠8 := by scalar_tac
            simp only [byte8,ite_false,ok.injEq] at bitsComputed
            cases bitsComputed
            simp [byteVal8]
        obtain ⟨operandEq,commaToken,commaFound,commaMeaning⟩ := source_take source tokens middle operandPosition _ commaComputed
        apply InstructionBody.store bytes address operand opcodeMeaning
        · simpa [positionEq] using source_parse_address source tokens position middle scope address counts addressComputed
        · have literal : utf8 ","=[44#u8] := by decide
          exact ⟨commaToken,commaFound,by simpa [TokenSpelling,literal,Array.to_slice,Array.make] using commaMeaning⟩
        · have meaning := source_parse_operand source tokens operandPosition scope bits operand next counts operandComputed
          simpa only [bitsVal,operandEq] using meaning

theorem source_parse_load (source : Slice U8) (tokens : Slice SourceToken) (position finish : Usize)
    (scope : Scope) (bytes : U8) (signed : Bool) (instruction : Instruction) (start : Nat)
    (counts : PositiveCounts scope) (positionEq : position.val=start+1)
    (opcodeMeaning : OpcodeAt source.val tokens.val start (.Load bytes signed))
    (computed : strict_memory_frontend_parse.parse_load source tokens position scope bytes signed=.ok (some (instruction,finish))) :
    InstructionBody source.val scope tokens.val start finish.val instruction := by
  unfold strict_memory_frontend_parse.parse_load at computed
  dsimp only at computed
  split at computed
  · simp at computed
  · obtain ⟨token,tokenComputed,computed⟩ := source_bind_success _ _ _ computed
    by_cases byte8 : bytes=8#u8
    · simp only [byte8,ite_true] at computed
      obtain ⟨register,registerComputed,computed⟩ := source_bind_success _ _ _ computed
      cases register with
      | none => simp at computed
      | some destination =>
        obtain ⟨commaPosition,commaAdded,computed⟩ := source_bind_success _ _ _ computed
        simp only [lift,bind_ok] at computed
        obtain ⟨commaResult,commaComputed,computed⟩ := source_bind_success _ _ _ computed
        cases commaResult with
        | none => simp at computed
        | some addressPosition =>
          obtain ⟨addressResult,addressComputed,computed⟩ := source_bind_success _ _ _ computed
          cases addressResult with
          | none => simp at computed
          | some pair =>
            rcases pair with ⟨address,next⟩
            rw [←byte8] at computed
            change ok (some ((.Load destination 64#u8 bytes signed address : Instruction),next))=ok (some (instruction,finish)) at computed
            simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
            rcases computed with ⟨rfl,rfl⟩
            have registerMeaning := source_lookup_register source scope token 64#u8 false destination counts registerComputed
            have commaEq := source_usize_add_value position 1#usize commaPosition commaAdded
            obtain ⟨addressEq,commaToken,commaFound,commaMeaning⟩ := source_take source tokens commaPosition addressPosition _ commaComputed
            apply InstructionBody.load bytes 64#u8 destination signed address opcodeMeaning
            · exact Or.inr rfl
            · intro _;rfl
            · exact ⟨token,by simpa [positionEq] using source_slice_index tokens position token tokenComputed,registerMeaning⟩
            · have commaAt : commaPosition.val=start+2 := by scalar_tac
              have literal : utf8 ","=[44#u8] := by decide
              exact ⟨commaToken,by simpa [commaAt] using commaFound,by simpa [TokenSpelling,literal,Array.to_slice,Array.make] using commaMeaning⟩
            · have addressAt : addressPosition.val=start+3 := by scalar_tac
              simpa [addressAt] using source_parse_address source tokens addressPosition next scope address counts addressComputed
    · simp only [byte8,ite_false] at computed
      obtain ⟨register,registerComputed,computed⟩ := source_bind_success _ _ _ computed
      cases register with
      | some destination =>
        obtain ⟨commaPosition,commaAdded,computed⟩ := source_bind_success _ _ _ computed
        simp only [lift,bind_ok] at computed
        obtain ⟨commaResult,commaComputed,computed⟩ := source_bind_success _ _ _ computed
        cases commaResult with
        | none => simp at computed
        | some addressPosition =>
          obtain ⟨addressResult,addressComputed,computed⟩ := source_bind_success _ _ _ computed
          cases addressResult with
          | none => simp at computed
          | some pair =>
            rcases pair with ⟨address,next⟩
            change ok (some ((.Load destination 32#u8 bytes signed address : Instruction),next))=ok (some (instruction,finish)) at computed
            simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
            rcases computed with ⟨rfl,rfl⟩
            have registerMeaning := source_lookup_register source scope token 32#u8 false destination counts registerComputed
            have commaEq := source_usize_add_value position 1#usize commaPosition commaAdded
            obtain ⟨addressEq,commaToken,commaFound,commaMeaning⟩ := source_take source tokens commaPosition addressPosition _ commaComputed
            apply InstructionBody.load bytes 32#u8 destination signed address opcodeMeaning
            · exact Or.inl rfl
            · intro byteVal8
              have impossible : False := by scalar_tac
              exact impossible.elim
            · exact ⟨token,by simpa [positionEq] using source_slice_index tokens position token tokenComputed,registerMeaning⟩
            · have commaAt : commaPosition.val=start+2 := by scalar_tac
              have literal : utf8 ","=[44#u8] := by decide
              exact ⟨commaToken,by simpa [commaAt] using commaFound,by simpa [TokenSpelling,literal,Array.to_slice,Array.make] using commaMeaning⟩
            · have addressAt : addressPosition.val=start+3 := by scalar_tac
              simpa [addressAt] using source_parse_address source tokens addressPosition next scope address counts addressComputed
      | none =>
        obtain ⟨register,registerComputed,computed⟩ := source_bind_success _ _ _ computed
        cases register with
        | none => simp at computed
        | some destination =>
          obtain ⟨commaPosition,commaAdded,computed⟩ := source_bind_success _ _ _ computed
          simp only [lift,bind_ok] at computed
          obtain ⟨commaResult,commaComputed,computed⟩ := source_bind_success _ _ _ computed
          cases commaResult with
          | none => simp at computed
          | some addressPosition =>
            obtain ⟨addressResult,addressComputed,computed⟩ := source_bind_success _ _ _ computed
            cases addressResult with
            | none => simp at computed
            | some pair =>
              rcases pair with ⟨address,next⟩
              change ok (some ((.Load destination 64#u8 bytes signed address : Instruction),next))=ok (some (instruction,finish)) at computed
              simp only [ok.injEq,Option.some.injEq,Prod.mk.injEq] at computed
              rcases computed with ⟨rfl,rfl⟩
              have registerMeaning := source_lookup_register source scope token 64#u8 false destination counts registerComputed
              have commaEq := source_usize_add_value position 1#usize commaPosition commaAdded
              obtain ⟨addressEq,commaToken,commaFound,commaMeaning⟩ := source_take source tokens commaPosition addressPosition _ commaComputed
              apply InstructionBody.load bytes 64#u8 destination signed address opcodeMeaning
              · exact Or.inr rfl
              · intro _;rfl
              · exact ⟨token,by simpa [positionEq] using source_slice_index tokens position token tokenComputed,registerMeaning⟩
              · have commaAt : commaPosition.val=start+2 := by scalar_tac
                have literal : utf8 ","=[44#u8] := by decide
                exact ⟨commaToken,by simpa [commaAt] using commaFound,by simpa [TokenSpelling,literal,Array.to_slice,Array.make] using commaMeaning⟩
              · have addressAt : addressPosition.val=start+3 := by scalar_tac
                simpa [addressAt] using source_parse_address source tokens addressPosition next scope address counts addressComputed

end PtxFrontendProgram
