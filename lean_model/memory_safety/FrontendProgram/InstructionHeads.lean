/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.InstructionExtent

open Aeneas Aeneas.Std PtxvMemorySafety
set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192

namespace PtxFrontendProgram

theorem token_at_first_byte (source : List U8) (tokens : List SourceToken) (position : Nat)
    (text : String) (byte : U8) (meaning : TokenAt source tokens position text)
    (first : (utf8 text)[0]?=some byte) :
    ∃token,tokens[position]?=some token ∧ source[token.start.val]?=some byte := by
  obtain ⟨token,selected,spelled⟩:=meaning
  exact ⟨token,selected,spelling_first source token text byte spelled first⟩

theorem token_selected_first (source : List U8) (tokens : List SourceToken) (position : Nat)
    (token : SourceToken) (text : String) (byte : U8) (selected : tokens[position]?=some token)
    (meaning : TokenAt source tokens position text) (first : (utf8 text)[0]?=some byte) :
    source[token.start.val]?=some byte := by
  obtain ⟨other,otherSelected,found⟩:=token_at_first_byte source tokens position text byte meaning first
  have same:=Option.some.inj (selected.symm.trans otherSelected)
  subst other
  exact found

theorem opcode_at_not_text (source : List U8) (tokens : List SourceToken) (position : Nat) (opcode : Opcode)
    (text : String) (byte : U8) (first : (utf8 text)[0]?=some byte)
    (outside : byte.val<97 ∨ 122<byte.val) (meaning : OpcodeAt source tokens position opcode)
    (spelled : TokenAt source tokens position text) : False := by
  obtain ⟨token,other,selected,found,lower,upper⟩:=opcode_at_first source tokens position opcode meaning
  have byteFound:=token_selected_first source tokens position token text byte selected spelled first
  have same : other=byte:=Option.some.inj (found.symm.trans byteFound)
  subst other
  omega

theorem identifier_at_not_text (source : List U8) (tokens : List SourceToken) (position : Nat)
    (token : SourceToken) (text : String) (byte : U8) (first : (utf8 text)[0]?=some byte)
    (outside : ¬InitialCharacter byte) (selected : tokens[position]?=some token)
    (valid : Identifier source token false) (spelled : TokenAt source tokens position text) : False := by
  obtain ⟨other,found,initial⟩:=identifier_false_first source token valid
  have byteFound:=token_selected_first source tokens position token text byte selected spelled first
  have same : other=byte:=Option.some.inj (found.symm.trans byteFound)
  subst other
  exact outside initial

theorem register_at_not_text (source entry : List U8)
    (program : strict_memory_frontend_parse.SourceProgram) (meaning : ModuleMeaning source entry program)
    (tokens : List SourceToken) (position bits : Nat) (predicate : Bool) (index : U16)
    (text : String) (byte : U8) (first : (utf8 text)[0]?=some byte) (outside : byte≠37#u8)
    (reference : RegisterAt source program.scope tokens position bits predicate index)
    (spelled : TokenAt source tokens position text) : False := by
  obtain ⟨token,selected,reference⟩:=reference
  have percent:=module_register_first source entry program meaning token bits predicate index reference
  have byteFound:=token_selected_first source tokens position token text byte selected spelled first
  exact outside (Option.some.inj (byteFound.symm.trans percent))

theorem module_label_first (source entry : List U8)
    (program : strict_memory_frontend_parse.SourceProgram) (meaning : ModuleMeaning source entry program)
    (token : SourceToken) (target : U32) (reference : LabelReference source program.scope token target) :
    ∃byte,source[token.start.val]?=some byte ∧ InitialCharacter byte := by
  obtain ⟨label,member,named,_⟩:=reference
  obtain ⟨byte,selected,initial⟩:=identifier_false_first source label.name
    ((module_scope_identifiers source entry program meaning).2.2 label member)
  exact ⟨byte,(name_equal_first source label.name token named).symm.trans selected,initial⟩

theorem label_reference_not_text (source entry : List U8)
    (program : strict_memory_frontend_parse.SourceProgram) (meaning : ModuleMeaning source entry program)
    (tokens : List SourceToken) (position : Nat) (token : SourceToken) (target : U32)
    (text : String) (byte : U8) (first : (utf8 text)[0]?=some byte) (outside : ¬InitialCharacter byte)
    (selected : tokens[position]?=some token) (reference : LabelReference source program.scope token target)
    (spelled : TokenAt source tokens position text) : False := by
  obtain ⟨other,found,initial⟩:=module_label_first source entry program meaning token target reference
  have byteFound:=token_selected_first source tokens position token text byte selected spelled first
  have same : other=byte:=Option.some.inj (found.symm.trans byteFound)
  subst other
  exact outside initial

theorem guarded_body_extent (source entry : List U8)
    (leftProgram rightProgram : strict_memory_frontend_parse.SourceProgram)
    (leftMeaning : ModuleMeaning source entry leftProgram) (rightMeaning : ModuleMeaning source entry rightProgram)
    (tokens : List SourceToken) (start leftGuarded rightGuarded leftFinish rightFinish : Nat)
    (leftGuard rightGuard : ptxv_memory_write.value_flow.Guard) (leftBody rightBody : Instruction)
    (firstGuard : GuardSyntax source leftProgram.scope tokens start leftGuarded leftGuard)
    (secondGuard : GuardSyntax source rightProgram.scope tokens start rightGuarded rightGuard)
    (first : InstructionBody source leftProgram.scope tokens leftGuarded leftFinish leftBody)
    (second : InstructionBody source rightProgram.scope tokens rightGuarded rightFinish rightBody) : leftFinish=rightFinish := by
  cases firstGuard with
  | unguarded =>
    cases secondGuard with
    | unguarded => exact instruction_body_extent source entry leftProgram rightProgram leftMeaning rightMeaning tokens start _ _ _ _ first second
    | positive index atSign selected reference =>
      exact False.elim (opcode_at_not_text source tokens start _ "@" 64#u8 (by decide) (by decide)
        (instruction_extent_opcode source leftProgram.scope tokens start _ _ first) atSign)
    | negative index atSign =>
      exact False.elim (opcode_at_not_text source tokens start _ "@" 64#u8 (by decide) (by decide)
        (instruction_extent_opcode source leftProgram.scope tokens start _ _ first) atSign)
  | positive index atSign selected reference =>
    cases secondGuard with
    | unguarded =>
      exact False.elim (opcode_at_not_text source tokens start _ "@" 64#u8 (by decide) (by decide)
        (instruction_extent_opcode source rightProgram.scope tokens start _ _ second) atSign)
    | positive => exact instruction_body_extent source entry leftProgram rightProgram leftMeaning rightMeaning tokens (start+2) _ _ _ _ first second
    | negative other _ bang =>
      exact False.elim (register_at_not_text source entry leftProgram leftMeaning tokens (start+1) 1 true index
        "!" 33#u8 (by decide) (by decide) ⟨_,selected,reference⟩ bang)
  | negative index atSign bang selected reference =>
    cases secondGuard with
    | unguarded =>
      exact False.elim (opcode_at_not_text source tokens start _ "@" 64#u8 (by decide) (by decide)
        (instruction_extent_opcode source rightProgram.scope tokens start _ _ second) atSign)
    | positive other _ otherSelected otherReference =>
      exact False.elim (register_at_not_text source entry rightProgram rightMeaning tokens (start+1) 1 true other
        "!" 33#u8 (by decide) (by decide) ⟨_,otherSelected,otherReference⟩ bang)
    | negative => exact instruction_body_extent source entry leftProgram rightProgram leftMeaning rightMeaning tokens (start+3) _ _ _ _ first second

theorem instruction_syntax_extent (source entry : List U8)
    (leftProgram rightProgram : strict_memory_frontend_parse.SourceProgram)
    (leftMeaning : ModuleMeaning source entry leftProgram) (rightMeaning : ModuleMeaning source entry rightProgram)
    (tokens : List SourceToken) (start leftFinish rightFinish : Nat) (left right : GuardedInstruction)
    (first : InstructionSyntax source leftProgram.scope tokens start leftFinish left)
    (second : InstructionSyntax source rightProgram.scope tokens start rightFinish right) : leftFinish=rightFinish := by
  cases first with
  | instruction leftGuard leftBody leftGuardMeaning leftBodyMeaning leftOperands leftEnd leftFinishEq =>
    cases second with
    | instruction rightGuard rightBody rightGuardMeaning rightBodyMeaning rightOperands rightEnd rightFinishEq =>
      have same:=guarded_body_extent source entry leftProgram rightProgram leftMeaning rightMeaning tokens start _ _ _ _
        leftGuard rightGuard leftBody rightBody leftGuardMeaning rightGuardMeaning leftBodyMeaning rightBodyMeaning
      omega


theorem token_at_bytes_distinct (source : List U8) (tokens : List SourceToken) (position : Nat)
    (left right : String) (leftByte rightByte : U8)
    (leftFirst : (utf8 left)[0]?=some leftByte) (rightFirst : (utf8 right)[0]?=some rightByte)
    (different : leftByte≠rightByte) (first : TokenAt source tokens position left)
    (second : TokenAt source tokens position right) : False := by
  obtain ⟨token,selected,found⟩:=token_at_first_byte source tokens position left leftByte first leftFirst
  have otherFound:=token_selected_first source tokens position token right rightByte selected second rightFirst
  exact different (Option.some.inj (found.symm.trans otherFound))

theorem instruction_start_not_text (source : List U8) (scope : Scope) (tokens : List SourceToken)
    (start finish : Nat) (instruction : GuardedInstruction) (text : String) (byte : U8)
    (first : (utf8 text)[0]?=some byte) (outside : byte.val<97 ∨ 122<byte.val) (notAt : byte≠64#u8)
    (meaning : InstructionSyntax source scope tokens start finish instruction)
    (spelled : TokenAt source tokens start text) : False := by
  cases meaning with
  | instruction guard body guardMeaning bodyMeaning =>
    cases guardMeaning with
    | unguarded =>
      exact opcode_at_not_text source tokens start _ text byte first outside
        (instruction_extent_opcode source scope tokens start _ body bodyMeaning) spelled
    | positive index atSign =>
      exact token_at_bytes_distinct source tokens start text "@" byte 64#u8 first (by decide) notAt spelled atSign
    | negative index atSign =>
      exact token_at_bytes_distinct source tokens start text "@" byte 64#u8 first (by decide) notAt spelled atSign

theorem address_syntax_open (source : List U8) (scope : Scope) (tokens : List SourceToken)
    (start finish : Nat) (address : ptxv_memory_write.value_flow.Address)
    (meaning : AddressSyntax source scope tokens start finish address) : TokenAt source tokens start "[" := by
  cases meaning
  assumption

theorem instruction_head_not_label (source entry : List U8)
    (program : strict_memory_frontend_parse.SourceProgram) (meaning : ModuleMeaning source entry program)
    (tokens : List SourceToken) (start finish : Nat) (instruction : GuardedInstruction)
    (token : SourceToken) (selected : tokens[start]?=some token) (valid : Identifier source token false)
    (colon : TokenAt source tokens (start+1) ":")
    (interpreted : InstructionSyntax source program.scope tokens start finish instruction) : False := by
  cases interpreted with
  | instruction guard body guardMeaning bodyMeaning supported semicolon finished =>
    cases guardMeaning with
    | positive index atSign =>
      exact identifier_at_not_text source tokens start token "@" 64#u8 (by decide) (by norm_num [InitialCharacter])
        selected valid atSign
    | negative index atSign =>
      exact identifier_at_not_text source tokens start token "@" 64#u8 (by decide) (by norm_num [InitialCharacter])
        selected valid atSign
    | unguarded =>
      cases bodyMeaning
      all_goals first
        | exact (by decide : utf8 ":"≠utf8 ";") (token_at_spelling_unique source tokens (start+1) ":" ";" colon semicolon)
        | exact label_reference_not_text source entry program meaning tokens (start+1) _ _ ":" 58#u8
            (by decide) (by norm_num [InitialCharacter]) (by assumption) (by assumption) colon
        | exact token_at_bytes_distinct source tokens (start+1) ":" "[" 58#u8 91#u8
            (by decide) (by decide) (by decide) colon
            (address_syntax_open source program.scope tokens (start+1) _ _ (by assumption))
        | exact register_at_not_text source entry program meaning tokens (start+1) _ _ _ ":" 58#u8
            (by decide) (by decide) (by assumption) colon

end PtxFrontendProgram
