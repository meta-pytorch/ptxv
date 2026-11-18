/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.RegisterReferenceUnique

open Aeneas Aeneas.Std PtxvMemorySafety
set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 16000000
set_option maxRecDepth 8192
namespace PtxFrontendProgram

def instructionOpcode : Instruction → Opcode
  | .Exit => .Exit
  | .Nop => .Nop
  | .Branch _ => .Branch
  | .Move bits _ _ => .Move bits
  | .Binary bits _ operation _ _ => .Binary bits operation
  | .MultiplyWide _ _ _ => .MultiplyWide
  | .ConvertUnsigned _ _ fromBits toBits => .Convert fromBits toBits
  | .Compare bits _ comparison _ _ => .Compare bits comparison
  | .Load _ _ bytes signed _ => .Load bytes signed
  | .Store bytes _ _ => .Store bytes

theorem instruction_body_opcode (source : List U8) (scope : Scope) (tokens : List SourceToken)
    (start finish : Nat) (instruction : Instruction)
    (meaning : InstructionBody source scope tokens start finish instruction) :
    OpcodeAt source tokens start (instructionOpcode instruction) := by
  cases meaning <;> assumption

theorem module_instruction_body_unique (source entry : List U8)
    (program : strict_memory_frontend_parse.SourceProgram) (meaning : ModuleMeaning source entry program)
    (tokens : List SourceToken) (start leftFinish rightFinish : Nat) (left right : Instruction)
    (first : InstructionBody source program.scope tokens start leftFinish left)
    (second : InstructionBody source program.scope tokens start rightFinish right) :
    leftFinish=rightFinish ∧ left=right := by
  have opcodeEq:=opcode_at_unique source tokens start _ _
    (instruction_body_opcode source program.scope tokens start leftFinish left first)
    (instruction_body_opcode source program.scope tokens start rightFinish right second)
  cases first <;> cases second <;> simp only [instructionOpcode] at opcodeEq <;> cases opcodeEq
  case exit.exit.intro => exact ⟨rfl,rfl⟩
  case nop.nop.intro => exact ⟨rfl,rfl⟩
  case branch.branch.intro =>
    rename_i leftToken leftTarget leftOpcode leftSelected leftReference rightToken rightTarget rightOpcode rightSelected rightReference
    have sameToken:=Option.some.inj (leftSelected.symm.trans rightSelected)
    subst rightToken
    have sameTarget:=module_label_reference_unique source entry program meaning leftToken leftTarget rightTarget leftReference rightReference
    subst rightTarget
    exact ⟨rfl,rfl⟩
  case move.move.refl =>
    rename_i bits leftDestination leftOperand leftOpcode leftRegister leftComma leftMeaning rightDestination rightOperand rightComma rightOpcode rightRegister rightMeaning
    have sameDestination:=(module_register_at_full_unique source entry program meaning tokens (start+1) bits.val bits.val false false leftDestination rightDestination leftRegister rightRegister).2.2
    obtain ⟨sameFinish,sameOperand⟩:=module_operand_syntax_unique source entry program meaning tokens bits.val (start+3) leftFinish rightFinish leftOperand rightOperand leftMeaning rightMeaning
    subst rightDestination
    subst rightOperand
    exact ⟨sameFinish,rfl⟩
  case binary.binary.refl =>
    rename_i leftMiddle bits leftDestination operation leftFirst leftSecond leftOpcode leftRegister leftComma leftFirstMeaning leftSeparator leftSecondMeaning rightMiddle rightDestination rightFirst rightSecond rightComma rightSeparator rightRegister rightFirstMeaning rightSecondMeaning rightOpcode
    have sameDestination:=(module_register_at_full_unique source entry program meaning tokens (start+1) bits.val bits.val false false leftDestination rightDestination leftRegister rightRegister).2.2
    obtain ⟨sameMiddle,sameFirst⟩:=module_operand_syntax_unique source entry program meaning tokens bits.val (start+3) leftMiddle rightMiddle leftFirst rightFirst leftFirstMeaning rightFirstMeaning
    subst rightMiddle
    obtain ⟨sameFinish,sameSecond⟩:=module_operand_syntax_unique source entry program meaning tokens bits.val (leftMiddle+1) leftFinish rightFinish leftSecond rightSecond leftSecondMeaning rightSecondMeaning
    subst rightDestination
    subst rightFirst
    subst rightSecond
    exact ⟨sameFinish,rfl⟩
  case wide.wide.intro =>
    rename_i leftMiddle leftDestination leftFirst leftSecond leftOpcode leftRegister leftComma leftFirstMeaning leftSeparator leftSecondMeaning rightMiddle rightDestination rightFirst rightSecond rightOpcode rightRegister rightComma rightFirstMeaning rightSeparator rightSecondMeaning
    have sameDestination:=(module_register_at_full_unique source entry program meaning tokens (start+1) 64 64 false false leftDestination rightDestination leftRegister rightRegister).2.2
    obtain ⟨sameMiddle,sameFirst⟩:=module_operand_syntax_unique source entry program meaning tokens 32 (start+3) leftMiddle rightMiddle leftFirst rightFirst leftFirstMeaning rightFirstMeaning
    subst rightMiddle
    obtain ⟨sameFinish,sameSecond⟩:=module_operand_syntax_unique source entry program meaning tokens 32 (leftMiddle+1) leftFinish rightFinish leftSecond rightSecond leftSecondMeaning rightSecondMeaning
    subst rightDestination
    subst rightFirst
    subst rightSecond
    exact ⟨sameFinish,rfl⟩
  case convert.convert.refl =>
    rename_i fromBits toBits leftDestination leftOperand leftOpcode leftRegister leftComma leftMeaning rightDestination rightOperand rightComma rightMeaning rightRegister rightOpcode
    have sameDestination:=(module_register_at_full_unique source entry program meaning tokens (start+1) toBits.val toBits.val false false leftDestination rightDestination leftRegister rightRegister).2.2
    obtain ⟨sameFinish,sameOperand⟩:=module_operand_syntax_unique source entry program meaning tokens fromBits.val (start+3) leftFinish rightFinish leftOperand rightOperand leftMeaning rightMeaning
    subst rightDestination
    subst rightOperand
    exact ⟨sameFinish,rfl⟩
  case compare.compare.refl =>
    rename_i leftMiddle bits leftDestination comparison leftFirst leftSecond leftOpcode leftRegister leftComma leftFirstMeaning leftSeparator leftSecondMeaning rightMiddle rightDestination rightFirst rightSecond rightRegister rightComma rightSeparator rightFirstMeaning rightSecondMeaning rightOpcode
    have sameDestination:=(module_register_at_full_unique source entry program meaning tokens (start+1) 1 1 true true leftDestination rightDestination leftRegister rightRegister).2.2
    obtain ⟨sameMiddle,sameFirst⟩:=module_operand_syntax_unique source entry program meaning tokens bits.val (start+3) leftMiddle rightMiddle leftFirst rightFirst leftFirstMeaning rightFirstMeaning
    subst rightMiddle
    obtain ⟨sameFinish,sameSecond⟩:=module_operand_syntax_unique source entry program meaning tokens bits.val (leftMiddle+1) leftFinish rightFinish leftSecond rightSecond leftSecondMeaning rightSecondMeaning
    subst rightDestination
    subst rightFirst
    subst rightSecond
    exact ⟨sameFinish,rfl⟩
  case load.load.refl =>
    rename_i bytes leftBits leftDestination signed leftAddress leftOpcode leftWidth leftWide leftRegister leftComma leftMeaning rightBits rightDestination rightAddress rightWidth rightRegister rightComma rightMeaning rightWide rightOpcode
    obtain ⟨sameBits,_,sameDestination⟩:=module_register_at_full_unique source entry program meaning tokens (start+1) leftBits.val rightBits.val false false leftDestination rightDestination leftRegister rightRegister
    have equalBits:leftBits=rightBits:=UScalar.eq_of_val_eq sameBits
    obtain ⟨sameFinish,sameAddress⟩:=module_address_syntax_unique source entry program meaning tokens (start+3) leftFinish rightFinish leftAddress rightAddress leftMeaning rightMeaning
    subst rightBits
    subst rightDestination
    subst rightAddress
    exact ⟨sameFinish,rfl⟩
  case store.store.refl =>
    rename_i leftMiddle bytes leftAddress leftOperand leftOpcode leftAddressMeaning leftComma leftMeaning rightMiddle rightAddress rightOperand rightAddressMeaning rightComma rightOpcode rightMeaning
    obtain ⟨sameMiddle,sameAddress⟩:=module_address_syntax_unique source entry program meaning tokens (start+1) leftMiddle rightMiddle leftAddress rightAddress leftAddressMeaning rightAddressMeaning
    subst rightMiddle
    obtain ⟨sameFinish,sameOperand⟩:=module_operand_syntax_unique source entry program meaning tokens (if bytes.val=8 then 64 else 32) (leftMiddle+1) leftFinish rightFinish leftOperand rightOperand leftMeaning rightMeaning
    subst rightAddress
    subst rightOperand
    exact ⟨sameFinish,rfl⟩


private theorem instruction_body_not_at (source : List U8) (scope : Scope) (tokens : List SourceToken)
    (start finish : Nat) (instruction : Instruction)
    (meaning : InstructionBody source scope tokens start finish instruction)
    (atSign : TokenAt source tokens start "@") : False := by
  obtain ⟨token,byte,selected,first,lower,upper⟩:=opcode_at_first source tokens start _
    (instruction_body_opcode source scope tokens start finish instruction meaning)
  obtain ⟨other,otherSelected,spelled⟩:=atSign
  have sameToken:=Option.some.inj (selected.symm.trans otherSelected)
  subst other
  have atFirst:=spelling_first source token "@" 64#u8 spelled (by decide)
  have sameByte:=Option.some.inj (first.symm.trans atFirst)
  subst byte
  norm_num at lower

private theorem module_predicate_not_bang (source entry : List U8)
    (program : strict_memory_frontend_parse.SourceProgram) (meaning : ModuleMeaning source entry program)
    (tokens : List SourceToken) (position : Nat) (token : SourceToken) (index : U16)
    (selected : tokens[position]?=some token) (reference : RegisterReference source program.scope token 1 true index)
    (bang : TokenAt source tokens position "!") : False := by
  have first:=module_register_first source entry program meaning token 1 true index reference
  obtain ⟨other,otherSelected,spelled⟩:=bang
  have sameToken:=Option.some.inj (selected.symm.trans otherSelected)
  subst other
  have bangFirst:=spelling_first source token "!" 33#u8 spelled (by decide)
  have impossible : 37#u8=33#u8:=Option.some.inj (first.symm.trans bangFirst)
  exact (by decide : (37#u8)≠33#u8) impossible

theorem module_instruction_syntax_unique (source entry : List U8)
    (program : strict_memory_frontend_parse.SourceProgram) (meaning : ModuleMeaning source entry program)
    (tokens : List SourceToken) (start leftFinish rightFinish : Nat) (left right : GuardedInstruction)
    (first : InstructionSyntax source program.scope tokens start leftFinish left)
    (second : InstructionSyntax source program.scope tokens start rightFinish right) :
    leftFinish=rightFinish ∧ left=right := by
  cases first with
  | instruction leftGuard leftBody leftGuardSyntax leftBodySyntax leftRestriction leftSemicolon leftEnd =>
    cases second with
    | instruction rightGuard rightBody rightGuardSyntax rightBodySyntax rightRestriction rightSemicolon rightEnd =>
      cases leftGuardSyntax <;> cases rightGuardSyntax
      case unguarded.unguarded =>
        obtain ⟨sameFinish,sameBody⟩:=module_instruction_body_unique source entry program meaning tokens start _ _ leftBody rightBody leftBodySyntax rightBodySyntax
        subst rightBody
        exact ⟨by omega,rfl⟩
      case unguarded.positive =>
        exact False.elim (instruction_body_not_at source program.scope tokens start _ leftBody leftBodySyntax (by assumption))
      case unguarded.negative =>
        exact False.elim (instruction_body_not_at source program.scope tokens start _ leftBody leftBodySyntax (by assumption))
      case positive.unguarded =>
        exact False.elim (instruction_body_not_at source program.scope tokens start _ rightBody rightBodySyntax (by assumption))
      case positive.positive =>
        rename_i leftBodyFinish rightBodyFinish leftToken leftRegister leftAt leftSelected leftReference rightToken rightRegister rightAt rightSelected rightReference
        have sameToken:=Option.some.inj (leftSelected.symm.trans rightSelected)
        subst rightToken
        have sameRegister:=module_register_reference_unique source entry program meaning leftToken 1 true leftRegister rightRegister leftReference rightReference
        obtain ⟨sameFinish,sameBody⟩:=module_instruction_body_unique source entry program meaning tokens (start+2) _ _ leftBody rightBody leftBodySyntax rightBodySyntax
        subst rightBody
        subst rightRegister
        exact ⟨by omega,rfl⟩
      case positive.negative =>
        exact False.elim (module_predicate_not_bang source entry program meaning tokens (start+1) _ _
          (by assumption) (by assumption) (by assumption))
      case negative.unguarded =>
        exact False.elim (instruction_body_not_at source program.scope tokens start _ rightBody rightBodySyntax (by assumption))
      case negative.positive =>
        exact False.elim (module_predicate_not_bang source entry program meaning tokens (start+1) _ _
          (by assumption) (by assumption) (by assumption))
      case negative.negative =>
        rename_i leftBodyFinish rightBodyFinish leftToken leftRegister leftAt leftBang leftSelected leftReference rightToken rightRegister rightAt rightBang rightSelected rightReference
        have sameToken:=Option.some.inj (leftSelected.symm.trans rightSelected)
        subst rightToken
        have sameRegister:=module_register_reference_unique source entry program meaning leftToken 1 true leftRegister rightRegister leftReference rightReference
        obtain ⟨sameFinish,sameBody⟩:=module_instruction_body_unique source entry program meaning tokens (start+3) _ _ leftBody rightBody leftBodySyntax rightBodySyntax
        subst rightBody
        subst rightRegister
        exact ⟨by omega,rfl⟩

end PtxFrontendProgram
