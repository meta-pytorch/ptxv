/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Source
import FrontendProgram.SpecialReferences

open Aeneas Aeneas.Std PtxvMemorySafety
set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192

namespace PtxFrontendProgram

theorem digit_unique (byte : U8) (radix left right : Nat)
    (first : Digit byte radix left) (second : Digit byte radix right) : left=right := by
  rcases first with ⟨_,first⟩
  rcases second with ⟨_,second⟩
  rcases first with first|first|first <;> rcases second with second|second|second <;> omega

theorem digits_ordered (source : List U8) (radix start finish initial result : Nat)
    (digits : DigitSequence source radix start finish initial result) : start≤finish := by
  induction digits <;> omega

theorem digits_unique (source : List U8) (radix start finish initial left right : Nat)
    (first : DigitSequence source radix start finish initial left)
    (second : DigitSequence source radix start finish initial right) : left=right := by
  induction first generalizing right with
  | empty position accumulator =>
    cases second with
    | empty => rfl
    | next _ _ tail => have := digits_ordered _ _ _ _ _ _ tail;omega
  | @next start finish accumulator digit result byte selected valid tail ih =>
    cases second with
    | empty => have := digits_ordered _ _ _ _ _ _ tail;omega
    | @next _ _ _ other _ otherByte otherSelected otherValid otherTail =>
      have same : byte=otherByte := Option.some.inj (selected.symm.trans otherSelected)
      subst otherByte
      have sameDigit := digit_unique byte radix digit other valid otherValid
      subst other
      exact ih right otherTail

theorem unsigned_numeral_unique (source : List U8) (start finish left right : Nat)
    (first : UnsignedNumeral source start finish left)
    (second : UnsignedNumeral source start finish right) : left=right := by
  rcases first with ⟨_,_,first⟩
  rcases second with ⟨_,_,second⟩
  rcases first with ⟨first,decimal⟩|⟨_,zero,_,_,long,first⟩ <;>
    rcases second with ⟨second,otherDecimal⟩|⟨_,otherZero,_,_,otherLong,second⟩
  · exact digits_unique _ _ _ _ _ _ _ first second
  · rcases decimal with short|⟨byte,selected,nonzero⟩
    · omega
    · have same : byte=48#u8:=Option.some.inj (selected.symm.trans otherZero)
      subst byte
      norm_num at nonzero
  · rcases otherDecimal with short|⟨byte,selected,nonzero⟩
    · omega
    · have same : byte=48#u8:=Option.some.inj (selected.symm.trans zero)
      subst byte
      norm_num at nonzero
  · exact digits_unique _ _ _ _ _ _ _ first second

theorem module_global_reference_unique (source entry : List U8)
    (program : strict_memory_frontend_parse.SourceProgram) (meaning : ModuleMeaning source entry program)
    (token : SourceToken) (left right : Usize)
    (first : GlobalReference source program.scope token left)
    (second : GlobalReference source program.scope token right) : left=right := by
  obtain ⟨leftGlobal,leftFound,leftNamed⟩:=first
  obtain ⟨rightGlobal,rightFound,rightNamed⟩:=second
  apply UScalar.eq_of_val_eq
  exact module_global_names_injective source entry program meaning left.val right.val leftGlobal rightGlobal
    leftFound rightFound (leftNamed.2.2.2.2.trans rightNamed.2.2.2.2.symm)

theorem module_label_reference_unique (source entry : List U8)
    (program : strict_memory_frontend_parse.SourceProgram) (meaning : ModuleMeaning source entry program)
    (token : SourceToken) (left right : U32)
    (first : LabelReference source program.scope token left)
    (second : LabelReference source program.scope token right) : left=right := by
  obtain ⟨_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,_,separate,_⟩:=meaning
  obtain ⟨leftLabel,leftMember,leftNamed,rfl⟩:=first
  obtain ⟨rightLabel,rightMember,rightNamed,rfl⟩:=second
  obtain ⟨i,leftFound⟩:=List.mem_iff_getElem?.mp leftMember
  obtain ⟨j,rightFound⟩:=List.mem_iff_getElem?.mp rightMember
  obtain ⟨leftBound,leftEq⟩:=List.getElem?_eq_some_iff.mp leftFound
  obtain ⟨rightBound,rightEq⟩:=List.getElem?_eq_some_iff.mp rightFound
  have equalBytes:=leftNamed.2.2.2.2.trans rightNamed.2.2.2.2.symm
  have same : NameEqual source leftLabel.name rightLabel.name :=
    ⟨leftNamed.1,leftNamed.2.1,rightNamed.1,rightNamed.2.1,equalBytes⟩
  have reverse : NameEqual source rightLabel.name leftLabel.name :=
    ⟨rightNamed.1,rightNamed.2.1,leftNamed.1,leftNamed.2.1,equalBytes.symm⟩
  rcases lt_trichotomy i j with before|equal|after
  · have excluded:=separate.rel_getElem_of_lt leftBound rightBound before
    rw [leftEq,rightEq] at excluded
    exact False.elim (excluded same)
  · subst j
    have sameLabel:=Option.some.inj (leftFound.symm.trans rightFound)
    exact congrArg (fun label : Label=>label.target) sameLabel
  · have excluded:=separate.rel_getElem_of_lt rightBound leftBound after
    rw [rightEq,leftEq] at excluded
    exact False.elim (excluded reverse)

theorem token_text_unique (source : List U8) (token : SourceToken) (left right : List U8)
    (first : TokenText source token.start.val token.end.val left)
    (second : TokenText source token.start.val token.end.val right) : left=right := by
  have lengthEq : left.length=right.length := first.2.2.1.symm.trans second.2.2.1
  apply List.ext_getElem?
  intro index
  by_cases inside : index<left.length
  · exact (first.2.2.2 index inside).symm.trans (second.2.2.2 index (by omega))
  · have otherOutside : ¬index<right.length := by omega
    simp [inside,otherOutside]

theorem table_denotation_unique {α : Type} (table : List (String × α))
    (separate : table.Pairwise (fun left right => utf8 left.1≠utf8 right.1))
    (leftText rightText : String) (left right : α)
    (first : (leftText,left)∈table) (second : (rightText,right)∈table)
    (bytes : utf8 leftText=utf8 rightText) : left=right := by
  obtain ⟨i,leftFound⟩:=List.mem_iff_getElem?.mp first
  obtain ⟨j,rightFound⟩:=List.mem_iff_getElem?.mp second
  obtain ⟨leftBound,leftEq⟩:=List.getElem?_eq_some_iff.mp leftFound
  obtain ⟨rightBound,rightEq⟩:=List.getElem?_eq_some_iff.mp rightFound
  rcases lt_trichotomy i j with before|equal|after
  · have excluded:=separate.rel_getElem_of_lt leftBound rightBound before
    rw [leftEq,rightEq] at excluded
    exact False.elim (excluded bytes)
  · subst j
    have same:=Option.some.inj (leftFound.symm.trans rightFound)
    exact congrArg Prod.snd same
  · have excluded:=separate.rel_getElem_of_lt rightBound leftBound after
    rw [rightEq,leftEq] at excluded
    exact False.elim (excluded bytes.symm)

theorem opcode_names_separate : opcodeNames.Pairwise (fun left right => utf8 left.1≠utf8 right.1) := by decide

theorem opcode_at_unique (source : List U8) (tokens : List SourceToken) (position : Nat)
    (left right : Opcode) (first : OpcodeAt source tokens position left)
    (second : OpcodeAt source tokens position right) : left=right := by
  obtain ⟨leftText,leftListed,leftToken,leftFound,leftTextMeaning⟩:=first
  obtain ⟨rightText,rightListed,rightToken,rightFound,rightTextMeaning⟩:=second
  have sameToken:=Option.some.inj (leftFound.symm.trans rightFound)
  subst rightToken
  exact table_denotation_unique opcodeNames opcode_names_separate leftText rightText left right
    leftListed rightListed (token_text_unique source leftToken _ _ leftTextMeaning rightTextMeaning)

theorem special_names_separate : specialNames.Pairwise (fun left right => utf8 left.1≠utf8 right.1) := by decide

theorem special_meaning_unique (source : List U8) (token : SourceToken)
    (leftText rightText : String) (left right : ptxv_memory_write.value_flow.SpecialRegister)
    (firstKnown : (leftText,left)∈specialNames) (secondKnown : (rightText,right)∈specialNames)
    (first : TokenSpelling source token leftText) (second : TokenSpelling source token rightText) : left=right := by
  exact table_denotation_unique specialNames special_names_separate leftText rightText left right
    firstKnown secondKnown (token_text_unique source token _ _ first second)

theorem decimal_points (source : List U8) (start finish initial result : Nat)
    (digits : DigitSequence source 10 start finish initial result) :
    ∀ position,start≤position → position<finish →
      ∃ byte,source[position]?=some byte ∧ 48≤byte.val ∧ byte.val≤57 := by
  induction digits with
  | empty position accumulator => intro position lower upper;omega
  | @next start finish accumulator digit result byte selected valid tail ih =>
    intro position lower upper
    by_cases same : position=start
    · subst position
      refine ⟨byte,selected,?_⟩
      rcases valid with ⟨bound,alternatives⟩
      rcases alternatives with decimal|hex|hex <;> omega
    · exact ih position (by omega) upper

theorem reference_prefix (source : List U8) (declaration : Declaration) (token : SourceToken) (number : Nat)
    (named : RegisterName source declaration token number) :
    (nameBytes source token).take (nameBytes source declaration.name).length=nameBytes source declaration.name ∧
    (nameBytes source declaration.name).length≤(nameBytes source token).length ∧
    ((nameBytes source declaration.name).length<(nameBytes source token).length →
      declaration.indexed=true ∧ ∃ byte,(nameBytes source token)[(nameBytes source declaration.name).length]?=some byte ∧
        48≤byte.val ∧ byte.val≤57) := by
  cases indexed : declaration.indexed with
  | false =>
    simp only [RegisterName,indexed] at named
    rw [←named.1.2.2.2.2]
    refine ⟨by simp,by omega,?_⟩
    intro strict
    omega
  | true =>
    simp only [RegisterName,indexed,ite_true] at named
    obtain ⟨positive,upper,nonempty,bounded,front,digits,_⟩:=named
    have declaredLength:=name_bytes_length source declaration.name (by omega) upper
    have tokenLength:=name_bytes_length source token (by omega) bounded
    have strict : (nameBytes source declaration.name).length<(nameBytes source token).length := by omega
    obtain ⟨byte,selected,decimal⟩:=decimal_points source _ _ _ _ digits
      (token.start.val+(declaration.name.end.val-declaration.name.start.val)) (by omega) nonempty
    refine ⟨by simpa only [declaredLength] using front,by omega,?_⟩
    intro _
    refine ⟨rfl,byte,?_,decimal⟩
    rw [declaredLength]
    simpa [nameBytes,List.getElem?_take,List.getElem?_drop,Nat.add_comm,
      show declaration.name.end.val-declaration.name.start.val<token.end.val-token.start.val by omega] using selected

theorem prefix_extension {α : Type} (whole left right : List α)
    (first : whole.take left.length=left) (second : whole.take right.length=right)
    (bounded : right.length≤whole.length) (shorter : left.length<right.length) :
    ∃ byte rest,right=left++byte::rest ∧ whole[left.length]?=some byte := by
  have inside : left.length<whole.length := by omega
  let byte:=whole[left.length]
  let rest:=(whole.drop (left.length+1)).take (right.length-left.length-1)
  have split:=List.drop_eq_getElem_cons inside
  refine ⟨byte,rest,?_,List.getElem?_eq_getElem inside⟩
  calc
    right=whole.take right.length:=second.symm
    _=whole.take (left.length+(right.length-left.length)):=by congr 1;omega
    _=whole.take left.length++(whole.drop left.length).take (right.length-left.length):=by rw [List.take_add]
    _=left++byte::rest:=by
      rw [first,split]
      have amount : right.length-left.length=(right.length-left.length-1)+1:=by omega
      rw [amount,List.take_succ_cons]

theorem separate_reference_names (source : List U8) (left right : Declaration) (token : SourceToken)
    (leftNumber rightNumber : Nat)
    (leftValid : Identifier source left.name true) (rightValid : Identifier source right.name true)
    (first : RegisterName source left token leftNumber) (second : RegisterName source right token rightNumber)
    (separate : ¬NameEqual source left.name right.name ∧
      (left.indexed=true → ¬DecimalExtension source left.name right.name) ∧
      (right.indexed=true → ¬DecimalExtension source right.name left.name)) : False := by
  have leftShape:=reference_prefix source left token leftNumber first
  have rightShape:=reference_prefix source right token rightNumber second
  rcases lt_trichotomy (nameBytes source left.name).length (nameBytes source right.name).length with shorter|same|longer
  · obtain ⟨indexed,byte,selected,decimal⟩:=leftShape.2.2 (by omega)
    obtain ⟨other,rest,extend,otherSelected⟩:=prefix_extension (nameBytes source token) (nameBytes source left.name)
      (nameBytes source right.name) leftShape.1 rightShape.1 rightShape.2.1 shorter
    have equal : byte=other:=Option.some.inj (selected.symm.trans otherSelected)
    subst other
    exact separate.2.1 indexed ⟨byte,rest,extend,decimal⟩
  · have bytes : nameBytes source left.name=nameBytes source right.name := by
      rw [←leftShape.1,←rightShape.1,same]
    exact separate.1 ⟨leftValid.2.1,leftValid.2.2.1,rightValid.2.1,rightValid.2.2.1,bytes⟩
  · obtain ⟨indexed,byte,selected,decimal⟩:=rightShape.2.2 (by omega)
    obtain ⟨other,rest,extend,otherSelected⟩:=prefix_extension (nameBytes source token) (nameBytes source right.name)
      (nameBytes source left.name) rightShape.1 leftShape.1 leftShape.2.1 longer
    have equal : byte=other:=Option.some.inj (selected.symm.trans otherSelected)
    subst other
    exact separate.2.2 indexed ⟨byte,rest,extend,decimal⟩

theorem register_reference_unique (source : List U8) (scope : Scope) (token : SourceToken)
    (bits : Nat) (predicate : Bool) (left right : U16)
    (identifiers : ∀ declaration∈scope.declarations.val,Identifier source declaration.name true)
    (separate : SeparateDeclarations source scope.declarations.val)
    (first : RegisterReference source scope token bits predicate left)
    (second : RegisterReference source scope token bits predicate right) : left=right := by
  obtain ⟨leftDeclaration,leftMember,_,_,leftNumber,_,leftNamed,leftIndex⟩:=first
  obtain ⟨rightDeclaration,rightMember,_,_,rightNumber,_,rightNamed,rightIndex⟩:=second
  have sameDeclaration : leftDeclaration=rightDeclaration := by
    obtain ⟨i,leftFound⟩:=List.mem_iff_getElem?.mp leftMember
    obtain ⟨j,rightFound⟩:=List.mem_iff_getElem?.mp rightMember
    obtain ⟨leftBound,leftEq⟩:=List.getElem?_eq_some_iff.mp leftFound
    obtain ⟨rightBound,rightEq⟩:=List.getElem?_eq_some_iff.mp rightFound
    rcases lt_trichotomy i j with before|equal|after
    · have excluded:=separate.rel_getElem_of_lt leftBound rightBound before
      rw [leftEq,rightEq] at excluded
      exact False.elim (separate_reference_names source leftDeclaration rightDeclaration token leftNumber rightNumber
        (identifiers leftDeclaration leftMember) (identifiers rightDeclaration rightMember) leftNamed rightNamed excluded)
    · subst j
      exact Option.some.inj (leftFound.symm.trans rightFound)
    · have excluded:=separate.rel_getElem_of_lt rightBound leftBound after
      rw [rightEq,leftEq] at excluded
      exact False.elim (separate_reference_names source rightDeclaration leftDeclaration token rightNumber leftNumber
        (identifiers rightDeclaration rightMember) (identifiers leftDeclaration leftMember) rightNamed leftNamed excluded)
  subst rightDeclaration
  have numberEq : leftNumber=rightNumber := by
    cases indexed : leftDeclaration.indexed with
    | false =>
      simp only [RegisterName,indexed] at leftNamed rightNamed
      exact leftNamed.2.trans rightNamed.2.symm
    | true =>
      simp only [RegisterName,indexed,ite_true] at leftNamed rightNamed
      obtain ⟨_,_,_,_,_,leftDigits,_⟩:=leftNamed
      obtain ⟨_,_,_,_,_,rightDigits,_⟩:=rightNamed
      exact digits_unique _ _ _ _ _ _ _ leftDigits rightDigits
  apply UScalar.eq_of_val_eq
  omega

theorem module_register_reference_unique (source entry : List U8)
    (program : strict_memory_frontend_parse.SourceProgram) (meaning : ModuleMeaning source entry program)
    (token : SourceToken) (bits : Nat) (predicate : Bool) (left right : U16)
    (first : RegisterReference source program.scope token bits predicate left)
    (second : RegisterReference source program.scope token bits predicate right) : left=right := by
  obtain ⟨tokens,globalsEnd,declarationsStart,instructionsStart,finish,
    _,_,_,_,declarations,_,_,_,_,separate,_,_,_⟩:=meaning
  have identifiers:=declarations_syntax_properties source tokens declarationsStart instructionsStart
    program.scope.declarations.val declarations
  exact register_reference_unique source program.scope token bits predicate left right
    (fun declaration member=>(identifiers.2 declaration member).1) separate first second


theorem numeral_first (source : List U8) (start finish value : Nat)
    (meaning : UnsignedNumeral source start finish value) :
    ∃ byte, source[start]?=some byte ∧ 48≤byte.val ∧ byte.val≤57 := by
  obtain ⟨ordered,_,decimal|hex⟩:=meaning
  · exact decimal_points source start finish 0 value decimal.1 start (by omega) ordered
  · obtain ⟨second,zero,_⟩:=hex
    exact ⟨48#u8,zero,by decide,by decide⟩

theorem name_equal_first (source : List U8) (left right : SourceToken)
    (meaning : NameEqual source left right) : source[left.start.val]?=source[right.start.val]? := by
  have points:=name_equal_points source left right meaning
  simpa using points.2 0 (by have := meaning.1;omega)

theorem register_name_first (source : List U8) (declaration : Declaration) (token : SourceToken)
    (number : Nat) (valid : Identifier source declaration.name true)
    (meaning : RegisterName source declaration token number) : source[token.start.val]?=some 37#u8 := by
  have firstByte := valid.2.2.2.2.1 rfl
  cases indexed : declaration.indexed with
  | false =>
    simp only [RegisterName,indexed] at meaning
    exact (name_equal_first source declaration.name token meaning.1).symm.trans firstByte
  | true =>
    simp only [RegisterName,indexed,ite_true] at meaning
    obtain ⟨nonempty,bounded,tokenLong,tokenBound,front,_,_⟩:=meaning
    have point:=congrArg (fun values : List U8 => values[0]?) front
    simpa [nameBytes,List.getElem?_take,List.getElem?_drop,show 0<declaration.name.end.val-declaration.name.start.val by omega,
      show 0<token.end.val-token.start.val by omega,firstByte] using point

theorem module_register_first (source entry : List U8)
    (program : strict_memory_frontend_parse.SourceProgram) (meaning : ModuleMeaning source entry program)
    (token : SourceToken) (bits : Nat) (predicate : Bool) (index : U16)
    (reference : RegisterReference source program.scope token bits predicate index) :
    source[token.start.val]?=some 37#u8 := by
  obtain ⟨tokens,globalsEnd,declarationsStart,instructionsStart,finish,
    _,_,_,_,declarations,_,_,_,_,_,_,_,_⟩:=meaning
  have identifiers:=declarations_syntax_properties source tokens declarationsStart instructionsStart
    program.scope.declarations.val declarations
  obtain ⟨declaration,member,_,_,number,_,named,_⟩:=reference
  exact register_name_first source declaration token number (identifiers.2 declaration member).1 named

theorem module_global_first (source entry : List U8)
    (program : strict_memory_frontend_parse.SourceProgram) (meaning : ModuleMeaning source entry program)
    (token : SourceToken) (index : Usize) (reference : GlobalReference source program.scope token index) :
    ∃ byte, source[token.start.val]?=some byte ∧ InitialCharacter byte := by
  obtain ⟨tokens,globalsEnd,_,_,_,_,_,globals,_,_,_,_,_,_,_,_,_,_⟩:=meaning
  have identifiers:=globals_syntax_identifiers source tokens 6 globalsEnd program.scope.globals.val globals
  obtain ⟨global,selected,named⟩:=reference
  have valid:=identifiers global (List.mem_of_getElem? selected)
  obtain ⟨byte,found,initial⟩:=valid.2.2.2.2.2.1
  refine ⟨byte,?_,initial⟩
  have startEq:=name_equal_first source global.name token named
  simpa using startEq.symm.trans found

theorem special_first (source : List U8) (token : SourceToken) (text : String)
    (kind : ptxv_memory_write.value_flow.SpecialRegister) (known : (text,kind)∈specialNames)
    (spelled : TokenSpelling source token text) : source[token.start.val]?=some 37#u8 := by
  have starts : (utf8 text)[0]?=some 37#u8 := by
    have named : text∈specialNames.map Prod.fst := List.mem_map.mpr ⟨(text,kind),known,rfl⟩
    simp only [specialNames,List.map_cons,List.map_nil,List.mem_cons,List.not_mem_nil,or_false] at named
    rcases named with h|h|h|h|h|h|h|h|h|h|h|h|h <;> subst text <;> decide
  have bound : 0<(utf8 text).length := (List.getElem?_eq_some_iff.mp starts).1
  have point:=spelled.2.2.2 0 bound
  simpa using point.trans starts


theorem numeral_not_percent (source : List U8) (token : SourceToken) (value : Nat)
    (number : UnsignedNumeral source token.start.val token.end.val value)
    (percent : source[token.start.val]?=some 37#u8) : False := by
  obtain ⟨byte,selected,lower,upper⟩:=numeral_first source token.start.val token.end.val value number
  have same : byte=37#u8:=Option.some.inj (selected.symm.trans percent)
  subst byte
  norm_num at lower

theorem numeral_not_identifier (source : List U8) (token : SourceToken) (value : Nat)
    (number : UnsignedNumeral source token.start.val token.end.val value)
    (initial : ∃ byte,source[token.start.val]?=some byte ∧ InitialCharacter byte) : False := by
  obtain ⟨byte,selected,lower,upper⟩:=numeral_first source token.start.val token.end.val value number
  obtain ⟨other,otherSelected,valid⟩:=initial
  have same : byte=other:=Option.some.inj (selected.symm.trans otherSelected)
  subst other
  rcases valid with valid|valid|valid|valid <;> omega

theorem percent_not_identifier (source : List U8) (token : SourceToken)
    (percent : source[token.start.val]?=some 37#u8)
    (initial : ∃ byte,source[token.start.val]?=some byte ∧ InitialCharacter byte) : False := by
  obtain ⟨byte,selected,valid⟩:=initial
  have same : byte=37#u8:=Option.some.inj (selected.symm.trans percent)
  subst byte
  rcases valid with valid|valid|valid|valid <;> norm_num at valid

theorem module_operand_meaning_unique (source entry : List U8)
    (program : strict_memory_frontend_parse.SourceProgram) (meaning : ModuleMeaning source entry program)
    (token : SourceToken) (bits : Nat) (left right : Operand)
    (first : OperandMeaning source program.scope token bits left)
    (second : OperandMeaning source program.scope token bits right) : left=right := by
  cases first with
  | immediate value number word width fits =>
    cases second with
    | immediate other otherNumber =>
      have same : value=other := UScalar.eq_of_val_eq
        (unsigned_numeral_unique source token.start.val token.end.val value.val other.val number otherNumber)
      subst other
      rfl
    | register index reference =>
      exact False.elim (numeral_not_percent source token value.val number
        (module_register_first source entry program meaning token bits false index reference))
    | global index _ reference =>
      exact False.elim (numeral_not_identifier source token value.val number
        (module_global_first source entry program meaning token index reference))
    | special kind text _ known spelled =>
      exact False.elim (numeral_not_percent source token value.val number
        (special_first source token text kind known spelled))
  | register index reference =>
    cases second with
    | immediate value number =>
      exact False.elim (numeral_not_percent source token value.val number
        (module_register_first source entry program meaning token bits false index reference))
    | register other otherReference =>
      have same:=module_register_reference_unique source entry program meaning token bits false index other reference otherReference
      subst other
      rfl
    | global other _ otherReference =>
      exact False.elim (percent_not_identifier source token
        (module_register_first source entry program meaning token bits false index reference)
        (module_global_first source entry program meaning token other otherReference))
    | special kind text _ known spelled =>
      exact False.elim (module_special_not_register source entry program meaning token text kind known spelled bits false index reference)
  | global index width reference =>
    cases second with
    | immediate value number =>
      exact False.elim (numeral_not_identifier source token value.val number
        (module_global_first source entry program meaning token index reference))
    | register other otherReference =>
      exact False.elim (percent_not_identifier source token
        (module_register_first source entry program meaning token bits false other otherReference)
        (module_global_first source entry program meaning token index reference))
    | global other _ otherReference =>
      have same:=module_global_reference_unique source entry program meaning token index other reference otherReference
      subst other
      rfl
    | special _ _ otherWidth => omega
  | special kind text width known spelled =>
    cases second with
    | immediate value number =>
      exact False.elim (numeral_not_percent source token value.val number
        (special_first source token text kind known spelled))
    | register index reference =>
      exact False.elim (module_special_not_register source entry program meaning token text kind known spelled bits false index reference)
    | global _ otherWidth => omega
    | special other otherText _ otherKnown otherSpelled =>
      have same:=special_meaning_unique source token text otherText kind other known otherKnown spelled otherSpelled
      subst other
      rfl

end PtxFrontendProgram
