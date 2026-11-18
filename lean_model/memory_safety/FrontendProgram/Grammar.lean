/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.RegisterTypes

open Aeneas Aeneas.Std PtxvMemorySafety

set_option autoImplicit false
set_option debug.skipKernelTC false

namespace PtxFrontendProgram

abbrev SourceToken := strict_memory_frontend.Token
abbrev Scope := strict_memory_frontend_parse.Scope
abbrev Declaration := strict_memory_frontend_parse.Declaration
abbrev Global := strict_memory_frontend_parse.Global
abbrev Label := strict_memory_frontend_parse.Label
abbrev Operand := ptxv_memory_write.value_flow.Operand
abbrev Instruction := ptxv_memory_write.value_flow.Instruction
abbrev GuardedInstruction := ptxv_memory_write.value_flow.GuardedInstruction
abbrev Opcode := strict_memory_frontend_parse.Opcode

def utf8 (text : String) : List U8 :=
  text.toUTF8.data.toList.map fun byte => U8.ofNat byte.toNat (by
    have bound := byte.toNat_lt_size
    change byte.toNat < 256 at bound
    change byte.toNat ≤ 255
    omega)

def TokenSpelling (source : List U8) (token : SourceToken) (text : String) : Prop :=
  TokenText source token.start.val token.end.val (utf8 text)

def TokenAt (source : List U8) (tokens : List SourceToken) (position : Nat) (text : String) : Prop :=
  ∃ token, tokens[position]? = some token ∧ TokenSpelling source token text

def NumberAt (source : List U8) (tokens : List SourceToken) (position value : Nat) : Prop :=
  ∃ token, tokens[position]? = some token ∧ token.word = true ∧
    UnsignedNumeral source token.start.val token.end.val value

def nameBytes (source : List U8) (token : SourceToken) : List U8 :=
  (source.drop token.start.val).take (token.end.val - token.start.val)

def NameEqual (source : List U8) (left right : SourceToken) : Prop :=
  left.start.val < left.end.val ∧ left.end.val ≤ source.length ∧
  right.start.val < right.end.val ∧ right.end.val ≤ source.length ∧
  nameBytes source left = nameBytes source right

def InitialCharacter (byte : U8) : Prop :=
  (97 ≤ byte.val ∧ byte.val ≤ 122) ∨ (65 ≤ byte.val ∧ byte.val ≤ 90) ∨ byte.val = 95 ∨ byte.val = 36

def FollowingCharacter (byte : U8) : Prop :=
  InitialCharacter byte ∨ (48 ≤ byte.val ∧ byte.val ≤ 57)

def Identifier (source : List U8) (token : SourceToken) (register : Bool) : Prop :=
  let first := token.start.val + (if register then 1 else 0)
  token.word = true ∧ token.start.val < token.end.val ∧ token.end.val ≤ source.length ∧
    first < token.end.val ∧
    (register = true → source[token.start.val]? = some 37#u8) ∧
    (∃ byte, source[first]? = some byte ∧ InitialCharacter byte) ∧
    ∀ slot, first + 1 ≤ slot → slot < token.end.val →
      ∃ byte, source[slot]? = some byte ∧ FollowingCharacter byte

def RegisterName (source : List U8) (declaration : Declaration) («name» : SourceToken) (number : Nat) : Prop :=
  if declaration.indexed then
    let width := declaration.«name».end.val - declaration.«name».start.val
    declaration.«name».start.val < declaration.«name».end.val ∧ declaration.«name».end.val ≤ source.length ∧
    «name».start.val + width < «name».end.val ∧ «name».end.val ≤ source.length ∧
    (nameBytes source «name»).take width = nameBytes source declaration.«name» ∧
    DigitSequence source 10 («name».start.val + width) «name».end.val 0 number ∧
    («name».end.val = «name».start.val + width + 1 ∨
      ∃ byte, source[«name».start.val + width]? = some byte ∧ byte.val ≠ 48)
  else NameEqual source declaration.«name» «name» ∧ number = 0

def RegisterReference (source : List U8) (scope : Scope) («name» : SourceToken)
    (bits : Nat) (predicate : Bool) (index : U16) : Prop :=
  ∃ declaration ∈ scope.declarations.val, declaration.bits.val = bits ∧ declaration.predicate = predicate ∧
    ∃ number, number < declaration.count.val ∧ RegisterName source declaration «name» number ∧
      index.val = declaration.first.val + number

def GlobalReference (source : List U8) (scope : Scope) («name» : SourceToken) (index : Usize) : Prop :=
  ∃ global, scope.globals.val[index.val]? = some global ∧ NameEqual source global.«name» «name»

def LabelReference (source : List U8) (scope : Scope) («name» : SourceToken) (target : U32) : Prop :=
  ∃ label ∈ scope.labels.val, NameEqual source label.«name» «name» ∧ label.target = target

def specialNames : List (String × ptxv_memory_write.value_flow.SpecialRegister) :=
  [("%tid.x", .Thread 0#u8), ("%tid.y", .Thread 1#u8), ("%tid.z", .Thread 2#u8),
   ("%ctaid.x", .Block 0#u8), ("%ctaid.y", .Block 1#u8), ("%ctaid.z", .Block 2#u8),
   ("%ntid.x", .BlockSize 0#u8), ("%ntid.y", .BlockSize 1#u8), ("%ntid.z", .BlockSize 2#u8),
   ("%nctaid.x", .GridSize 0#u8), ("%nctaid.y", .GridSize 1#u8), ("%nctaid.z", .GridSize 2#u8),
   ("%laneid", .Lane)]

inductive OperandMeaning (source : List U8) (scope : Scope) (token : SourceToken) (bits : Nat) : Operand → Prop
  | immediate (value : U64) : UnsignedNumeral source token.start.val token.end.val value.val →
      token.word = true → (bits = 32 ∨ bits = 64) → value.val < 2 ^ bits → OperandMeaning source scope token bits (.Immediate value)
  | register (index : U16) : RegisterReference source scope token bits false index → OperandMeaning source scope token bits (.Virtual index)
  | global (index : Usize) : bits = 64 → GlobalReference source scope token index →
      OperandMeaning source scope token bits (.Pointer ⟨index, 0#u64⟩)
  | special (register : ptxv_memory_write.value_flow.SpecialRegister) (text : String) :
      bits = 32 → (text, register) ∈ specialNames → TokenSpelling source token text →
      OperandMeaning source scope token bits (.Special register)

inductive OperandSyntax (source : List U8) (scope : Scope) (tokens : List SourceToken) (bits : Nat) : Nat → Nat → Operand → Prop
  | single {start token operand} : tokens[start]? = some token → OperandMeaning source scope token bits operand →
      OperandSyntax source scope tokens bits start (start + 1) operand
  | negative {start value : Nat} (encoded : U64) : TokenAt source tokens start "-" → NumberAt source tokens (start + 1) value →
      (bits = 32 ∨ bits = 64) → value ≤ 2 ^ (bits - 1) → encoded.val = (2 ^ 64 - value) % 2 ^ 64 →
      OperandSyntax source scope tokens bits start (start + 2) (.Immediate encoded)

inductive OffsetSyntax (source : List U8) (tokens : List SourceToken) : Nat → Nat → Int → Prop
  | absent (position) : OffsetSyntax source tokens position position 0
  | positive {start value} : TokenAt source tokens start "+" → NumberAt source tokens (start + 1) value → value < 2 ^ 63 →
      OffsetSyntax source tokens start (start + 2) (Int.ofNat value)

inductive AddressTail (source : List U8) (scope : Scope) (tokens : List SourceToken) : Nat → Nat → Operand → Int → Prop
  | offset {start finish value} : OffsetSyntax source tokens start finish value →
      AddressTail source scope tokens start finish (.Immediate 0#u64) value

inductive AddressSyntax (source : List U8) (scope : Scope) (tokens : List SourceToken) : Nat → Nat → ptxv_memory_write.value_flow.Address → Prop
  | address {start middle token base index offset finish} (encoded : I64) : TokenAt source tokens start "[" →
      tokens[start + 1]? = some token → OperandMeaning source scope token 64 base →
      AddressTail source scope tokens (start + 2) middle index offset → encoded.val = offset →
      TokenAt source tokens middle "]" → finish = middle + 1 →
      AddressSyntax source scope tokens start finish ⟨base, index, encoded⟩

inductive GuardSyntax (source : List U8) (scope : Scope) (tokens : List SourceToken) : Nat → Nat → ptxv_memory_write.value_flow.Guard → Prop
  | unguarded (position) : GuardSyntax source scope tokens position position ⟨none, false⟩
  | positive {start token} (register : U16) : TokenAt source tokens start "@" → tokens[start + 1]? = some token →
      RegisterReference source scope token 1 true register → GuardSyntax source scope tokens start (start + 2) ⟨some register, false⟩
  | negative {start token} (register : U16) : TokenAt source tokens start "@" → TokenAt source tokens (start + 1) "!" →
      tokens[start + 2]? = some token → RegisterReference source scope token 1 true register →
      GuardSyntax source scope tokens start (start + 3) ⟨some register, true⟩

def binaryNames : List (String × U8 × ptxv_memory_write.value_flow.BinaryOperation) :=
  [("add.u32",32#u8,.Add),("add.u64",64#u8,.Add),("sub.u32",32#u8,.Subtract),("sub.u64",64#u8,.Subtract),
   ("and.b32",32#u8,.And),("and.b64",64#u8,.And),("or.b32",32#u8,.Or),("or.b64",64#u8,.Or),
   ("xor.b32",32#u8,.Xor),("xor.b64",64#u8,.Xor),("mul.lo.u32",32#u8,.MultiplyLow),("mul.lo.u64",64#u8,.MultiplyLow)]

def comparisonNames : List (String × U8 × ptxv_memory_write.value_flow.Comparison) :=
  [("setp.eq.u32",32#u8,.Equal),("setp.eq.u64",64#u8,.Equal),("setp.ne.u32",32#u8,.NotEqual),("setp.ne.u64",64#u8,.NotEqual),
   ("setp.lt.u32",32#u8,.LessUnsigned),("setp.lt.u64",64#u8,.LessUnsigned),("setp.le.u32",32#u8,.LessEqualUnsigned),("setp.le.u64",64#u8,.LessEqualUnsigned),
   ("setp.gt.u32",32#u8,.GreaterUnsigned),("setp.gt.u64",64#u8,.GreaterUnsigned),("setp.ge.u32",32#u8,.GreaterEqualUnsigned),("setp.ge.u64",64#u8,.GreaterEqualUnsigned)]

def opcodeNames : List (String × Opcode) :=
  [("ret",.Exit),("exit",.Exit),("bra",.Branch),
   ("mov.u32",.Move 32#u8),("mov.b32",.Move 32#u8),("mov.u64",.Move 64#u8),("mov.b64",.Move 64#u8),
   ("cvta.global.u64",.Move 64#u8),("mul.wide.u32",.MultiplyWide),("cvt.u64.u32",.Convert 32#u8 64#u8),("cvt.u32.u64",.Convert 64#u8 32#u8),
   ("ld.global.u8",.Load 1#u8 false),("ld.global.u16",.Load 2#u8 false),("ld.global.u32",.Load 4#u8 false),
   ("ld.global.b32",.Load 4#u8 false),("ld.global.u64",.Load 8#u8 false),("ld.global.b64",.Load 8#u8 false),
   ("ld.global.s8",.Load 1#u8 true),("ld.global.s16",.Load 2#u8 true),("ld.global.s32",.Load 4#u8 true),
   ("st.global.u8",.Store 1#u8),("st.global.b8",.Store 1#u8),("st.global.u16",.Store 2#u8),("st.global.b16",.Store 2#u8),
   ("st.global.u32",.Store 4#u8),("st.global.b32",.Store 4#u8),("st.global.u64",.Store 8#u8),("st.global.b64",.Store 8#u8)] ++
   binaryNames.map (fun (text,bits,operation) => (text,.Binary bits operation)) ++
   comparisonNames.map (fun (text,bits,comparison) => (text,.Compare bits comparison))

def OpcodeAt (source : List U8) (tokens : List SourceToken) (position : Nat) (opcode : Opcode) : Prop :=
  ∃ text, (text,opcode) ∈ opcodeNames ∧ TokenAt source tokens position text

def RegisterAt (source : List U8) (scope : Scope) (tokens : List SourceToken) (position bits : Nat)
    (predicate : Bool) (register : U16) : Prop :=
  ∃ token, tokens[position]? = some token ∧ RegisterReference source scope token bits predicate register

inductive InstructionBody (source : List U8) (scope : Scope) (tokens : List SourceToken) : Nat → Nat → Instruction → Prop
  | exit (position) : OpcodeAt source tokens position .Exit → InstructionBody source scope tokens position (position+1) .Exit
  | nop (position) : OpcodeAt source tokens position .Nop → InstructionBody source scope tokens position (position+1) .Nop
  | branch {start token} (target : U32) : OpcodeAt source tokens start .Branch → tokens[start+1]? = some token →
      LabelReference source scope token target → InstructionBody source scope tokens start (start+2) (.Branch target)
  | move {start finish} (bits : U8) (destination : U16) (operand : Operand) : OpcodeAt source tokens start (.Move bits) →
      RegisterAt source scope tokens (start+1) bits.val false destination → TokenAt source tokens (start+2) "," →
      OperandSyntax source scope tokens bits.val (start+3) finish operand →
      InstructionBody source scope tokens start finish (.Move bits destination operand)
  | binary {start middle finish} (bits : U8) (destination : U16) (operation : ptxv_memory_write.value_flow.BinaryOperation) (left right : Operand) :
      OpcodeAt source tokens start (.Binary bits operation) → RegisterAt source scope tokens (start+1) bits.val false destination →
      TokenAt source tokens (start+2) "," → OperandSyntax source scope tokens bits.val (start+3) middle left →
      TokenAt source tokens middle "," → OperandSyntax source scope tokens bits.val (middle+1) finish right →
      InstructionBody source scope tokens start finish (.Binary bits destination operation left right)
  | wide {start middle finish} (destination : U16) (left right : Operand) : OpcodeAt source tokens start .MultiplyWide →
      RegisterAt source scope tokens (start+1) 64 false destination → TokenAt source tokens (start+2) "," →
      OperandSyntax source scope tokens 32 (start+3) middle left → TokenAt source tokens middle "," →
      OperandSyntax source scope tokens 32 (middle+1) finish right → InstructionBody source scope tokens start finish (.MultiplyWide destination left right)
  | convert {start finish} (fromBits toBits : U8) (destination : U16) (operand : Operand) :
      OpcodeAt source tokens start (.Convert fromBits toBits) → RegisterAt source scope tokens (start+1) toBits.val false destination →
      TokenAt source tokens (start+2) "," → OperandSyntax source scope tokens fromBits.val (start+3) finish operand →
      InstructionBody source scope tokens start finish (.ConvertUnsigned destination operand fromBits toBits)
  | compare {start middle finish} (bits : U8) (destination : U16) (comparison : ptxv_memory_write.value_flow.Comparison) (left right : Operand) :
      OpcodeAt source tokens start (.Compare bits comparison) → RegisterAt source scope tokens (start+1) 1 true destination →
      TokenAt source tokens (start+2) "," → OperandSyntax source scope tokens bits.val (start+3) middle left →
      TokenAt source tokens middle "," → OperandSyntax source scope tokens bits.val (middle+1) finish right →
      InstructionBody source scope tokens start finish (.Compare bits destination comparison left right)
  | load {start finish} (bytes destinationBits : U8) (destination : U16) (signed : Bool) (address : ptxv_memory_write.value_flow.Address) :
      OpcodeAt source tokens start (.Load bytes signed) → (destinationBits.val=32 ∨ destinationBits.val=64) →
      (bytes.val=8 → destinationBits.val=64) → RegisterAt source scope tokens (start+1) destinationBits.val false destination →
      TokenAt source tokens (start+2) "," → AddressSyntax source scope tokens (start+3) finish address →
      InstructionBody source scope tokens start finish (.Load destination destinationBits bytes signed address)
  | store {start middle finish} (bytes : U8) (address : ptxv_memory_write.value_flow.Address) (operand : Operand) :
      OpcodeAt source tokens start (.Store bytes) → AddressSyntax source scope tokens (start+1) middle address →
      TokenAt source tokens middle "," → OperandSyntax source scope tokens (if bytes.val=8 then 64 else 32) (middle+1) finish operand →
      InstructionBody source scope tokens start finish (.Store bytes address operand)

def ScalarOperand : Operand → Prop
  | .Immediate _ | .Virtual _ => True
  | .Pointer _ | .Special _ => False

def AddressBase : Operand → Prop
  | .Virtual _ | .Pointer _ => True
  | .Immediate _ | .Special _ => False

instance (operand : Operand) : Decidable (ScalarOperand operand) := by
  cases operand <;> simp [ScalarOperand] <;> infer_instance

instance (operand : Operand) : Decidable (AddressBase operand) := by
  cases operand <;> simp [AddressBase] <;> infer_instance

def OperandRestrictions : Instruction → Prop
  | .Move bits _ source => match source with
      | .Pointer _ => bits.val = 64
      | .Special _ => bits.val = 32
      | _ => True
  | .Binary _ _ _ left right | .MultiplyWide _ left right | .Compare _ _ _ left right =>
      ScalarOperand left ∧ ScalarOperand right
  | .ConvertUnsigned _ source _ _ => ScalarOperand source
  | .Load _ _ _ _ address => AddressBase address.base
  | .Store _ address source => AddressBase address.base ∧ ScalarOperand source
  | .Branch _ | .Exit => True
  | .Nop => False

inductive InstructionSyntax (source : List U8) (scope : Scope) (tokens : List SourceToken) : Nat → Nat → GuardedInstruction → Prop
  | instruction {start guarded finish last} (guard : ptxv_memory_write.value_flow.Guard) (body : Instruction) :
      GuardSyntax source scope tokens start guarded guard → InstructionBody source scope tokens guarded finish body →
      OperandRestrictions body → TokenAt source tokens finish ";" → last=finish+1 → InstructionSyntax source scope tokens start last ⟨guard,body⟩

inductive CodeSyntax (source : List U8) (scope : Scope) (tokens : List SourceToken) : Nat → Nat → List GuardedInstruction → List Label → Nat → Prop
  | empty {position pc} : TokenAt source tokens position "}" → CodeSyntax source scope tokens position pc [] [] position
  | label {start pc code labels finish «name»} (target : U32) : tokens[start]?=some «name» → Identifier source «name» false →
      TokenAt source tokens (start+1) ":" → target.val=pc → CodeSyntax source scope tokens (start+2) pc code labels finish →
      CodeSyntax source scope tokens start pc code (⟨«name»,target⟩::labels) finish
  | instruction {start pc next code labels finish instruction} : InstructionSyntax source scope tokens start next instruction →
      CodeSyntax source scope tokens next (pc+1) code labels finish → CodeSyntax source scope tokens start pc (instruction::code) labels finish

def GlobalElementType (source : List U8) (token : SourceToken) (bytes : Nat) : Prop :=
  ∃ text, (text, bytes) ∈ [(".u8",1),(".b8",1),(".u16",2),(".b16",2),
    (".u32",4),(".b32",4),(".u64",8),(".b64",8)] ∧ TokenSpelling source token text

def littleEndian (width value : Nat) : List U8 :=
  (List.range width).map fun byte => U8.ofNat (value / 256 ^ byte % 256) (by change value / 256 ^ byte % 256 ≤ 255; omega)

inductive InitializerSyntax (source : List U8) (tokens : List SourceToken) (width : Nat) : Nat → List Nat → Nat → Prop
  | single {start value} : NumberAt source tokens start value → value < 256 ^ width →
      InitializerSyntax source tokens width start [value] (start+1)
  | cons {start value values finish} : NumberAt source tokens start value → value < 256 ^ width →
      TokenAt source tokens (start+1) "," → InitializerSyntax source tokens width (start+2) values finish →
      InitializerSyntax source tokens width start (value::values) finish

inductive GlobalSyntax (source : List U8) (tokens : List SourceToken) : Nat → Nat → Global → Prop
  | global {start middle finish width alignment count values «name» typeToken} (global : Global) :
      TokenAt source tokens start ".global" → TokenAt source tokens (start+1) ".align" →
      NumberAt source tokens (start+2) alignment → (∃ power, alignment = 2 ^ power) →
      width ≤ alignment → tokens[start+3]?=some typeToken → GlobalElementType source typeToken width →
      tokens[start+4]?=some «name» → Identifier source «name» false → TokenAt source tokens (start+5) "[" →
      NumberAt source tokens (start+6) count → count>0 → TokenAt source tokens (start+7) "]" →
      TokenAt source tokens (start+8) "=" → TokenAt source tokens (start+9) "{" →
      InitializerSyntax source tokens width (start+10) values middle → count=values.length →
      TokenAt source tokens middle "}" → TokenAt source tokens (middle+1) ";" → finish=middle+2 →
      global.«name»=«name» → global.alignment.val=alignment → global.bytes.val=values.flatMap (littleEndian width) →
      GlobalSyntax source tokens start finish global

inductive GlobalsSyntax (source : List U8) (tokens : List SourceToken) : Nat → List Global → Nat → Prop
  | empty (position) : GlobalsSyntax source tokens position [] position
  | cons {start middle finish global globals} : GlobalSyntax source tokens start middle global →
      GlobalsSyntax source tokens middle globals finish → GlobalsSyntax source tokens start (global::globals) finish

inductive DeclarationSyntax (source : List U8) (tokens : List SourceToken) : Nat → Nat → Declaration → Prop
  | single {start typeToken «name»} (declaration : Declaration) :
      TokenAt source tokens start ".reg" → tokens[start+1]?=some typeToken →
      RegisterTypeMeaning source typeToken declaration.bits.val declaration.predicate →
      tokens[start+2]?=some «name» → Identifier source «name» true → declaration.«name»=«name» →
      declaration.count.val=1 → declaration.indexed=false → TokenAt source tokens (start+3) ";" →
      DeclarationSyntax source tokens start (start+4) declaration
  | indexed {start typeToken «name» count} (declaration : Declaration) :
      TokenAt source tokens start ".reg" → tokens[start+1]?=some typeToken →
      RegisterTypeMeaning source typeToken declaration.bits.val declaration.predicate →
      tokens[start+2]?=some «name» → Identifier source «name» true → declaration.«name»=«name» →
      TokenAt source tokens (start+3) "<" → NumberAt source tokens (start+4) count →
      0<count → declaration.count.val=count → declaration.indexed=true → TokenAt source tokens (start+5) ">" →
      TokenAt source tokens (start+6) ";" → DeclarationSyntax source tokens start (start+7) declaration

inductive DeclarationsSyntax (source : List U8) (tokens : List SourceToken) : Nat → List Declaration → Nat → Prop
  | empty (position) : DeclarationsSyntax source tokens position [] position
  | cons {start middle finish declaration declarations} : DeclarationSyntax source tokens start middle declaration →
      DeclarationsSyntax source tokens middle declarations finish →
      DeclarationsSyntax source tokens start (declaration::declarations) finish

def bankSize (declarations : List Declaration) (predicate : Bool) : Nat :=
  ((declarations.filter fun declaration => declaration.predicate == predicate).map fun declaration => declaration.count.val).sum

def DeclarationLayout (scope : Scope) : Prop :=
  scope.virtual_count.val=bankSize scope.declarations.val false ∧ scope.predicate_count.val=bankSize scope.declarations.val true ∧
  ∀ position declaration, scope.declarations.val[position]?=some declaration →
    declaration.first.val=bankSize (scope.declarations.val.take position) declaration.predicate

def DecimalExtension (source : List U8) (shorter longer : SourceToken) : Prop :=
  ∃ byte rest, nameBytes source longer=nameBytes source shorter ++ byte::rest ∧ 48≤byte.val ∧ byte.val≤57

def SeparateDeclarations (source : List U8) (declarations : List Declaration) : Prop :=
  declarations.Pairwise fun left right =>
    ¬NameEqual source left.«name» right.«name» ∧
    (left.indexed=true → ¬DecimalExtension source left.«name» right.«name») ∧
    (right.indexed=true → ¬DecimalExtension source right.«name» left.«name»)

def headerPairs : List (String × String) :=
  [("8.0","sm_80"), ("8.0","sm_90"), ("8.0","sm_90a"), ("8.7","sm_80"), ("8.7","sm_90"), ("8.7","sm_90a"), ("8.7","sm_100"), ("8.7","sm_100a"), ("9.0","sm_80"), ("9.0","sm_90"), ("9.0","sm_90a"), ("9.0","sm_100"), ("9.0","sm_100a"), ("9.0","sm_103a")]

inductive HeaderSyntax (source : List U8) (tokens : List SourceToken) : Prop
  | header : TokenAt source tokens 0 ".version" → (∃ version target, (version,target)∈headerPairs ∧
        TokenAt source tokens 1 version ∧ TokenAt source tokens 3 target) →
      TokenAt source tokens 2 ".target" →
      TokenAt source tokens 4 ".address_size" → NumberAt source tokens 5 64 → HeaderSyntax source tokens

inductive EntrySyntax (source entry : List U8) (tokens : List SourceToken) : Nat → Nat → SourceToken → Prop
  | entry {start «name»} : TokenAt source tokens start ".entry" → tokens[start+1]?=some «name» → Identifier source «name» false →
      TokenText source «name».start.val «name».end.val entry → TokenAt source tokens (start+2) "(" →
      TokenAt source tokens (start+3) ")" → TokenAt source tokens (start+4) "{" →
      EntrySyntax source entry tokens start (start+5) «name»
  | visible {start finish «name»} : TokenAt source tokens start ".visible" → EntrySyntax source entry tokens (start+1) finish «name» →
      EntrySyntax source entry tokens start finish «name»

def GlobalSymbolsSeparate (source : List U8) (scope : Scope) (entry : SourceToken) : Prop :=
  ∀ global ∈ scope.globals.val, ¬ NameEqual source global.name entry ∧
    ∀ label ∈ scope.labels.val, ¬ NameEqual source global.name label.name

def RegisterNamesSeparate (source : List U8) (scope : Scope) : Prop :=
  ∀ declaration ∈ scope.declarations.val,
    ∀ text ∈ ["%tid", "%ctaid", "%ntid", "%nctaid", "%laneid"],
      ¬ TokenSpelling source declaration.name text

def SymbolNameSupported (source : List U8) (token : SourceToken) : Prop :=
  ∀ text ∈ ["_", "$", "WARP_SZ"], ¬TokenSpelling source token text

def SymbolNamesSupported (source : List U8) (scope : Scope) (entry : SourceToken) : Prop :=
  SymbolNameSupported source entry ∧
  (∀ global ∈ scope.globals.val, SymbolNameSupported source global.name) ∧
  (∀ label ∈ scope.labels.val, SymbolNameSupported source label.name)

def SourceByte (byte : U8) : Prop :=
  (9 ≤ byte.val ∧ byte.val ≤ 13) ∨ (32 ≤ byte.val ∧ byte.val ≤ 126)

def SourceText (source : List U8) : Prop :=
  ∀ byte ∈ source, SourceByte byte

def IndexedStem (source : List U8) (token : SourceToken) : Prop :=
  token.word = true ∧ token.start.val < token.end.val ∧ token.end.val ≤ source.length ∧
    ∃ byte, source[token.end.val - 1]? = some byte ∧ (byte.val < 48 ∨ 57 < byte.val)

def IndexedNamesSupported (source : List U8) (scope : Scope) : Prop :=
  ∀ declaration ∈ scope.declarations.val, declaration.indexed = true →
    IndexedStem source declaration.name

def PredicateStem (source : List U8) (token : SourceToken) : Prop :=
  token.word = true ∧ token.start.val + 2 ≤ token.end.val ∧ token.end.val ≤ source.length ∧
    source[token.start.val]? = some 37#u8 ∧ source[token.start.val + 1]? = some 112#u8 ∧
    ∀ position byte, token.start.val + 2 ≤ position → position < token.end.val →
      source[position]? = some byte → 48 ≤ byte.val ∧ byte.val ≤ 57

def PredicateNamesSupported (source : List U8) (scope : Scope) : Prop :=
  ∀ declaration ∈ scope.declarations.val, declaration.predicate = true →
    PredicateStem source declaration.name

def ScopeSymbolsSeparate (source : List U8) (scope : Scope) (entry : SourceToken) : Prop :=
  GlobalSymbolsSeparate source scope entry ∧ RegisterNamesSeparate source scope ∧
    SymbolNamesSupported source scope entry ∧ SourceText source ∧ PredicateNamesSupported source scope ∧ IndexedNamesSupported source scope

def ModuleMeaning (source entry : List U8) (program : strict_memory_frontend_parse.SourceProgram) : Prop :=
  ∃ tokens globalsEnd declarationsStart instructionsStart finish,
    PtxFrontend.Lexed source (tokens.map tokenModel) ∧ HeaderSyntax source tokens ∧
    GlobalsSyntax source tokens 6 program.scope.globals.val globalsEnd ∧
    EntrySyntax source entry tokens globalsEnd declarationsStart program.entry ∧
    DeclarationsSyntax source tokens declarationsStart program.scope.declarations.val instructionsStart ∧
    CodeSyntax source program.scope tokens instructionsStart 0 program.code.val program.scope.labels.val finish ∧
    finish+1=tokens.length ∧ program.code.val≠[] ∧ DeclarationLayout program.scope ∧
    SeparateDeclarations source program.scope.declarations.val ∧
    program.scope.globals.val.Pairwise (fun left right => ¬NameEqual source left.«name» right.«name») ∧
    program.scope.labels.val.Pairwise (fun left right => ¬NameEqual source left.«name» right.«name») ∧
    ScopeSymbolsSeparate source program.scope program.entry

end PtxFrontendProgram
