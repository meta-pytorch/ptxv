/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ptxv_memory_write::value_flow as flow;

use crate::strict_memory_frontend::Token;
use crate::strict_memory_frontend::lex;
use crate::strict_memory_frontend::parse_digits;
use crate::strict_memory_frontend::parse_unsigned;
use crate::strict_memory_frontend::token_equals;

macro_rules! option_value {
    ($value:expr) => {
        match $value {
            Some(value) => value,
            None => return None,
        }
    };
}

pub(crate) use option_value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Declaration {
    pub name: Token,
    pub count: u16,
    pub first: u16,
    pub bits: u8,
    pub predicate: bool,
    pub indexed: bool,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Global {
    pub name: Token,
    pub alignment: u64,
    pub bytes: Vec<u8>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Label {
    pub name: Token,
    pub target: u32,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Scope {
    pub declarations: Vec<Declaration>,
    pub globals: Vec<Global>,
    pub labels: Vec<Label>,
    pub virtual_count: u16,
    pub predicate_count: u16,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SourceProgram {
    pub entry: Token,
    pub scope: Scope,
    pub code: Vec<flow::GuardedInstruction>,
}

pub fn names_equal(source: &[u8], left: Token, right: Token) -> bool {
    if left.start >= left.end
        || left.end > source.len()
        || right.start >= right.end
        || right.end > source.len()
        || left.end - left.start != right.end - right.start
    {
        return false;
    }
    let mut equal = true;
    let mut offset = 0;
    while equal && offset < left.end - left.start {
        equal = source[left.start + offset] == source[right.start + offset];
        offset += 1;
    }
    equal
}

pub fn identifier(source: &[u8], token: Token, register: bool) -> bool {
    if !token.word || token.start >= token.end || token.end > source.len() {
        return false;
    }
    let mut position = token.start;
    if register {
        if source[position] != b'%' {
            return false;
        }
        position += 1;
    }
    if position == token.end {
        return false;
    }
    let first = source[position];
    let mut valid = (first >= b'a' && first <= b'z')
        || (first >= b'A' && first <= b'Z')
        || first == b'_'
        || first == b'$';
    position += 1;
    while valid && position < token.end {
        let byte = source[position];
        valid = (byte >= b'a' && byte <= b'z')
            || (byte >= b'A' && byte <= b'Z')
            || (byte >= b'0' && byte <= b'9')
            || byte == b'_'
            || byte == b'$';
        position += 1;
    }
    valid
}

pub fn matches(source: &[u8], tokens: &[Token], at: usize, expected: &[u8]) -> bool {
    at < tokens.len() && token_equals(source, tokens[at], expected)
}

pub fn take(source: &[u8], tokens: &[Token], at: usize, expected: &[u8]) -> Option<usize> {
    if matches(source, tokens, at, expected) {
        at.checked_add(1)
    } else {
        None
    }
}

pub fn unsigned(source: &[u8], tokens: &[Token], at: usize) -> Option<(u64, usize)> {
    if at >= tokens.len() {
        return None;
    }
    match parse_unsigned(source, tokens[at]) {
        Some(value) => Some((value, at + 1)),
        None => None,
    }
}

pub fn register_type(source: &[u8], token: Token) -> Option<(u8, bool)> {
    if token_equals(source, token, b".pred") {
        Some((1, true))
    } else if token_equals(source, token, b".b32")
        || token_equals(source, token, b".u32")
        || token_equals(source, token, b".s32")
    {
        Some((32, false))
    } else if token_equals(source, token, b".b64")
        || token_equals(source, token, b".u64")
        || token_equals(source, token, b".s64")
    {
        Some((64, false))
    } else {
        None
    }
}

pub fn global_type(source: &[u8], token: Token) -> Option<u8> {
    if token_equals(source, token, b".b8") || token_equals(source, token, b".u8") {
        Some(1)
    } else if token_equals(source, token, b".b16") || token_equals(source, token, b".u16") {
        Some(2)
    } else if token_equals(source, token, b".b32") || token_equals(source, token, b".u32") {
        Some(4)
    } else if token_equals(source, token, b".b64") || token_equals(source, token, b".u64") {
        Some(8)
    } else {
        None
    }
}

pub fn decimal_prefix(source: &[u8], shorter: Token, longer: Token) -> bool {
    if shorter.start >= shorter.end
        || shorter.end > source.len()
        || longer.start >= longer.end
        || longer.end > source.len()
        || shorter.end - shorter.start >= longer.end - longer.start
    {
        return false;
    }
    let length = shorter.end - shorter.start;
    let head = Token {
        start: longer.start,
        end: longer.start + length,
        word: true,
    };
    names_equal(source, shorter, head) && source[head.end] >= b'0' && source[head.end] <= b'9'
}

pub fn declarations_separate(source: &[u8], left: Declaration, right: Declaration) -> bool {
    !names_equal(source, left.name, right.name)
        && !(left.indexed && decimal_prefix(source, left.name, right.name))
        && !(right.indexed && decimal_prefix(source, right.name, left.name))
}

pub fn parse_declaration(
    source: &[u8],
    tokens: &[Token],
    at: usize,
    scope: &Scope,
) -> Option<(Declaration, usize)> {
    let at = option_value!(take(source, tokens, at, b".reg"));
    if at + 1 >= tokens.len() {
        return None;
    }
    let (bits, predicate) = option_value!(register_type(source, tokens[at]));
    let name = tokens[at + 1];
    if !identifier(source, name, true) {
        return None;
    }
    let mut after = at + 2;
    let mut count = 1u16;
    let indexed = matches(source, tokens, after, b"<");
    if indexed {
        let (number, next) = option_value!(unsigned(source, tokens, after + 1));
        if number == 0 || number > u16::MAX as u64 {
            return None;
        }
        count = number as u16;
        after = option_value!(take(source, tokens, next, b">"));
    }
    after = option_value!(take(source, tokens, after, b";"));
    let first = if predicate {
        scope.predicate_count
    } else {
        scope.virtual_count
    };
    option_value!(first.checked_add(count));
    let mut index = 0;
    let mut fresh = true;
    while fresh && index < scope.declarations.len() {
        fresh = declarations_separate(
            source,
            scope.declarations[index],
            Declaration {
                name,
                count,
                first,
                bits,
                predicate,
                indexed,
            },
        );
        index += 1;
    }
    if fresh {
        Some((
            Declaration {
                name,
                count,
                first,
                bits,
                predicate,
                indexed,
            },
            after,
        ))
    } else {
        None
    }
}

pub fn declaration_register(source: &[u8], declaration: Declaration, name: Token) -> Option<u16> {
    if !identifier(source, name, true) {
        return None;
    }
    if !declaration.indexed {
        return if names_equal(source, declaration.name, name) {
            Some(declaration.first)
        } else {
            None
        };
    }
    let declared = declaration.name;
    if declared.end > source.len()
        || declared.start >= declared.end
        || name.end - name.start <= declared.end - declared.start
    {
        return None;
    }
    let length = declared.end - declared.start;
    let prefix = Token {
        start: name.start,
        end: name.start + length,
        word: true,
    };
    if !names_equal(source, declared, prefix) {
        return None;
    }
    let suffix = Token {
        start: prefix.end,
        end: name.end,
        word: true,
    };
    if suffix.end - suffix.start > 1 && source[suffix.start] == b'0' {
        return None;
    }
    let number = option_value!(parse_digits(source, suffix.start, suffix.end, 10));
    if number < declaration.count as u64 {
        declaration.first.checked_add(number as u16)
    } else {
        None
    }
}

pub fn lookup_register(
    source: &[u8],
    scope: &Scope,
    name: Token,
    bits: u8,
    predicate: bool,
) -> Option<u16> {
    let mut found = None;
    let mut unique = true;
    let mut index = 0;
    while unique && index < scope.declarations.len() {
        let declaration = scope.declarations[index];
        if let Some(register) = declaration_register(source, declaration, name) {
            unique =
                found.is_none() && declaration.bits == bits && declaration.predicate == predicate;
            found = Some(register);
        }
        index += 1;
    }
    if unique { found } else { None }
}

pub fn lookup_global(source: &[u8], scope: &Scope, name: Token) -> Option<usize> {
    let mut found = None;
    let mut unique = true;
    let mut index = 0;
    while unique && index < scope.globals.len() {
        if names_equal(source, scope.globals[index].name, name) {
            unique = found.is_none();
            found = Some(index);
        }
        index += 1;
    }
    if unique { found } else { None }
}

pub fn lookup_label(source: &[u8], scope: &Scope, name: Token) -> Option<u32> {
    let mut found = None;
    let mut unique = true;
    let mut index = 0;
    while unique && index < scope.labels.len() {
        let label = scope.labels[index];
        if names_equal(source, label.name, name) {
            unique = found.is_none();
            found = Some(label.target);
        }
        index += 1;
    }
    if unique { found } else { None }
}

pub fn parse_global(
    source: &[u8],
    tokens: &[Token],
    at: usize,
    scope: &Scope,
) -> Option<(Global, usize)> {
    let at = option_value!(take(source, tokens, at, b".global"));
    let at = option_value!(take(source, tokens, at, b".align"));
    let (alignment, at) = option_value!(unsigned(source, tokens, at));
    if alignment > 4_294_967_295 {
        return None;
    }
    if alignment == 0 || alignment & (alignment - 1) != 0 {
        return None;
    }
    if at + 1 >= tokens.len() {
        return None;
    }
    let width = option_value!(global_type(source, tokens[at]));
    if alignment < width as u64 {
        return None;
    }
    let name = tokens[at + 1];
    if !identifier(source, name, false) || lookup_global(source, scope, name).is_some() {
        return None;
    }
    let at = option_value!(take(source, tokens, at + 2, b"["));
    let (count, at) = option_value!(unsigned(source, tokens, at));
    if count == 0 || count > usize::MAX as u64 / width as u64 {
        return None;
    }
    let at = option_value!(take(source, tokens, at, b"]"));
    let at = option_value!(take(source, tokens, at, b"="));
    let at = option_value!(take(source, tokens, at, b"{"));
    let mut after = at;
    let mut bytes = Vec::new();
    let mut element = 0;
    let mut valid = true;
    while valid && element < count {
        match unsigned(source, tokens, after) {
            None => valid = false,
            Some((value, next)) => {
                after = next;
                if width != 8 && value >= 1u64 << (width * 8) {
                    valid = false;
                }
                let mut byte = 0;
                while valid && byte < width {
                    bytes.push((value >> (byte * 8)) as u8);
                    byte += 1;
                }
                element += 1;
                if element < count {
                    match take(source, tokens, after, b",") {
                        Some(next) => after = next,
                        None => valid = false,
                    }
                }
            }
        }
    }
    if !valid {
        return None;
    }
    let after = option_value!(take(source, tokens, after, b"}"));
    let after = option_value!(take(source, tokens, after, b";"));
    Some((
        Global {
            name,
            alignment,
            bytes,
        },
        after,
    ))
}

pub fn special(source: &[u8], token: Token) -> Option<flow::SpecialRegister> {
    if token_equals(source, token, b"%tid.x") {
        Some(flow::SpecialRegister::Thread(0))
    } else if token_equals(source, token, b"%tid.y") {
        Some(flow::SpecialRegister::Thread(1))
    } else if token_equals(source, token, b"%tid.z") {
        Some(flow::SpecialRegister::Thread(2))
    } else if token_equals(source, token, b"%ctaid.x") {
        Some(flow::SpecialRegister::Block(0))
    } else if token_equals(source, token, b"%ctaid.y") {
        Some(flow::SpecialRegister::Block(1))
    } else if token_equals(source, token, b"%ctaid.z") {
        Some(flow::SpecialRegister::Block(2))
    } else if token_equals(source, token, b"%ntid.x") {
        Some(flow::SpecialRegister::BlockSize(0))
    } else if token_equals(source, token, b"%ntid.y") {
        Some(flow::SpecialRegister::BlockSize(1))
    } else if token_equals(source, token, b"%ntid.z") {
        Some(flow::SpecialRegister::BlockSize(2))
    } else if token_equals(source, token, b"%nctaid.x") {
        Some(flow::SpecialRegister::GridSize(0))
    } else if token_equals(source, token, b"%nctaid.y") {
        Some(flow::SpecialRegister::GridSize(1))
    } else if token_equals(source, token, b"%nctaid.z") {
        Some(flow::SpecialRegister::GridSize(2))
    } else if token_equals(source, token, b"%laneid") {
        Some(flow::SpecialRegister::Lane)
    } else {
        None
    }
}

pub fn operand(source: &[u8], scope: &Scope, token: Token, bits: u8) -> Option<flow::Operand> {
    if bits != 32 && bits != 64 {
        return None;
    }
    if let Some(value) = parse_unsigned(source, token) {
        if bits == 32 && value > 0xffff_ffff {
            return None;
        }
        return Some(flow::Operand::Immediate(value));
    }
    if let Some(register) = lookup_register(source, scope, token, bits, false) {
        return Some(flow::Operand::Virtual(register));
    }
    if bits == 32 {
        if let Some(register) = special(source, token) {
            return Some(flow::Operand::Special(register));
        }
    }
    if bits == 64 {
        if let Some(allocation) = lookup_global(source, scope, token) {
            return Some(flow::Operand::Pointer(flow::Pointer {
                allocation,
                offset: 0,
            }));
        }
    }
    None
}

pub fn parse_operand(
    source: &[u8],
    tokens: &[Token],
    at: usize,
    scope: &Scope,
    bits: u8,
) -> Option<(flow::Operand, usize)> {
    if at >= tokens.len() {
        return None;
    }
    if bits != 32 && bits != 64 {
        return None;
    }
    if matches(source, tokens, at, b"-") {
        let (value, after) = option_value!(unsigned(source, tokens, at + 1));
        if (bits == 32 && value > 0x8000_0000) || (bits == 64 && value > 0x8000_0000_0000_0000) {
            return None;
        }
        Some((flow::Operand::Immediate(0u64.wrapping_sub(value)), after))
    } else {
        Some((
            option_value!(operand(source, scope, tokens[at], bits)),
            at + 1,
        ))
    }
}

pub fn parse_address_offset(source: &[u8], tokens: &[Token], at: usize) -> Option<(i64, usize)> {
    if matches(source, tokens, at, b"+") {
        let (number, after) = option_value!(unsigned(source, tokens, at + 1));
        if number > i64::MAX as u64 {
            return None;
        }
        Some((number as i64, after))
    } else {
        Some((0, at))
    }
}

pub fn parse_address_tail(
    source: &[u8],
    tokens: &[Token],
    at: usize,
    _scope: &Scope,
) -> Option<(flow::Operand, i64, usize)> {
    let (offset, after) = option_value!(parse_address_offset(source, tokens, at));
    Some((flow::Operand::Immediate(0), offset, after))
}

pub fn parse_address(
    source: &[u8],
    tokens: &[Token],
    at: usize,
    scope: &Scope,
) -> Option<(flow::Address, usize)> {
    let at = option_value!(take(source, tokens, at, b"["));
    if at >= tokens.len() {
        return None;
    }
    let base = option_value!(operand(source, scope, tokens[at], 64));
    let (index, offset, after) = option_value!(parse_address_tail(source, tokens, at + 1, scope));
    let after = option_value!(take(source, tokens, after, b"]"));
    Some((
        flow::Address {
            base,
            index,
            offset,
        },
        after,
    ))
}

pub fn parse_guard(
    source: &[u8],
    tokens: &[Token],
    at: usize,
    scope: &Scope,
) -> Option<(flow::Guard, usize)> {
    if !matches(source, tokens, at, b"@") {
        return Some((
            flow::Guard {
                predicate: None,
                negated: false,
            },
            at,
        ));
    }
    let mut after = at + 1;
    let negated = matches(source, tokens, after, b"!");
    if negated {
        after += 1;
    }
    if after >= tokens.len() {
        return None;
    }
    let predicate = option_value!(lookup_register(source, scope, tokens[after], 1, true));
    Some((
        flow::Guard {
            predicate: Some(predicate),
            negated,
        },
        after + 1,
    ))
}

pub fn parse_binary(source: &[u8], token: Token) -> Option<(u8, flow::BinaryOperation)> {
    if token_equals(source, token, b"add.u32") {
        Some((32, flow::BinaryOperation::Add))
    } else if token_equals(source, token, b"add.u64") {
        Some((64, flow::BinaryOperation::Add))
    } else if token_equals(source, token, b"sub.u32") {
        Some((32, flow::BinaryOperation::Subtract))
    } else if token_equals(source, token, b"sub.u64") {
        Some((64, flow::BinaryOperation::Subtract))
    } else if token_equals(source, token, b"and.b32") {
        Some((32, flow::BinaryOperation::And))
    } else if token_equals(source, token, b"and.b64") {
        Some((64, flow::BinaryOperation::And))
    } else if token_equals(source, token, b"or.b32") {
        Some((32, flow::BinaryOperation::Or))
    } else if token_equals(source, token, b"or.b64") {
        Some((64, flow::BinaryOperation::Or))
    } else if token_equals(source, token, b"xor.b32") {
        Some((32, flow::BinaryOperation::Xor))
    } else if token_equals(source, token, b"xor.b64") {
        Some((64, flow::BinaryOperation::Xor))
    } else if token_equals(source, token, b"mul.lo.u32") {
        Some((32, flow::BinaryOperation::MultiplyLow))
    } else if token_equals(source, token, b"mul.lo.u64") {
        Some((64, flow::BinaryOperation::MultiplyLow))
    } else {
        None
    }
}

pub fn parse_compare(source: &[u8], token: Token) -> Option<(u8, flow::Comparison)> {
    if token_equals(source, token, b"setp.eq.u32") {
        Some((32, flow::Comparison::Equal))
    } else if token_equals(source, token, b"setp.eq.u64") {
        Some((64, flow::Comparison::Equal))
    } else if token_equals(source, token, b"setp.ne.u32") {
        Some((32, flow::Comparison::NotEqual))
    } else if token_equals(source, token, b"setp.ne.u64") {
        Some((64, flow::Comparison::NotEqual))
    } else if token_equals(source, token, b"setp.lt.u32") {
        Some((32, flow::Comparison::LessUnsigned))
    } else if token_equals(source, token, b"setp.lt.u64") {
        Some((64, flow::Comparison::LessUnsigned))
    } else if token_equals(source, token, b"setp.le.u32") {
        Some((32, flow::Comparison::LessEqualUnsigned))
    } else if token_equals(source, token, b"setp.le.u64") {
        Some((64, flow::Comparison::LessEqualUnsigned))
    } else if token_equals(source, token, b"setp.gt.u32") {
        Some((32, flow::Comparison::GreaterUnsigned))
    } else if token_equals(source, token, b"setp.gt.u64") {
        Some((64, flow::Comparison::GreaterUnsigned))
    } else if token_equals(source, token, b"setp.ge.u32") {
        Some((32, flow::Comparison::GreaterEqualUnsigned))
    } else if token_equals(source, token, b"setp.ge.u64") {
        Some((64, flow::Comparison::GreaterEqualUnsigned))
    } else {
        None
    }
}

pub fn memory_type(source: &[u8], token: Token, load: bool) -> Option<(u8, bool)> {
    if load {
        if token_equals(source, token, b"ld.global.u8") {
            Some((1, false))
        } else if token_equals(source, token, b"ld.global.u16") {
            Some((2, false))
        } else if token_equals(source, token, b"ld.global.u32")
            || token_equals(source, token, b"ld.global.b32")
        {
            Some((4, false))
        } else if token_equals(source, token, b"ld.global.u64")
            || token_equals(source, token, b"ld.global.b64")
        {
            Some((8, false))
        } else if token_equals(source, token, b"ld.global.s8") {
            Some((1, true))
        } else if token_equals(source, token, b"ld.global.s16") {
            Some((2, true))
        } else if token_equals(source, token, b"ld.global.s32") {
            Some((4, true))
        } else {
            None
        }
    } else {
        if token_equals(source, token, b"st.global.u8")
            || token_equals(source, token, b"st.global.b8")
        {
            Some((1, false))
        } else if token_equals(source, token, b"st.global.u16")
            || token_equals(source, token, b"st.global.b16")
        {
            Some((2, false))
        } else if token_equals(source, token, b"st.global.u32")
            || token_equals(source, token, b"st.global.b32")
        {
            Some((4, false))
        } else if token_equals(source, token, b"st.global.u64")
            || token_equals(source, token, b"st.global.b64")
        {
            Some((8, false))
        } else {
            None
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Opcode {
    Exit,
    Nop,
    Branch,
    Move(u8),
    Binary(u8, flow::BinaryOperation),
    MultiplyWide,
    Convert(u8, u8),
    Compare(u8, flow::Comparison),
    Load(u8, bool),
    Store(u8),
}

pub fn classify(source: &[u8], token: Token) -> Option<Opcode> {
    if token_equals(source, token, b"ret") || token_equals(source, token, b"exit") {
        return Some(Opcode::Exit);
    }
    if token_equals(source, token, b"bra") {
        return Some(Opcode::Branch);
    }
    if token_equals(source, token, b"mov.u32") || token_equals(source, token, b"mov.b32") {
        return Some(Opcode::Move(32));
    }
    if token_equals(source, token, b"mov.u64")
        || token_equals(source, token, b"mov.b64")
        || token_equals(source, token, b"cvta.global.u64")
    {
        return Some(Opcode::Move(64));
    }
    if let Some((bits, operation)) = parse_binary(source, token) {
        return Some(Opcode::Binary(bits, operation));
    }
    if token_equals(source, token, b"mul.wide.u32") {
        return Some(Opcode::MultiplyWide);
    }
    if token_equals(source, token, b"cvt.u64.u32") {
        return Some(Opcode::Convert(32, 64));
    }
    if token_equals(source, token, b"cvt.u32.u64") {
        return Some(Opcode::Convert(64, 32));
    }
    if let Some((bits, comparison)) = parse_compare(source, token) {
        return Some(Opcode::Compare(bits, comparison));
    }
    if let Some((bytes, signed)) = memory_type(source, token, true) {
        return Some(Opcode::Load(bytes, signed));
    }
    if let Some((bytes, _)) = memory_type(source, token, false) {
        return Some(Opcode::Store(bytes));
    }
    None
}

pub fn parse_destination(
    source: &[u8],
    tokens: &[Token],
    at: usize,
    scope: &Scope,
    bits: u8,
    predicate: bool,
) -> Option<(u16, usize)> {
    if at >= tokens.len() {
        return None;
    }
    let destination = option_value!(lookup_register(source, scope, tokens[at], bits, predicate));
    let after = option_value!(take(source, tokens, at + 1, b","));
    Some((destination, after))
}

pub fn parse_move(
    source: &[u8],
    tokens: &[Token],
    at: usize,
    scope: &Scope,
    bits: u8,
) -> Option<(flow::Instruction, usize)> {
    let (destination, after) =
        option_value!(parse_destination(source, tokens, at, scope, bits, false));
    let (source, after) = option_value!(parse_operand(source, tokens, after, scope, bits));
    Some((
        flow::Instruction::Move {
            bits,
            destination,
            source,
        },
        after,
    ))
}

pub fn parse_binary_instruction(
    source: &[u8],
    tokens: &[Token],
    at: usize,
    scope: &Scope,
    bits: u8,
    operation: flow::BinaryOperation,
) -> Option<(flow::Instruction, usize)> {
    let (destination, after) =
        option_value!(parse_destination(source, tokens, at, scope, bits, false));
    let (left, after) = option_value!(parse_operand(source, tokens, after, scope, bits));
    let after = option_value!(take(source, tokens, after, b","));
    let (right, after) = option_value!(parse_operand(source, tokens, after, scope, bits));
    Some((
        flow::Instruction::Binary {
            bits,
            destination,
            operation,
            left,
            right,
        },
        after,
    ))
}

pub fn parse_wide(
    source: &[u8],
    tokens: &[Token],
    at: usize,
    scope: &Scope,
) -> Option<(flow::Instruction, usize)> {
    let (destination, after) =
        option_value!(parse_destination(source, tokens, at, scope, 64, false));
    let (left, after) = option_value!(parse_operand(source, tokens, after, scope, 32));
    let after = option_value!(take(source, tokens, after, b","));
    let (right, after) = option_value!(parse_operand(source, tokens, after, scope, 32));
    Some((
        flow::Instruction::MultiplyWide {
            destination,
            left,
            right,
        },
        after,
    ))
}

pub fn parse_convert(
    source: &[u8],
    tokens: &[Token],
    at: usize,
    scope: &Scope,
    from_bits: u8,
    to_bits: u8,
) -> Option<(flow::Instruction, usize)> {
    let (destination, after) =
        option_value!(parse_destination(source, tokens, at, scope, to_bits, false));
    let (source, after) = option_value!(parse_operand(source, tokens, after, scope, from_bits));
    Some((
        flow::Instruction::ConvertUnsigned {
            destination,
            source,
            from_bits,
            to_bits,
        },
        after,
    ))
}

pub fn parse_comparison_instruction(
    source: &[u8],
    tokens: &[Token],
    at: usize,
    scope: &Scope,
    bits: u8,
    comparison: flow::Comparison,
) -> Option<(flow::Instruction, usize)> {
    let (destination, after) = option_value!(parse_destination(source, tokens, at, scope, 1, true));
    let (left, after) = option_value!(parse_operand(source, tokens, after, scope, bits));
    let after = option_value!(take(source, tokens, after, b","));
    let (right, after) = option_value!(parse_operand(source, tokens, after, scope, bits));
    Some((
        flow::Instruction::Compare {
            bits,
            destination,
            comparison,
            left,
            right,
        },
        after,
    ))
}

pub fn parse_load(
    source: &[u8],
    tokens: &[Token],
    at: usize,
    scope: &Scope,
    bytes: u8,
    signed: bool,
) -> Option<(flow::Instruction, usize)> {
    if at >= tokens.len() {
        return None;
    }
    let name = tokens[at];
    let (destination, destination_bits) = if bytes == 8 {
        (
            option_value!(lookup_register(source, scope, name, 64, false)),
            64,
        )
    } else if let Some(register) = lookup_register(source, scope, name, 32, false) {
        (register, 32)
    } else {
        (
            option_value!(lookup_register(source, scope, name, 64, false)),
            64,
        )
    };
    let after = option_value!(take(source, tokens, at + 1, b","));
    let (address, after) = option_value!(parse_address(source, tokens, after, scope));
    Some((
        flow::Instruction::Load {
            destination,
            destination_bits,
            bytes,
            signed,
            address,
        },
        after,
    ))
}

pub fn parse_store(
    source: &[u8],
    tokens: &[Token],
    at: usize,
    scope: &Scope,
    bytes: u8,
) -> Option<(flow::Instruction, usize)> {
    let (address, after) = option_value!(parse_address(source, tokens, at, scope));
    let after = option_value!(take(source, tokens, after, b","));
    let bits = if bytes == 8 { 64 } else { 32 };
    let (source, after) = option_value!(parse_operand(source, tokens, after, scope, bits));
    Some((
        flow::Instruction::Store {
            bytes,
            address,
            source,
        },
        after,
    ))
}

pub fn parse_body(
    source: &[u8],
    tokens: &[Token],
    at: usize,
    scope: &Scope,
    opcode: Opcode,
) -> Option<(flow::Instruction, usize)> {
    match opcode {
        Opcode::Exit => Some((flow::Instruction::Exit, at)),
        Opcode::Nop => Some((flow::Instruction::Nop, at)),
        Opcode::Branch => {
            if at >= tokens.len() {
                return None;
            }
            Some((
                flow::Instruction::Branch {
                    target: option_value!(lookup_label(source, scope, tokens[at])),
                },
                at + 1,
            ))
        }
        Opcode::Move(bits) => parse_move(source, tokens, at, scope, bits),
        Opcode::Binary(bits, operation) => {
            parse_binary_instruction(source, tokens, at, scope, bits, operation)
        }
        Opcode::MultiplyWide => parse_wide(source, tokens, at, scope),
        Opcode::Convert(from_bits, to_bits) => {
            parse_convert(source, tokens, at, scope, from_bits, to_bits)
        }
        Opcode::Compare(bits, comparison) => {
            parse_comparison_instruction(source, tokens, at, scope, bits, comparison)
        }
        Opcode::Load(bytes, signed) => parse_load(source, tokens, at, scope, bytes, signed),
        Opcode::Store(bytes) => parse_store(source, tokens, at, scope, bytes),
    }
}

pub fn scalar_operand(operand: flow::Operand) -> bool {
    matches!(
        operand,
        flow::Operand::Immediate(_) | flow::Operand::Virtual(_)
    )
}

pub fn address_base(operand: flow::Operand) -> bool {
    matches!(
        operand,
        flow::Operand::Pointer(_) | flow::Operand::Virtual(_)
    )
}

pub fn instruction_operands_supported(instruction: flow::Instruction) -> bool {
    match instruction {
        flow::Instruction::Move { bits, source, .. } => match source {
            flow::Operand::Pointer(_) => bits == 64,
            flow::Operand::Special(_) => bits == 32,
            _ => true,
        },
        flow::Instruction::Binary { left, right, .. }
        | flow::Instruction::MultiplyWide { left, right, .. }
        | flow::Instruction::Compare { left, right, .. } => {
            scalar_operand(left) && scalar_operand(right)
        }
        flow::Instruction::ConvertUnsigned { source, .. } => scalar_operand(source),
        flow::Instruction::Load { address, .. } => address_base(address.base),
        flow::Instruction::Store {
            address, source, ..
        } => address_base(address.base) && scalar_operand(source),
        flow::Instruction::Branch { .. } | flow::Instruction::Exit => true,
        flow::Instruction::Nop => false,
    }
}

pub fn parse_instruction(
    source: &[u8],
    tokens: &[Token],
    at: usize,
    scope: &Scope,
) -> Option<(flow::GuardedInstruction, usize)> {
    let (guard, at) = option_value!(parse_guard(source, tokens, at, scope));
    if at >= tokens.len() {
        return None;
    }
    let opcode = option_value!(classify(source, tokens[at]));
    let (instruction, after) = option_value!(parse_body(source, tokens, at + 1, scope, opcode));
    if !instruction_operands_supported(instruction) {
        return None;
    }
    let after = option_value!(take(source, tokens, after, b";"));
    Some((flow::GuardedInstruction { guard, instruction }, after))
}

pub fn collect_labels(source: &[u8], tokens: &[Token], start: usize) -> Option<Vec<Label>> {
    let mut labels = Vec::<Label>::new();
    let mut position = start;
    let mut pc = 0u32;
    let mut statement_start = true;
    let mut valid = true;
    while valid && position < tokens.len() && !matches(source, tokens, position, b"}") {
        if statement_start
            && position + 1 < tokens.len()
            && matches(source, tokens, position + 1, b":")
        {
            let name = tokens[position];
            valid = identifier(source, name, false);
            let mut index = 0;
            while valid && index < labels.len() {
                valid = !names_equal(source, labels[index].name, name);
                index += 1;
            }
            labels.push(Label { name, target: pc });
            position += 2;
        } else {
            if matches(source, tokens, position, b";") {
                match pc.checked_add(1) {
                    Some(next) => pc = next,
                    None => valid = false,
                }
                statement_start = true;
            } else {
                statement_start = false;
            }
            position += 1;
        }
    }
    if valid && statement_start {
        Some(labels)
    } else {
        None
    }
}

pub fn version_supported(source: &[u8], token: Token) -> bool {
    token_equals(source, token, b"8.0")
        || token_equals(source, token, b"8.7")
        || token_equals(source, token, b"9.0")
}

pub fn target_supported(source: &[u8], token: Token) -> bool {
    token_equals(source, token, b"sm_80")
        || token_equals(source, token, b"sm_90")
        || token_equals(source, token, b"sm_90a")
        || token_equals(source, token, b"sm_100")
        || token_equals(source, token, b"sm_100a")
        || token_equals(source, token, b"sm_103a")
}

pub fn header_pair_supported(source: &[u8], version: Token, target: Token) -> bool {
    if token_equals(source, version, b"8.0") {
        token_equals(source, target, b"sm_80")
            || token_equals(source, target, b"sm_90")
            || token_equals(source, target, b"sm_90a")
    } else if token_equals(source, version, b"8.7") {
        token_equals(source, target, b"sm_80")
            || token_equals(source, target, b"sm_90")
            || token_equals(source, target, b"sm_90a")
            || token_equals(source, target, b"sm_100")
            || token_equals(source, target, b"sm_100a")
    } else {
        token_equals(source, version, b"9.0") && target_supported(source, target)
    }
}

pub fn parse_header(source: &[u8], tokens: &[Token]) -> Option<usize> {
    let at = option_value!(take(source, tokens, 0, b".version"));
    if at >= tokens.len() {
        return None;
    }
    let version = tokens[at];
    if !version_supported(source, version) {
        return None;
    }
    let at = option_value!(take(source, tokens, at + 1, b".target"));
    if at >= tokens.len() {
        return None;
    }
    if !header_pair_supported(source, version, tokens[at]) {
        return None;
    }
    let at = option_value!(take(source, tokens, at + 1, b".address_size"));
    let (address_size, position) = option_value!(unsigned(source, tokens, at));
    if address_size != 64 {
        return None;
    }
    Some(position)
}

pub fn parse_globals(
    source: &[u8],
    tokens: &[Token],
    mut position: usize,
    mut scope: Scope,
) -> Option<(Scope, usize)> {
    let mut valid = true;
    while valid && matches(source, tokens, position, b".global") {
        match parse_global(source, tokens, position, &scope) {
            Some((global, next)) => {
                scope.globals.push(global);
                position = next;
            }
            None => valid = false,
        }
    }
    if !valid {
        return None;
    }
    Some((scope, position))
}

pub fn parse_entry(
    source: &[u8],
    tokens: &[Token],
    mut position: usize,
    entry: &[u8],
) -> Option<(Token, usize)> {
    if matches(source, tokens, position, b".visible") {
        position += 1;
    }
    let position = option_value!(take(source, tokens, position, b".entry"));
    if position >= tokens.len() {
        return None;
    }
    let selected = tokens[position];
    if !identifier(source, selected, false) || !token_equals(source, selected, entry) {
        return None;
    }
    let position = option_value!(take(source, tokens, position + 1, b"("));
    let position = option_value!(take(source, tokens, position, b")"));
    let position = option_value!(take(source, tokens, position, b"{"));
    Some((selected, position))
}

pub fn parse_declarations(
    source: &[u8],
    tokens: &[Token],
    mut position: usize,
    mut scope: Scope,
) -> Option<(Scope, usize)> {
    let mut valid = true;
    while valid && matches(source, tokens, position, b".reg") {
        match parse_declaration(source, tokens, position, &scope) {
            Some((declaration, next)) => {
                if declaration.predicate {
                    scope.predicate_count += declaration.count;
                } else {
                    scope.virtual_count += declaration.count;
                }
                scope.declarations.push(declaration);
                position = next;
            }
            None => valid = false,
        }
    }
    if !valid {
        return None;
    }
    Some((scope, position))
}

pub fn label_matches(label: Label, name: Token, pc: usize) -> bool {
    label.name.start == name.start
        && label.name.end == name.end
        && label.name.word == name.word
        && label.target as usize == pc
}

pub fn parse_code(
    source: &[u8],
    tokens: &[Token],
    mut position: usize,
    scope: &Scope,
) -> Option<Vec<flow::GuardedInstruction>> {
    let mut valid = true;
    let mut code = Vec::new();
    let mut label_index = 0;
    while valid && position < tokens.len() && !matches(source, tokens, position, b"}") {
        if position + 1 < tokens.len() && matches(source, tokens, position + 1, b":") {
            if !identifier(source, tokens[position], false) || label_index >= scope.labels.len() {
                valid = false;
            } else if label_matches(scope.labels[label_index], tokens[position], code.len()) {
                label_index += 1;
                position += 2;
            } else {
                valid = false;
            }
        } else {
            match parse_instruction(source, tokens, position, scope) {
                Some((instruction, next)) => {
                    code.push(instruction);
                    position = next;
                }
                None => valid = false,
            }
        }
    }
    if !valid || (code.len() == 0) || label_index != scope.labels.len() {
        return None;
    }
    let after = option_value!(take(source, tokens, position, b"}"));
    if after != tokens.len() {
        return None;
    }
    Some(code)
}

pub fn global_symbol_supported(source: &[u8], name: Token, entry: Token, labels: &[Label]) -> bool {
    let mut valid = !names_equal(source, name, entry);
    let mut index = 0;
    while valid && index < labels.len() {
        valid = !names_equal(source, name, labels[index].name);
        index += 1;
    }
    valid
}

pub fn scope_global_symbols_supported(source: &[u8], scope: &Scope, entry: Token) -> bool {
    let mut valid = true;
    let mut index = 0;
    while valid && index < scope.globals.len() {
        valid = global_symbol_supported(source, scope.globals[index].name, entry, &scope.labels);
        index += 1;
    }
    valid
}

pub fn register_name_supported(source: &[u8], name: Token) -> bool {
    !token_equals(source, name, b"%tid")
        && !token_equals(source, name, b"%ctaid")
        && !token_equals(source, name, b"%ntid")
        && !token_equals(source, name, b"%nctaid")
        && !token_equals(source, name, b"%laneid")
}

pub fn scope_register_names_supported(source: &[u8], scope: &Scope) -> bool {
    let mut valid = true;
    let mut index = 0;
    while valid && index < scope.declarations.len() {
        valid = register_name_supported(source, scope.declarations[index].name);
        index += 1;
    }
    valid
}

pub fn indexed_stem_supported(source: &[u8], name: Token) -> bool {
    if !name.word || name.start >= name.end || name.end > source.len() {
        return false;
    }
    let byte = source[name.end - 1];
    byte < b'0' || byte > b'9'
}

pub fn scope_indexed_names_supported(source: &[u8], scope: &Scope) -> bool {
    let mut valid = true;
    let mut index = 0;
    while valid && index < scope.declarations.len() {
        let declaration = scope.declarations[index];
        if declaration.indexed {
            valid = indexed_stem_supported(source, declaration.name);
        }
        index += 1;
    }
    valid
}

pub fn predicate_stem_supported(source: &[u8], name: Token) -> bool {
    if !name.word || name.start > name.end || name.end > source.len() || name.end - name.start < 2 {
        return false;
    }
    if source[name.start] != b'%' || source[name.start + 1] != b'p' {
        return false;
    }
    let mut valid = true;
    let mut position = name.start + 2;
    while valid && position < name.end {
        let byte = source[position];
        valid = byte >= b'0' && byte <= b'9';
        position += 1;
    }
    valid
}

pub fn scope_predicate_names_supported(source: &[u8], scope: &Scope) -> bool {
    let mut valid = true;
    let mut index = 0;
    while valid && index < scope.declarations.len() {
        let declaration = scope.declarations[index];
        if declaration.predicate {
            valid = predicate_stem_supported(source, declaration.name);
        }
        index += 1;
    }
    valid
}

pub fn symbol_name_supported(source: &[u8], name: Token) -> bool {
    !token_equals(source, name, b"_")
        && !token_equals(source, name, b"$")
        && !token_equals(source, name, b"WARP_SZ")
}

pub fn scope_symbol_names_supported(source: &[u8], scope: &Scope, entry: Token) -> bool {
    let mut valid = symbol_name_supported(source, entry);
    let mut index = 0;
    while valid && index < scope.globals.len() {
        valid = symbol_name_supported(source, scope.globals[index].name);
        index += 1;
    }
    index = 0;
    while valid && index < scope.labels.len() {
        valid = symbol_name_supported(source, scope.labels[index].name);
        index += 1;
    }
    valid
}

pub fn scope_symbols_supported(source: &[u8], scope: &Scope, entry: Token) -> bool {
    scope_global_symbols_supported(source, scope, entry)
        && scope_register_names_supported(source, scope)
        && scope_symbol_names_supported(source, scope, entry)
        && crate::strict_memory_frontend::source_text_supported(source)
        && scope_predicate_names_supported(source, scope)
        && scope_indexed_names_supported(source, scope)
}

pub fn parse_source(source: &[u8], entry: &[u8]) -> Option<SourceProgram> {
    let tokens = option_value!(lex(source));
    let position = option_value!(parse_header(source, &tokens));
    let scope = Scope {
        declarations: Vec::new(),
        globals: Vec::new(),
        labels: Vec::new(),
        virtual_count: 0,
        predicate_count: 0,
    };
    let (scope, position) = option_value!(parse_globals(source, &tokens, position, scope));
    let (selected, position) = option_value!(parse_entry(source, &tokens, position, entry));
    let (mut scope, position) = option_value!(parse_declarations(source, &tokens, position, scope));
    scope.labels = option_value!(collect_labels(source, &tokens, position));
    if !scope_symbols_supported(source, &scope, selected) {
        return None;
    }
    let code = option_value!(parse_code(source, &tokens, position, &scope));
    Some(SourceProgram {
        entry: selected,
        scope,
        code,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn module(body: &str) -> String {
        format!(
            ".version 9.0 .target sm_103a .address_size 64 .global .align 8 .u64 x[2]={{0,0}}; .visible .entry k() {{ .reg .b32 %r<4>; .reg .b64 %rd<4>; .reg .pred %p<8>; {body} }}"
        )
    }

    #[test]
    fn builtin_vector_declaration_names_are_refused() {
        for builtin in ["tid", "ctaid", "ntid", "nctaid", "laneid"] {
            let source = module("ret;").replace(
                ".reg .b32 %r<4>;",
                &format!(".reg .b32 %r<4>; .reg .b32 %{builtin};"),
            );
            assert!(parse_source(source.as_bytes(), b"k").is_none());
        }
        let source = module("mov.u32 %laneid,7; ret;")
            .replace(".reg .b32 %r<4>;", ".reg .b32 %r<4>; .reg .b32 %laneid;");
        assert!(parse_source(source.as_bytes(), b"k").is_none());
    }

    #[test]
    fn declaration_prefix_aliases_are_refused() {
        for declarations in [
            ".reg .b32 %r<20>; .reg .b32 %r1<10>;",
            ".reg .b32 %r<20>; .reg .b32 %r12;",
        ] {
            let source = module("ret;").replace(".reg .b32 %r<4>;", declarations);
            assert!(parse_source(source.as_bytes(), b"k").is_none());
        }
        let source =
            module("ret;").replace(".reg .b32 %r<4>;", ".reg .b32 %r<4>; .reg .b32 %q<4>;");
        assert!(parse_source(source.as_bytes(), b"k").is_some());
    }

    #[test]
    fn reserved_nonregister_symbol_names_are_refused() {
        for name in ["_", "$", "WARP_SZ"] {
            for source in [
                module("ret;").replace(".u64 x[2]", &format!(".u64 {name}[2]")),
                module("ret;").replace(".entry k()", &format!(".entry {name}()")),
                module(&format!("{name}: ret;")),
            ] {
                let entry = if source.contains(&format!(".entry {name}()")) {
                    name
                } else {
                    "k"
                };
                assert!(parse_source(source.as_bytes(), entry.as_bytes()).is_none());
            }
            let source = module(&format!("mov.u32 %{name},7; ret;")).replace(
                ".reg .b32 %r<4>;",
                &format!(".reg .b32 %r<4>; .reg .b32 %{name};"),
            );
            assert!(parse_source(source.as_bytes(), b"k").is_some());
        }
    }

    #[test]
    fn retains_width_origin_and_dynamic_addresses() {
        let source = module(
            "mov.u64 %rd0,x; mov.u32 %r0,%tid.x; mul.wide.u32 %rd1,%r0,8; add.u64 %rd0,%rd0,%rd1; st.global.u64 [%rd0],7; ld.global.u64 %rd2,[%rd0]; ret;",
        );
        let parsed = parse_source(source.as_bytes(), b"k").unwrap();
        assert_eq!(parsed.code.len(), 7);
        assert_eq!(parsed.scope.globals[0].bytes.len(), 16);
        assert_eq!(parsed.scope.globals[0].alignment, 8);
        assert!(matches!(
            parsed.code[0].instruction,
            flow::Instruction::Move {
                bits: 64,
                source: flow::Operand::Pointer(flow::Pointer {
                    allocation: 0,
                    offset: 0
                }),
                ..
            }
        ));
    }

    #[test]
    fn ordinary_p7_and_forward_backward_labels() {
        let source = module(
            "mov.u32 %r0,0; loop: add.u32 %r0,%r0,1; setp.lt.u32 %p7,%r0,3; @%p7 bra loop; @!%p7 bra done; done: ret;",
        );
        let parsed = parse_source(source.as_bytes(), b"k").unwrap();
        assert_eq!(parsed.code[3].guard.predicate, Some(7));
        assert_eq!(
            parsed.code[3].instruction,
            flow::Instruction::Branch { target: 1 }
        );
        assert_eq!(
            parsed.code[4].instruction,
            flow::Instruction::Branch { target: 5 }
        );
        assert!(parsed.code[4].guard.negated);
    }

    #[test]
    fn refuses_width_modifier_unknown_and_trailing_inputs() {
        for body in [
            "nop; ret;",
            "mov.u64 %r0,x; ret;",
            "mov.u32 %r0x1,7; ret;",
            "mov.u32 %r01,7; ret;",
            "mov.u32 %r0,4294967296; ret;",
            "mov.u32 %r0,-2147483649; ret;",
            "mov.u64 %rd0,-9223372036854775809; ret;",
            "ld.global.volatile.u64 %rd0,[x]; ret;",
            ".ignored x; ret;",
            "@%p8 ret;",
            "bra missing;",
            "ret; } .entry other() { ret;",
        ] {
            assert!(
                parse_source(module(body).as_bytes(), b"k").is_none(),
                "{body}"
            );
        }
        assert!(parse_source(module("ret;").as_bytes(), b"other").is_none());
        let overlapping = module("ret;").replace(".reg .b64", ".reg .b32 %r1; .reg .b64");
        assert!(parse_source(overlapping.as_bytes(), b"k").is_none());
    }

    #[test]
    fn preserves_object_extent_in_cross_object_escape() {
        let source = module("ld.global.u64 %rd0,[x+4294967296]; ret;");
        let parsed = parse_source(source.as_bytes(), b"k").unwrap();
        let flow::Instruction::Load { address, .. } = parsed.code[0].instruction else {
            panic!()
        };
        assert_eq!(
            address.base,
            flow::Operand::Pointer(flow::Pointer {
                allocation: 0,
                offset: 0
            })
        );
        assert_eq!(address.offset, 4294967296);
        assert_eq!(parsed.scope.globals[0].bytes.len(), 16);
    }
    #[test]
    fn positive_address_offsets_and_index_tails() {
        for (expression, expected_index, expected_offset) in [
            ("x", flow::Operand::Immediate(0), 0),
            ("x+8", flow::Operand::Immediate(0), 8),
            (
                "x+9223372036854775807",
                flow::Operand::Immediate(0),
                i64::MAX,
            ),
        ] {
            let source = module(&format!("ld.global.u64 %rd0,[{expression}]; ret;"));
            let parsed = parse_source(source.as_bytes(), b"k").unwrap();
            let flow::Instruction::Load { address, .. } = parsed.code[0].instruction else {
                panic!()
            };
            assert_eq!(address.index, expected_index, "{expression}");
            assert_eq!(address.offset, expected_offset, "{expression}");
        }
        for expression in [
            "x+%rd1",
            "x+%rd1+8",
            "x-8",
            "x+%rd1-8",
            "x+-8",
            "x+9223372036854775808",
            "x-9223372036854775808",
            "x+%rd1+9223372036854775808",
            "x+%rd1-9223372036854775808",
            "x+",
            "x-",
            "x+%rd1+",
        ] {
            let source = module(&format!("ld.global.u64 %rd0,[{expression}]; ret;"));
            assert!(
                parse_source(source.as_bytes(), b"k").is_none(),
                "{expression}"
            );
        }
    }
    #[test]
    fn checked_label_inventory_rejects_inconsistent_prepass() {
        let source = module("first: alias: mov.u32 %r0,0; bra later; later: @%p0 bra first; ret;");
        let parsed = parse_source(source.as_bytes(), b"k").unwrap();
        assert_eq!(
            parsed
                .scope
                .labels
                .iter()
                .map(|label| label.target)
                .collect::<Vec<_>>(),
            vec![0, 0, 2]
        );
        assert_eq!(
            parsed.code[1].instruction,
            flow::Instruction::Branch { target: 2 }
        );
        assert_eq!(
            parsed.code[2].instruction,
            flow::Instruction::Branch { target: 0 }
        );
        let source = source.as_bytes();
        let tokens = lex(source).unwrap();
        let position = parse_header(source, &tokens).unwrap();
        let scope = Scope {
            declarations: Vec::new(),
            globals: Vec::new(),
            labels: Vec::new(),
            virtual_count: 0,
            predicate_count: 0,
        };
        let (scope, position) = parse_globals(source, &tokens, position, scope).unwrap();
        let (_, position) = parse_entry(source, &tokens, position, b"k").unwrap();
        let (mut scope, position) = parse_declarations(source, &tokens, position, scope).unwrap();
        scope.labels = collect_labels(source, &tokens, position).unwrap();
        assert!(parse_code(source, &tokens, position, &scope).is_some());
        for corruption in 0..6 {
            let mut corrupted = scope.clone();
            match corruption {
                0 => corrupted.labels[0].target += 1,
                1 => corrupted.labels[0].name.start += 1,
                2 => corrupted.labels[0].name.end -= 1,
                3 => corrupted.labels[0].name.word = false,
                4 => {
                    corrupted.labels.pop();
                }
                _ => corrupted.labels.push(corrupted.labels[0]),
            }
            assert!(
                parse_code(source, &tokens, position, &corrupted).is_none(),
                "corruption {corruption}"
            );
        }
    }
    #[test]
    fn header_version_target_pairs_must_be_assembler_compatible() {
        for version in ["8.0", "8.7", "9.0"] {
            for target in ["sm_80", "sm_90", "sm_90a", "sm_100", "sm_100a", "sm_103a"] {
                let source = module("ret;")
                    .replace(".version 9.0", &format!(".version {version}"))
                    .replace(".target sm_103a", &format!(".target {target}"));
                let supported = match version {
                    "8.0" => matches!(target, "sm_80" | "sm_90" | "sm_90a"),
                    "8.7" => target != "sm_103a",
                    _ => true,
                };
                assert_eq!(
                    parse_source(source.as_bytes(), b"k").is_some(),
                    supported,
                    "{version}/{target}"
                );
            }
        }
    }

    #[test]
    fn global_symbols_cannot_be_shadowed_by_entry_or_labels() {
        let entry_collision = module("ret;").replace(".u64 x[2]", ".u64 k[2]");
        assert!(parse_source(entry_collision.as_bytes(), b"k").is_none());
        assert!(parse_source(module("x: ld.global.u64 %rd0,[x]; ret;").as_bytes(), b"k").is_none());
        assert!(parse_source(module("k: ret;").as_bytes(), b"k").is_some());
        let keyword_symbol = module("ret;").replace(".u64 x[2]", ".u64 ret[2]");
        assert!(parse_source(keyword_symbol.as_bytes(), b"k").is_some());
    }
}
