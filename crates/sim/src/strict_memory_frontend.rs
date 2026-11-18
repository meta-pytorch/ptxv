/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Token {
    pub start: usize,
    pub end: usize,
    pub word: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TriviaStep {
    Advance(usize),
    Done,
    Invalid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Scan {
    Token(Token),
    End,
    Invalid,
}

pub fn source_byte_supported(byte: u8) -> bool {
    (byte >= 9 && byte <= 13) || (byte >= 32 && byte <= 126)
}

pub fn source_text_supported(source: &[u8]) -> bool {
    let mut supported = true;
    let mut index = 0;
    while supported && index < source.len() {
        supported = source_byte_supported(source[index]);
        index += 1;
    }
    supported
}

pub fn is_space(byte: u8) -> bool {
    byte == 9 || byte == 10 || byte == 13 || byte == 32
}

pub fn is_word(byte: u8) -> bool {
    (byte >= b'a' && byte <= b'z')
        || (byte >= b'A' && byte <= b'Z')
        || (byte >= b'0' && byte <= b'9')
        || byte == b'_'
        || byte == b'.'
        || byte == b'$'
        || byte == b'%'
}

pub fn is_punctuation(byte: u8) -> bool {
    byte == b'('
        || byte == b')'
        || byte == b'{'
        || byte == b'}'
        || byte == b'['
        || byte == b']'
        || byte == b';'
        || byte == b','
        || byte == b'='
        || byte == b'+'
        || byte == b'-'
        || byte == b'<'
        || byte == b'>'
        || byte == b':'
        || byte == b'@'
        || byte == b'!'
}

pub fn scan_word(source: &[u8], start: usize) -> usize {
    let mut position = start;
    while position < source.len() && is_word(source[position]) {
        position += 1;
    }
    position
}

pub fn scan_line(source: &[u8], start: usize) -> usize {
    let mut position = start;
    while position < source.len() && source[position] != b'\n' {
        position += 1;
    }
    position
}

pub fn scan_block(source: &[u8], start: usize) -> Option<usize> {
    let mut position = start;
    let mut closed = false;
    while position < source.len() && !closed {
        if source[position] == b'*' && position + 1 < source.len() && source[position + 1] == b'/' {
            position += 2;
            closed = true;
        } else {
            position += 1;
        }
    }
    if closed { Some(position) } else { None }
}

pub fn trivia_step(source: &[u8], position: usize) -> TriviaStep {
    if position >= source.len() {
        return TriviaStep::Done;
    }
    let byte = source[position];
    if is_space(byte) {
        return TriviaStep::Advance(position + 1);
    }
    if byte == b'/' && position + 1 < source.len() {
        let following = source[position + 1];
        if following == b'/' {
            return TriviaStep::Advance(scan_line(source, position + 2));
        }
        if following == b'*' {
            return match scan_block(source, position + 2) {
                Some(after) => TriviaStep::Advance(after),
                None => TriviaStep::Invalid,
            };
        }
    }
    TriviaStep::Done
}

pub fn skip_trivia(source: &[u8], start: usize) -> Option<usize> {
    let mut position = start;
    let mut active = true;
    let mut valid = true;
    while active && valid {
        match trivia_step(source, position) {
            TriviaStep::Advance(after) => position = after,
            TriviaStep::Done => active = false,
            TriviaStep::Invalid => valid = false,
        }
    }
    if valid { Some(position) } else { None }
}

pub fn next_token(source: &[u8], start: usize) -> Scan {
    let position = match skip_trivia(source, start) {
        Some(position) => position,
        None => return Scan::Invalid,
    };
    if position == source.len() {
        return Scan::End;
    }
    if position > source.len() {
        return Scan::Invalid;
    }
    let byte = source[position];
    if is_word(byte) {
        Scan::Token(Token {
            start: position,
            end: scan_word(source, position),
            word: true,
        })
    } else if is_punctuation(byte) {
        Scan::Token(Token {
            start: position,
            end: position + 1,
            word: false,
        })
    } else {
        Scan::Invalid
    }
}

pub fn lex(source: &[u8]) -> Option<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut position = 0;
    let mut active = true;
    let mut valid = true;
    while active && valid {
        match next_token(source, position) {
            Scan::Token(token) => {
                position = token.end;
                tokens.push(token);
            }
            Scan::End => active = false,
            Scan::Invalid => valid = false,
        }
    }
    if valid { Some(tokens) } else { None }
}

pub fn token_equals(source: &[u8], token: Token, expected: &[u8]) -> bool {
    if token.start > token.end
        || token.end > source.len()
        || token.end - token.start != expected.len()
    {
        return false;
    }
    let mut matches = true;
    let mut index = 0;
    while index < expected.len() && matches {
        matches = source[token.start + index] == expected[index];
        index += 1;
    }
    matches
}

pub fn digit_value(byte: u8, radix: u64) -> Option<u64> {
    let digit = if byte >= b'0' && byte <= b'9' {
        u64::from(byte - b'0')
    } else if byte >= b'a' && byte <= b'f' {
        u64::from(byte - b'a') + 10
    } else if byte >= b'A' && byte <= b'F' {
        u64::from(byte - b'A') + 10
    } else {
        return None;
    };
    if digit < radix { Some(digit) } else { None }
}

pub fn parse_digits(source: &[u8], start: usize, end: usize, radix: u64) -> Option<u64> {
    if start >= end || end > source.len() || (radix != 10 && radix != 16) {
        return None;
    }
    let mut value = Some(0u64);
    let mut position = start;
    while position < end && value.is_some() {
        value = match (value, digit_value(source[position], radix)) {
            (Some(before), Some(digit)) => match before.checked_mul(radix) {
                Some(multiplied) => multiplied.checked_add(digit),
                None => None,
            },
            _ => None,
        };
        position += 1;
    }
    value
}

pub fn parse_unsigned(source: &[u8], token: Token) -> Option<u64> {
    if !token.word || token.start >= token.end || token.end > source.len() {
        return None;
    }
    if source[token.start] == b'0' && token.start + 1 < token.end {
        let following = source[token.start + 1];
        if following == b'x' || following == b'X' {
            parse_digits(source, token.start + 2, token.end, 16)
        } else {
            None
        }
    } else {
        parse_digits(source, token.start, token.end, 10)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comments_cannot_hide_an_instruction_or_unknown_byte() {
        let source = b"st.global.u64 [x], 0; // ignored\nld.global.u64 %r0,[x];";
        let tokens = lex(source).unwrap();
        assert!(token_equals(source, tokens[0], b"st.global.u64"));
        assert!(
            tokens
                .iter()
                .any(|&token| token_equals(source, token, b"ld.global.u64"))
        );
        assert_eq!(lex(b"nop; /* unfinished"), None);
        assert_eq!(lex(b"nop; \xff"), None);
        assert_eq!(lex(b"nop; \"not a directive\""), None);
    }

    #[test]
    fn block_comments_end_at_first_closer() {
        let source = b"/* before */nop;/* after */";
        let tokens = lex(source).unwrap();
        assert_eq!(tokens.len(), 2);
        assert!(token_equals(source, tokens[0], b"nop"));
        assert!(token_equals(source, tokens[1], b";"));
        assert_eq!(lex(b"/* outer /* nested */ trailing */"), None);
    }

    #[test]
    fn every_nontrivia_byte_belongs_to_a_token() {
        let source = b".entry k(){.reg .b64 %r0;ld.global.u64 %r0,[x+8];ret;}";
        let tokens = lex(source).unwrap();
        assert_eq!(tokens[0].start, 0);
        assert_eq!(tokens.last().unwrap().end, source.len());
        for pair in tokens.windows(2) {
            let after = skip_trivia(source, pair[0].end).unwrap();
            assert_eq!(after, pair[1].start);
        }
        assert_eq!(
            skip_trivia(source, tokens.last().unwrap().end),
            Some(source.len())
        );
    }

    #[test]
    fn unsigned_literals_are_unambiguous_and_do_not_wrap() {
        let cases: &[(&[u8], Option<u64>)] = &[
            (b"0", Some(0)),
            (b"18446744073709551615", Some(u64::MAX)),
            (b"18446744073709551616", None),
            (b"0xffffffffffffffff", Some(u64::MAX)),
            (b"0x10000000000000000", None),
            (b"0x", None),
            (b"010", None),
            (b"0b10", None),
            (b"1.0", None),
        ];
        for &(source, expected) in cases {
            let token = lex(source).unwrap()[0];
            assert_eq!(parse_unsigned(source, token), expected, "{source:?}");
        }
    }
}
