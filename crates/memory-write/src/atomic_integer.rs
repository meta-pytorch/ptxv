/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Integer atomic bit updates, normalized to the instruction's width.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum IntegerOp {
    Or,
    And,
    Xor,
    Add,
    Exch,
    MinU,
    MaxU,
    MinS,
    MaxS,
    Inc,
    Dec,
    Cas,
    CasSpin,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AtomicError {
    UnsupportedWidth,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntegerInput {
    pub op: IntegerOp,
    pub bytes: u8,
    pub old: u64,
    pub value: u64,
    pub compare: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IntegerResult {
    pub old: u64,
    pub replacement: u64,
    pub swapped: bool,
}

/// Computes the stored 32-bit pattern, including PTX's bounded increment/decrement.
pub fn replacement32(op: IntegerOp, old: u32, value: u32, compare: u32) -> u32 {
    match op {
        IntegerOp::Or => old | value,
        IntegerOp::And => old & value,
        IntegerOp::Xor => old ^ value,
        IntegerOp::Add => old.wrapping_add(value),
        IntegerOp::Exch => value,
        IntegerOp::MinU => {
            if old < value {
                old
            } else {
                value
            }
        }
        IntegerOp::MaxU => {
            if old > value {
                old
            } else {
                value
            }
        }
        IntegerOp::MinS => {
            if (old as i32) < (value as i32) {
                old
            } else {
                value
            }
        }
        IntegerOp::MaxS => {
            if (old as i32) > (value as i32) {
                old
            } else {
                value
            }
        }
        IntegerOp::Inc => {
            if old >= value {
                0
            } else {
                old + 1
            }
        }
        IntegerOp::Dec => {
            if old == 0 || old > value {
                value
            } else {
                old - 1
            }
        }
        IntegerOp::Cas | IntegerOp::CasSpin => {
            if old == compare {
                value
            } else {
                old
            }
        }
    }
}

/// Computes the stored 64-bit pattern; bounded increment/decrement are unsupported.
pub fn replacement64(
    op: IntegerOp,
    old: u64,
    value: u64,
    compare: u64,
) -> Result<u64, AtomicError> {
    let replacement = match op {
        IntegerOp::Or => old | value,
        IntegerOp::And => old & value,
        IntegerOp::Xor => old ^ value,
        IntegerOp::Add => old.wrapping_add(value),
        IntegerOp::Exch => value,
        IntegerOp::MinU => {
            if old < value {
                old
            } else {
                value
            }
        }
        IntegerOp::MaxU => {
            if old > value {
                old
            } else {
                value
            }
        }
        IntegerOp::MinS => {
            if (old as i64) < (value as i64) {
                old
            } else {
                value
            }
        }
        IntegerOp::MaxS => {
            if (old as i64) > (value as i64) {
                old
            } else {
                value
            }
        }
        IntegerOp::Inc | IntegerOp::Dec => return Err(AtomicError::UnsupportedWidth),
        IntegerOp::Cas | IntegerOp::CasSpin => {
            if old == compare {
                value
            } else {
                old
            }
        }
    };
    Ok(replacement)
}

pub fn is_cas(op: IntegerOp) -> bool {
    matches!(op, IntegerOp::Cas | IntegerOp::CasSpin)
}

/// Normalizes every operand before comparing or updating, then reports the old bits.
pub fn eval(input: IntegerInput) -> Result<IntegerResult, AtomicError> {
    let (old, compare, replacement) = if input.bytes == 4 {
        let old = input.old as u32;
        let value = input.value as u32;
        let compare = input.compare as u32;
        (
            old as u64,
            compare as u64,
            replacement32(input.op, old, value, compare) as u64,
        )
    } else if input.bytes == 2 && is_cas(input.op) {
        // `atom.cas.b16`: the operands' low halves (MISS 160).
        let old = input.old as u16 as u64;
        let compare = input.compare as u16 as u64;
        let value = input.value as u16 as u64;
        (old, compare, if old == compare { value } else { old })
    } else if input.bytes == 8 {
        let replacement = match replacement64(input.op, input.old, input.value, input.compare) {
            Ok(replacement) => replacement,
            Err(error) => return Err(error),
        };
        (input.old, input.compare, replacement)
    } else {
        return Err(AtomicError::UnsupportedWidth);
    };
    Ok(IntegerResult {
        old,
        replacement,
        swapped: is_cas(input.op) && old == compare,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(op: IntegerOp, bytes: u8, old: u64, value: u64, compare: u64) -> IntegerResult {
        eval(IntegerInput {
            op,
            bytes,
            old,
            value,
            compare,
        })
        .unwrap()
    }

    #[test]
    fn integer_width_and_signed_order() {
        assert_eq!(run(IntegerOp::Add, 4, u64::MAX, 1, 0).replacement, 0);
        assert_eq!(run(IntegerOp::Add, 8, u64::MAX, 1, 0).replacement, 0);
        assert_eq!(
            run(IntegerOp::MinS, 4, 0x8000_0000, 1, 0).replacement,
            0x8000_0000
        );
        assert_eq!(run(IntegerOp::MaxS, 4, 0xffff_ffff, 1, 0).replacement, 1);
        assert_eq!(run(IntegerOp::MinS, 8, 1 << 63, 1, 0).replacement, 1 << 63);
        assert_eq!(run(IntegerOp::MaxS, 8, u64::MAX, 1, 0).replacement, 1);
        assert_eq!(run(IntegerOp::MinU, 8, 1 << 63, 1, 0).replacement, 1);
    }

    #[test]
    fn compare_and_replacement_use_the_instruction_width() {
        let cas = run(
            IntegerOp::Cas,
            4,
            0xaaaa_0000_0000_0007,
            u64::MAX,
            0xbbbb_0000_0000_0007,
        );
        assert!(cas.swapped);
        assert_eq!((cas.old, cas.replacement), (7, u32::MAX as u64));
        let miss = run(IntegerOp::Cas, 8, 7, 9, 0x1_0000_0007);
        assert!(!miss.swapped);
        assert_eq!(miss.replacement, 7);
    }

    #[test]
    fn bounded_increment_and_decrement_cover_edges() {
        assert_eq!(run(IntegerOp::Inc, 4, 7, 7, 0).replacement, 0);
        assert_eq!(run(IntegerOp::Inc, 4, 6, 7, 0).replacement, 7);
        assert_eq!(run(IntegerOp::Dec, 4, 0, 7, 0).replacement, 7);
        assert_eq!(run(IntegerOp::Dec, 4, 8, 7, 0).replacement, 7);
        assert_eq!(run(IntegerOp::Dec, 4, 7, 7, 0).replacement, 6);
        for op in [IntegerOp::Inc, IntegerOp::Dec] {
            assert_eq!(
                eval(IntegerInput {
                    op,
                    bytes: 8,
                    old: 0,
                    value: 0,
                    compare: 0
                }),
                Err(AtomicError::UnsupportedWidth)
            );
        }
        assert_eq!(
            eval(IntegerInput {
                op: IntegerOp::Add,
                bytes: 2,
                old: 0,
                value: 0,
                compare: 0
            }),
            Err(AtomicError::UnsupportedWidth)
        );
    }
}
