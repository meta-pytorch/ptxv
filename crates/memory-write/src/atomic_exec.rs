/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Atomic computation and architectural results at one resolved memory location.

use crate::atomic_integer;
pub use crate::atomic_integer::AtomicError;
use crate::atomic_integer::IntegerInput;
use crate::atomic_integer::IntegerOp;
use crate::atomic_integer::IntegerResult;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Operation {
    Integer(IntegerOp),
    AddF32Ftz,
    AddF64,
    AddBf16,
    AddBf16x2,
}

/// Floating instruction bit results supplied by the arithmetic subsystem.
pub trait FloatingArithmetic {
    fn add_f32_ftz(old: u32, value: u32) -> u32;
    fn add_f64(old: u64, value: u64) -> u64;
    fn add_bf16(old: u16, value: u16) -> u16;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtomicInput {
    pub op: Operation,
    pub bytes: u8,
    pub old: u64,
    pub value: u64,
    pub compare: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtomicRequest {
    pub op: Operation,
    pub bytes: u8,
    pub value: u64,
    pub compare: u64,
}

pub fn resolve_old(request: AtomicRequest, old: u64) -> AtomicInput {
    AtomicInput {
        op: request.op,
        bytes: request.bytes,
        old,
        value: request.value,
        compare: request.compare,
    }
}

pub fn fadd_operand(request: AtomicRequest) -> Option<u32> {
    if matches!(request.op, Operation::AddF32Ftz) && request.bytes == 4 {
        Some(request.value as u32)
    } else {
        None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtomicOutcome {
    pub bytes: u8,
    pub old: u64,
    pub replacement: u64,
    pub swapped: bool,
    pub did_write: bool,
    pub returned: u64,
    pub return_words: u8,
    pub predicate: bool,
    pub advance_epoch: bool,
}

/// Rejects unsupported widths before arithmetic or caller memory effects.
pub fn validate(op: Operation, bytes: u8) -> Result<(), AtomicError> {
    let supported = match op {
        Operation::Integer(IntegerOp::Inc | IntegerOp::Dec) => bytes == 4,
        Operation::Integer(_) => bytes == 4 || bytes == 8,
        Operation::AddF32Ftz | Operation::AddBf16x2 => bytes == 4,
        Operation::AddF64 => bytes == 8,
        Operation::AddBf16 => bytes == 2,
    };
    if supported {
        Ok(())
    } else {
        Err(AtomicError::UnsupportedWidth)
    }
}

pub fn finish(input: AtomicInput, result: IntegerResult) -> AtomicOutcome {
    let cas = matches!(input.op, Operation::Integer(IntegerOp::Cas));
    let spin = matches!(input.op, Operation::Integer(IntegerOp::CasSpin));
    let did_write = !cas || result.swapped;
    AtomicOutcome {
        bytes: input.bytes,
        old: result.old,
        replacement: result.replacement,
        swapped: result.swapped,
        did_write,
        returned: if spin {
            result.swapped as u64
        } else {
            result.old
        },
        return_words: if spin || input.bytes != 8 { 1 } else { 2 },
        predicate: !spin || result.swapped,
        advance_epoch: did_write,
    }
}

/// Produces one location's replacement, return value, and memory/order decisions.
pub fn execute<F: FloatingArithmetic>(input: AtomicInput) -> Result<AtomicOutcome, AtomicError> {
    match validate(input.op, input.bytes) {
        Ok(()) => {}
        Err(error) => return Err(error),
    }
    let result = match input.op {
        Operation::Integer(op) => match atomic_integer::eval(IntegerInput {
            op,
            bytes: input.bytes,
            old: input.old,
            value: input.value,
            compare: input.compare,
        }) {
            Ok(result) => result,
            Err(error) => return Err(error),
        },
        Operation::AddF32Ftz => {
            let old = input.old as u32;
            let value = input.value as u32;
            IntegerResult {
                old: old as u64,
                replacement: F::add_f32_ftz(old, value) as u64,
                swapped: false,
            }
        }
        Operation::AddF64 => IntegerResult {
            old: input.old,
            replacement: F::add_f64(input.old, input.value),
            swapped: false,
        },
        Operation::AddBf16 => {
            let old = input.old as u16;
            let value = input.value as u16;
            IntegerResult {
                old: old as u64,
                replacement: F::add_bf16(old, value) as u64,
                swapped: false,
            }
        }
        Operation::AddBf16x2 => {
            let low = F::add_bf16(input.old as u16, input.value as u16) as u64;
            let high = F::add_bf16((input.old >> 16) as u16, (input.value >> 16) as u16) as u64;
            IntegerResult {
                old: input.old as u32 as u64,
                replacement: low | (high << 16),
                swapped: false,
            }
        }
    };
    Ok(finish(input, result))
}

#[cfg(test)]
mod tests {
    use super::*;

    struct NoFloat;

    impl FloatingArithmetic for NoFloat {
        fn add_f32_ftz(_: u32, _: u32) -> u32 {
            panic!("unexpected floating arithmetic")
        }
        fn add_f64(_: u64, _: u64) -> u64 {
            panic!("unexpected floating arithmetic")
        }
        fn add_bf16(_: u16, _: u16) -> u16 {
            panic!("unexpected floating arithmetic")
        }
    }

    #[test]
    fn compare_outcomes_drive_all_effects_and_returns() {
        for bytes in [4, 8] {
            for op in [IntegerOp::Cas, IntegerOp::CasSpin] {
                for compare in [7, 9] {
                    let out = execute::<NoFloat>(AtomicInput {
                        op: Operation::Integer(op),
                        bytes,
                        old: 7,
                        value: 11,
                        compare,
                    })
                    .unwrap();
                    let swapped = compare == 7;
                    let spin = op == IntegerOp::CasSpin;
                    assert_eq!(out.replacement, if swapped { 11 } else { 7 });
                    assert_eq!(out.swapped, swapped);
                    assert_eq!(out.did_write, spin || swapped);
                    assert_eq!(out.advance_epoch, out.did_write);
                    assert_eq!(out.returned, if spin { swapped as u64 } else { 7 });
                    assert_eq!(out.return_words, if !spin && bytes == 8 { 2 } else { 1 });
                    assert_eq!(out.predicate, !spin || swapped);
                }
            }
        }
    }

    #[test]
    fn invalid_float_widths_reject_before_arithmetic() {
        for (op, valid) in [
            (Operation::AddF32Ftz, 4),
            (Operation::AddF64, 8),
            (Operation::AddBf16, 2),
            (Operation::AddBf16x2, 4),
        ] {
            for bytes in [0, 1, 2, 3, 4, 8, 16] {
                if bytes != valid {
                    assert_eq!(
                        execute::<NoFloat>(AtomicInput {
                            op,
                            bytes,
                            old: 0,
                            value: 0,
                            compare: 0
                        }),
                        Err(AtomicError::UnsupportedWidth)
                    );
                }
            }
        }
    }

    struct Routes;

    impl FloatingArithmetic for Routes {
        fn add_f32_ftz(old: u32, value: u32) -> u32 {
            assert_eq!((old, value), (0x1122_3344, 0x5566_7788));
            0xabcd_ef01
        }
        fn add_f64(old: u64, value: u64) -> u64 {
            assert_eq!((old, value), (0xffff_0000_1122_3344, 0xeeee_0000_5566_7788));
            u64::MAX
        }
        fn add_bf16(old: u16, value: u16) -> u16 {
            match (old, value) {
                (0x3344, 0x7788) => 0x1234,
                (0x1122, 0x5566) => 0xabcd,
                _ => panic!("wrong packed lane"),
            }
        }
    }

    #[test]
    fn request_log_and_arithmetic_share_the_actual_value() {
        let request = AtomicRequest {
            op: Operation::AddF32Ftz,
            bytes: 4,
            value: 0xeeee_0000_5566_7788,
            compare: 0x1234_5678,
        };
        let input = resolve_old(request, 0xffff_0000_1122_3344);
        let out = execute::<Routes>(input).unwrap();
        assert_eq!(fadd_operand(request), Some(0x5566_7788));
        assert_eq!(out.replacement, 0xabcd_ef01);
        assert_eq!(input.compare, 0x1234_5678);
        assert_eq!(
            fadd_operand(AtomicRequest {
                bytes: 8,
                ..request
            }),
            None
        );
        assert_eq!(
            fadd_operand(AtomicRequest {
                op: Operation::AddBf16x2,
                ..request
            }),
            None
        );
    }

    #[test]
    fn floating_dispatch_preserves_width_and_packed_lanes() {
        for (op, bytes, expected) in [
            (Operation::AddF32Ftz, 4, 0xabcd_ef01),
            (Operation::AddF64, 8, u64::MAX),
            (Operation::AddBf16, 2, 0x1234),
            (Operation::AddBf16x2, 4, 0xabcd_1234),
        ] {
            let out = execute::<Routes>(AtomicInput {
                op,
                bytes,
                old: 0xffff_0000_1122_3344,
                value: 0xeeee_0000_5566_7788,
                compare: 0,
            })
            .unwrap();
            assert_eq!(out.replacement, expected);
            assert!(out.did_write && out.advance_epoch && out.predicate);
            assert!(!out.swapped);
        }
    }
}
