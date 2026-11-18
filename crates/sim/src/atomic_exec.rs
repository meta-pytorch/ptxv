/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ptxv_ir::AtomOp;
use ptxv_ir::Rnd;
use ptxv_memory_write::atomic_exec::FloatingArithmetic;
use ptxv_memory_write::atomic_exec::Operation;
use ptxv_memory_write::atomic_integer::IntegerOp;

use crate::fp;

pub(crate) struct Arithmetic;

impl FloatingArithmetic for Arithmetic {
    fn add_f32_ftz(old: u32, value: u32) -> u32 {
        fp::f32_fma(old, 0x3f80_0000, Some(value), Rnd::Rn, true)
    }

    fn add_f64(old: u64, value: u64) -> u64 {
        fp::f64_fma(old, 1f64.to_bits(), Some(value), Rnd::Rn)
    }

    fn add_bf16(old: u16, value: u16) -> u16 {
        fp::bf16_fma(old, 0x3f80, value)
    }
}

pub(crate) fn operation(op: AtomOp) -> Operation {
    match op {
        AtomOp::Or => Operation::Integer(IntegerOp::Or),
        AtomOp::And => Operation::Integer(IntegerOp::And),
        AtomOp::Xor => Operation::Integer(IntegerOp::Xor),
        AtomOp::Add => Operation::Integer(IntegerOp::Add),
        AtomOp::Exch => Operation::Integer(IntegerOp::Exch),
        AtomOp::MinU => Operation::Integer(IntegerOp::MinU),
        AtomOp::MaxU => Operation::Integer(IntegerOp::MaxU),
        AtomOp::MinS => Operation::Integer(IntegerOp::MinS),
        AtomOp::MaxS => Operation::Integer(IntegerOp::MaxS),
        AtomOp::Inc => Operation::Integer(IntegerOp::Inc),
        AtomOp::Dec => Operation::Integer(IntegerOp::Dec),
        AtomOp::Cas => Operation::Integer(IntegerOp::Cas),
        AtomOp::CasSpin => Operation::Integer(IntegerOp::CasSpin),
        AtomOp::FAddFtz => Operation::AddF32Ftz,
        AtomOp::FAdd64 => Operation::AddF64,
        AtomOp::AddBf16 => Operation::AddBf16,
        AtomOp::AddBf16x2 => Operation::AddBf16x2,
    }
}
