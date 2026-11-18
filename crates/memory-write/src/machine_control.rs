/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::machine_registers::LaneRegisters;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ThreadCore<S> {
    pub pc: u32,
    pub st: S,
    pub registers: LaneRegisters,
    pub acq_bar: Option<u64>,
}

impl<S> std::ops::Deref for ThreadCore<S> {
    type Target = LaneRegisters;

    fn deref(&self) -> &Self::Target {
        &self.registers
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Flow {
    Next,
    Stay,
    Jump(u32),
}

pub fn set_pc<S>(state: &mut ThreadCore<S>, pc: u32) {
    state.pc = pc;
}

pub fn advance<S>(state: &mut ThreadCore<S>) {
    state.pc += 1;
}

pub fn set_status<S>(state: &mut ThreadCore<S>, status: S) {
    state.st = status;
}

pub fn apply_flow<S: PartialEq>(state: &mut ThreadCore<S>, pc: u32, flow: Flow, running: S) {
    match flow {
        Flow::Next if state.st == running => state.pc = pc + 1,
        Flow::Jump(target) => state.pc = target,
        _ => {}
    }
}

pub fn resume<S>(state: &mut ThreadCore<S>, running: S) {
    state.st = running;
    state.pc += 1;
}

pub fn release<S>(state: &mut ThreadCore<S>, running: S, key: u64) {
    resume(state, running);
    state.acq_bar = Some(key);
}

pub fn take_acquire<S>(state: &mut ThreadCore<S>) -> Option<u64> {
    state.acq_bar.take()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> ThreadCore<u8> {
        ThreadCore {
            pc: 10,
            st: 0,
            registers: LaneRegisters {
                regs: vec![11, 12],
                preds: vec![13],
                pdef: vec![14],
                written: vec![15],
                vund: vec![(1, 16)],
            },
            acq_bar: Some(17),
        }
    }

    #[test]
    fn blocked_and_staying_flows_preserve_control_and_registers() {
        for status in [0, 1] {
            for flow in [Flow::Next, Flow::Stay, Flow::Jump(20)] {
                let mut actual = state();
                actual.st = status;
                let before = actual.clone();
                apply_flow(&mut actual, 30, flow, 0);
                assert_eq!(actual.st, before.st);
                assert_eq!(actual.registers, before.registers);
                assert_eq!(actual.acq_bar, before.acq_bar);
                assert_eq!(
                    actual.pc,
                    match flow {
                        Flow::Next if status == 0 => 31,
                        Flow::Jump(_) => 20,
                        _ => 10,
                    }
                );
            }
        }
    }

    #[test]
    fn barrier_release_publishes_one_acquire_without_changing_registers() {
        let mut actual = state();
        let registers = actual.registers.clone();
        actual.st = 2;
        release(&mut actual, 0, 123);
        assert_eq!((actual.pc, actual.st, actual.acq_bar), (11, 0, Some(123)));
        assert_eq!(take_acquire(&mut actual), Some(123));
        assert_eq!(take_acquire(&mut actual), None);
        assert_eq!(actual.registers, registers);
        resume(&mut actual, 0);
        assert_eq!((actual.pc, actual.st, actual.acq_bar), (12, 0, None));
    }
}
