/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::effects;
use crate::effects::AccessKind;
use crate::effects::Effect;
use crate::effects::Visibility;
use crate::machine_recorder::Access;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct LaneRegisters {
    pub regs: Vec<u64>,
    pub preds: Vec<u64>,
    pub pdef: Vec<u64>,
    pub written: Vec<u64>,
    pub vund: Vec<(u16, u64)>,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WarpRegisters {
    pub ur: Vec<u64>,
    pub ur_written: [u64; 4],
    pub up: u8,
    pub up_def: u8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WordWrite {
    pub index: u16,
    pub value: u64,
    pub slots: u8,
    pub mask: Option<u64>,
    pub stale: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PredicateWrite {
    pub index: u16,
    pub value: bool,
    pub defined: bool,
}

pub fn set_word(words: &mut Vec<u64>, index: usize, value: u64) {
    words[index] = value;
}

pub fn set_bit(words: &mut Vec<u64>, index: u16, enabled: bool) {
    let word = usize::from(index / 64);
    let bit = 1u64 << (index % 64);
    words[word] = (words[word] & !bit) | if enabled { bit } else { 0 };
}

pub fn set_virtual_undefined(registers: &mut LaneRegisters, index: u16, mask: u64) {
    let mut at = 0;
    while at < registers.vund.len() && registers.vund[at].0 < index {
        at += 1;
    }
    let present = at < registers.vund.len() && registers.vund[at].0 == index;
    if mask != 0 && mask != u64::MAX {
        if present {
            registers.vund[at].1 = mask;
        } else if at == registers.vund.len() {
            registers.vund.push((index, mask));
        } else {
            registers.vund.insert(at, (index, mask));
        }
    } else if present {
        let mut retained = Vec::new();
        let mut index = 0;
        while index < registers.vund.len() {
            if index != at {
                retained.push(registers.vund[index]);
            }
            index += 1;
        }
        registers.vund = retained;
    }
}

pub fn write_virtual(registers: &mut LaneRegisters, index: u16, value: u64, undefined: u64) {
    set_word(&mut registers.regs, usize::from(index), value);
    set_bit(&mut registers.written, index, undefined != u64::MAX);
    set_virtual_undefined(registers, index, undefined);
}

pub fn slot_undefined(mask: Option<u64>, stale: bool, slot: u8) -> u32 {
    if stale {
        u32::MAX
    } else {
        match mask {
            Some(mask) if slot < 2 => (mask >> (32 * slot)) as u32,
            Some(mask) if mask != 0 => u32::MAX,
            _ => 0,
        }
    }
}

pub fn write_scalar(registers: &mut LaneRegisters, write: WordWrite) {
    if write.index >= 255 {
        return;
    }
    assert!(
        write.slots <= 2,
        "register write exceeds its 64-bit payload"
    );
    let count = write.slots.max(1);
    let mut slot = 0;
    while slot < count {
        let index = write.index + u16::from(slot);
        let undefined = slot_undefined(write.mask, write.stale, slot);
        if index < 255 {
            let value = (write.value >> (32 * slot)) & 0xffff_ffff
                | if undefined == u32::MAX {
                    0
                } else {
                    u64::from(undefined) << 32
                };
            set_word(&mut registers.regs, usize::from(index), value);
            set_bit(&mut registers.written, index, undefined != u32::MAX);
        }
        slot += 1;
    }
}

pub fn write_predicate(registers: &mut LaneRegisters, index: u16, value: bool, defined: bool) {
    set_bit(&mut registers.preds, index, value);
    set_bit(&mut registers.pdef, index, defined);
}

pub fn uniform_access(warp: u32) -> Access {
    Access {
        resource: effects::uniform_register(warp),
        kind: AccessKind::Write,
        visibility: Visibility::All,
    }
}

pub fn uniform_effect(warp: u32) -> Effect {
    let access = uniform_access(warp);
    Effect::Access {
        resource: access.resource,
        kind: access.kind,
        visibility: access.visibility,
    }
}

pub fn write_uniform(registers: &mut WarpRegisters, warp: u32, write: WordWrite) -> Option<Access> {
    if write.index >= 255 {
        return None;
    }
    assert!(
        write.slots <= 2,
        "register write exceeds its 64-bit payload"
    );
    let count = write.slots.max(1);
    let mut slot = 0;
    while slot < count {
        let index = write.index + u16::from(slot);
        let undefined = slot_undefined(write.mask, write.stale, slot);
        if index < 255 {
            registers.ur[usize::from(index)] = (write.value >> (32 * slot)) & 0xffff_ffff
                | if undefined == u32::MAX {
                    0
                } else {
                    u64::from(undefined) << 32
                };
            let word = usize::from(index / 64);
            let bit = 1u64 << (index % 64);
            registers.ur_written[word] =
                (registers.ur_written[word] & !bit) | if undefined == u32::MAX { 0 } else { bit };
        }
        slot += 1;
    }
    Some(uniform_access(warp))
}

pub fn write_uniform_predicate(
    registers: &mut WarpRegisters,
    warp: u32,
    write: PredicateWrite,
) -> Option<Access> {
    assert!(
        write.index < 8,
        "uniform predicate index exceeds its register file"
    );
    let bit = 1u8 << write.index;
    registers.up = (registers.up & !bit) | if write.value { bit } else { 0 };
    registers.up_def = (registers.up_def & !bit) | if write.defined { bit } else { 0 };
    Some(uniform_access(warp))
}

pub fn poison_scalar(registers: &mut LaneRegisters, index: u16) {
    let word = usize::from(index / 64);
    if word < registers.written.len() {
        registers.written[word] &= !(1u64 << (index % 64));
    }
}

pub fn poison_predicate(registers: &mut LaneRegisters, index: u16) {
    let word = usize::from(index / 64);
    if word < registers.pdef.len() {
        registers.pdef[word] &= !(1u64 << (index % 64));
    }
}

pub fn poison_uniform(registers: &mut WarpRegisters, warp: u32, index: u16) -> Option<Access> {
    if index >= 255 {
        return None;
    }
    registers.ur_written[usize::from(index / 64)] &= !(1u64 << (index % 64));
    Some(uniform_access(warp))
}

pub fn poison_uniform_predicate(
    registers: &mut WarpRegisters,
    warp: u32,
    index: u16,
) -> Option<Access> {
    assert!(
        index < 8,
        "uniform predicate index exceeds its register file"
    );
    registers.up_def &= !(1u8 << index);
    Some(uniform_access(warp))
}

pub fn fill_scalar_if_unwritten(registers: &mut LaneRegisters, index: usize, value: u64) {
    let word = index / 64;
    if word >= registers.written.len() || registers.written[word] >> (index % 64) & 1 == 0 {
        registers.regs[index] = value;
    }
}

pub fn fill_uniform_if_unwritten(registers: &mut WarpRegisters, index: usize, value: u64) {
    let word = index / 64;
    if word >= registers.ur_written.len() || registers.ur_written[word] >> (index % 64) & 1 == 0 {
        registers.ur[index] = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lane() -> LaneRegisters {
        LaneRegisters {
            regs: vec![0; 256],
            preds: vec![0; 2],
            pdef: vec![0; 2],
            written: vec![0; 4],
            vund: Vec::new(),
        }
    }

    fn warp() -> WarpRegisters {
        WarpRegisters {
            ur: vec![0; 256],
            ur_written: [0; 4],
            up: 0,
            up_def: 0,
        }
    }

    #[test]
    fn virtual_updates_replace_and_remove_partial_definedness() {
        let mut registers = lane();
        for (index, mask) in [(65, 1), (2, 4), (64, 8), (65, 16)] {
            write_virtual(&mut registers, index, 123, mask);
        }
        assert_eq!(registers.vund, [(2, 4), (64, 8), (65, 16)]);
        write_virtual(&mut registers, 64, 456, u64::MAX);
        assert_eq!(registers.regs[64], 456);
        assert_eq!(registers.written[1] & 3, 2);
        assert_eq!(registers.vund, [(2, 4), (65, 16)]);
        write_virtual(&mut registers, 65, 789, 0);
        assert_eq!(registers.vund, [(2, 4)]);
        assert_eq!(registers.written[1] & 3, 2);
    }

    #[test]
    fn paired_words_keep_independent_definedness_across_bitset_words() {
        let mut registers = lane();
        write_scalar(
            &mut registers,
            WordWrite {
                index: 63,
                value: 0x1122_3344_5566_7788,
                slots: 2,
                mask: Some(0xffff_ffff_0000_0001),
                stale: false,
            },
        );
        assert_eq!(registers.regs[63], 0x0000_0001_5566_7788);
        assert_eq!(registers.regs[64], 0x1122_3344);
        assert_eq!(registers.written[0], 1 << 63);
        assert_eq!(registers.written[1], 0);
        assert_eq!(registers.regs[62], 0);
        assert_eq!(registers.regs[65], 0);
    }

    #[test]
    fn shared_updates_return_the_actual_warp_resource_in_both_fold_modes() {
        let mut registers = warp();
        let word = write_uniform(
            &mut registers,
            9,
            WordWrite {
                index: 254,
                value: 0x1122_3344_5566_7788,
                slots: 2,
                mask: None,
                stale: false,
            },
        );
        let predicate = write_uniform_predicate(
            &mut registers,
            9,
            PredicateWrite {
                index: 3,
                value: true,
                defined: false,
            },
        );
        assert_eq!(word, Some(uniform_access(9)));
        assert_eq!(predicate, Some(uniform_access(9)));
        assert_eq!(registers.ur[254], 0x5566_7788);
        assert_eq!(registers.ur[255], 0);
        assert_eq!(registers.ur_written[3], 1 << 62);
        assert_eq!(registers.up, 8);
        assert_eq!(registers.up_def, 0);
        let events = [Effect::Release(42), uniform_effect(9), uniform_effect(9)];
        for mode in [effects::FoldMode::Full, effects::FoldMode::SyncOnly] {
            let summary = effects::fold(mode, &events);
            assert_eq!(summary.writes, [(17, 9, 0, 1), (17, 9, 0, 1)]);
            assert_eq!(summary.releases, [42]);
        }
    }

    #[test]
    fn poisoning_preserves_values_and_partial_history() {
        let mut registers = lane();
        write_virtual(&mut registers, 4, 99, 1);
        write_predicate(&mut registers, 67, true, true);
        poison_scalar(&mut registers, 4);
        poison_predicate(&mut registers, 67);
        poison_scalar(&mut registers, u16::MAX);
        poison_predicate(&mut registers, u16::MAX);
        assert_eq!(registers.regs[4], 99);
        assert_eq!(registers.vund, [(4, 1)]);
        assert_eq!(registers.written[0], 0);
        assert_eq!(registers.preds[1], 8);
        assert_eq!(registers.pdef[1], 0);
        let mut registers = warp();
        let written = write_uniform_predicate(
            &mut registers,
            5,
            PredicateWrite {
                index: 2,
                value: true,
                defined: true,
            },
        );
        let poisoned = poison_uniform_predicate(&mut registers, 5, 2);
        assert_eq!(registers.up, 4);
        assert_eq!(registers.up_def, 0);
        assert_eq!(written, Some(uniform_access(5)));
        assert_eq!(poisoned, Some(uniform_access(5)));
    }

    #[test]
    fn constant_word_destinations_preserve_storage_and_return_none() {
        let mut lane = lane();
        let before_lane = lane.clone();
        let mut warp = warp();
        let before_warp = warp.clone();
        for index in [255, 256, u16::MAX] {
            let write = WordWrite {
                index,
                value: 123,
                slots: 2,
                mask: None,
                stale: false,
            };
            write_scalar(&mut lane, write);
            assert_eq!(write_uniform(&mut warp, 7, write), None);
            assert_eq!(poison_uniform(&mut warp, 7, index), None);
        }
        assert_eq!(lane, before_lane);
        assert_eq!(warp, before_warp);
    }

    #[test]
    fn fills_change_only_unwritten_bits_without_defining_them() {
        let mut registers = lane();
        write_virtual(&mut registers, 2, 99, 1);
        fill_scalar_if_unwritten(&mut registers, 2, 123);
        fill_scalar_if_unwritten(&mut registers, 3, 456);
        assert_eq!(registers.regs[2], 99);
        assert_eq!(registers.regs[3], 456);
        assert_eq!(registers.written[0], 4);
        assert_eq!(registers.vund, [(2, 1)]);
        let mut registers = warp();
        let access = write_uniform(
            &mut registers,
            3,
            WordWrite {
                index: 2,
                value: 99,
                slots: 1,
                mask: None,
                stale: false,
            },
        );
        fill_uniform_if_unwritten(&mut registers, 2, 123);
        fill_uniform_if_unwritten(&mut registers, 3, 456);
        assert_eq!(registers.ur[2], 99);
        assert_eq!(registers.ur[3], 456);
        assert_eq!(registers.ur_written[0], 4);
        assert_eq!(access, Some(uniform_access(3)));
    }
}
