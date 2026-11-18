/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::atomic_components::Component;
use crate::atomic_components::Plan;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Returns {
    pub low: u16,
    pub high: u16,
    pub low_undefined: bool,
    pub high_undefined: bool,
}

pub fn plan() -> Plan {
    Plan {
        first: true,
        second: true,
        third: false,
        fourth: false,
    }
}

pub fn offset(component: Component) -> Option<u8> {
    match component {
        Component::First => Some(0),
        Component::Second => Some(2),
        _ => None,
    }
}

pub fn begin_component(
    input: crate::atomic_components::FrozenOperands,
    plan: Plan,
    component: Component,
) -> Option<crate::atomic_components::Step> {
    let offset = match offset(component) {
        Some(offset) => offset,
        None => return None,
    };
    match crate::atomic_components::finish(plan, component) {
        None => None,
        Some(remaining) => Some(crate::atomic_components::Step {
            address: input.address.wrapping_add(offset as u64),
            source: crate::atomic_components::frozen_operand(input, component),
            remaining,
        }),
    }
}

pub fn operand(word: u32, component: Component) -> Option<u16> {
    match component {
        Component::First => Some(word as u16),
        Component::Second => Some((word / 65536) as u16),
        _ => None,
    }
}

pub fn operand_defined(undefined_mask: u32, component: Component) -> Option<bool> {
    match operand(undefined_mask, component) {
        Some(mask) => Some(mask == 0),
        None => None,
    }
}

pub fn ticket_access_eligible(packed_bf16: bool, bytes: u8) -> bool {
    !packed_bf16 || bytes != 4
}

pub fn assemble(low: u16, high: u16) -> u32 {
    low as u32 + (high as u32) * 65536
}

pub fn empty_returns() -> Returns {
    Returns {
        low: 0,
        high: 0,
        low_undefined: false,
        high_undefined: false,
    }
}

pub fn record(
    mut returns: Returns,
    component: Component,
    old: u16,
    undefined: bool,
) -> Option<Returns> {
    match component {
        Component::First => {
            returns.low = old;
            returns.low_undefined = undefined;
        }
        Component::Second => {
            returns.high = old;
            returns.high_undefined = undefined;
        }
        _ => return None,
    }
    Some(returns)
}

pub fn returned(returns: Returns) -> u32 {
    assemble(returns.low, returns.high)
}

pub fn undefined_mask(returns: Returns) -> u32 {
    assemble(
        if returns.low_undefined { u16::MAX } else { 0 },
        if returns.high_undefined { u16::MAX } else { 0 },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_half_operands_and_assembly_preserve_all_bits() {
        for word in [
            0,
            1,
            65535,
            65536,
            0x41203f80,
            0x41a04000,
            0x80008000,
            u32::MAX,
        ] {
            let low = operand(word, Component::First).unwrap();
            let high = operand(word, Component::Second).unwrap();
            assert_eq!(assemble(low, high), word);
            assert_eq!(low, word as u16);
            assert_eq!(high, (word >> 16) as u16);
        }
        assert_eq!(offset(Component::First), Some(0));
        assert_eq!(offset(Component::Second), Some(2));
        for unsupported in [Component::Third, Component::Fourth] {
            assert_eq!(offset(unsupported), None);
            assert_eq!(operand(0, unsupported), None);
            assert_eq!(record(empty_returns(), unsupported, 1, false), None);
        }
    }

    #[test]
    fn packed_transitions_use_frozen_halves_and_wrapping_two_byte_addresses() {
        use crate::atomic_components::FrozenOperands;
        let first = crate::symbolic::prepare(1, None, false, true);
        let second = crate::symbolic::prepare(2, None, true, false);
        for address in [0, 4, u64::MAX - 1, u64::MAX] {
            let input = FrozenOperands {
                address,
                first,
                second,
                third: first,
                fourth: second,
            };
            for (component, offset, source) in
                [(Component::First, 0, first), (Component::Second, 2, second)]
            {
                let step = begin_component(input, plan(), component).unwrap();
                assert_eq!(step.address, address.wrapping_add(offset));
                assert_eq!(step.source, source);
                assert!(!crate::atomic_components::pending(
                    step.remaining,
                    component
                ));
                assert_eq!(begin_component(input, step.remaining, component), None);
            }
            assert_eq!(begin_component(input, plan(), Component::Third), None);
            assert_eq!(begin_component(input, plan(), Component::Fourth), None);
        }
    }

    #[test]
    fn definedness_and_ticket_eligibility_follow_the_selected_half() {
        for mask in [0, 1, 65535, 65536, 0xffff0000, u32::MAX] {
            assert_eq!(
                operand_defined(mask, Component::First),
                Some(mask & 65535 == 0)
            );
            assert_eq!(
                operand_defined(mask, Component::Second),
                Some(mask >> 16 == 0)
            );
        }
        assert_eq!(operand_defined(0, Component::Third), None);
        assert_eq!(operand_defined(0, Component::Fourth), None);
        for bytes in 0..=u8::MAX {
            assert!(ticket_access_eligible(false, bytes));
            assert_eq!(ticket_access_eligible(true, bytes), bytes != 4);
        }
    }

    #[test]
    fn independent_return_halves_preserve_each_others_bits_and_definedness() {
        for (low, high) in [(0, 0), (1, 2), (0x8000, 0xffff), (u16::MAX, u16::MAX)] {
            for low_undefined in [false, true] {
                for high_undefined in [false, true] {
                    let a = record(
                        record(empty_returns(), Component::First, low, low_undefined).unwrap(),
                        Component::Second,
                        high,
                        high_undefined,
                    )
                    .unwrap();
                    let b = record(
                        record(empty_returns(), Component::Second, high, high_undefined).unwrap(),
                        Component::First,
                        low,
                        low_undefined,
                    )
                    .unwrap();
                    assert_eq!(a, b);
                    assert_eq!(returned(a), low as u32 | ((high as u32) << 16));
                    assert_eq!(
                        undefined_mask(a),
                        if low_undefined { 0xffff } else { 0 }
                            | if high_undefined { 0xffff0000 } else { 0 }
                    );
                }
            }
        }
    }
}
