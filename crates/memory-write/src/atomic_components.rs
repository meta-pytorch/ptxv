/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Component {
    First,
    Second,
    Third,
    Fourth,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Plan {
    pub first: bool,
    pub second: bool,
    pub third: bool,
    pub fourth: bool,
}

pub fn vector_plan(bytes: u8) -> Option<Plan> {
    if bytes == 8 {
        Some(Plan {
            first: true,
            second: true,
            third: false,
            fourth: false,
        })
    } else if bytes == 16 {
        Some(Plan {
            first: true,
            second: true,
            third: true,
            fourth: true,
        })
    } else {
        None
    }
}

pub fn pending(plan: Plan, component: Component) -> bool {
    match component {
        Component::First => plan.first,
        Component::Second => plan.second,
        Component::Third => plan.third,
        Component::Fourth => plan.fourth,
    }
}

pub fn finish(mut plan: Plan, component: Component) -> Option<Plan> {
    if !pending(plan, component) {
        return None;
    }
    match component {
        Component::First => plan.first = false,
        Component::Second => plan.second = false,
        Component::Third => plan.third = false,
        Component::Fourth => plan.fourth = false,
    }
    Some(plan)
}

pub fn retired(plan: Plan) -> bool {
    !(plan.first || plan.second || plan.third || plan.fourth)
}

pub fn byte_offset(component: Component) -> u8 {
    match component {
        Component::First => 0,
        Component::Second => 4,
        Component::Third => 8,
        Component::Fourth => 12,
    }
}

pub fn ticket_access_eligible(float_add: bool, bytes: u8) -> bool {
    !(float_add && (bytes == 8 || bytes == 16))
}

pub fn choices(plan: Plan) -> Vec<Component> {
    let mut result = Vec::with_capacity(4);
    if plan.first {
        result.push(Component::First);
    }
    if plan.second {
        result.push(Component::Second);
    }
    if plan.third {
        result.push(Component::Third);
    }
    if plan.fourth {
        result.push(Component::Fourth);
    }
    result
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct FrozenOperands {
    pub address: u64,
    pub first: crate::symbolic::Metadata,
    pub second: crate::symbolic::Metadata,
    pub third: crate::symbolic::Metadata,
    pub fourth: crate::symbolic::Metadata,
}

pub fn frozen_operand(input: FrozenOperands, component: Component) -> crate::symbolic::Metadata {
    match component {
        Component::First => input.first,
        Component::Second => input.second,
        Component::Third => input.third,
        Component::Fourth => input.fourth,
    }
}

pub fn frozen_address(input: FrozenOperands, component: Component) -> u64 {
    input.address.wrapping_add(byte_offset(component) as u64)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Step {
    pub address: u64,
    pub source: crate::symbolic::Metadata,
    pub remaining: Plan,
}

pub fn begin_component(input: FrozenOperands, plan: Plan, component: Component) -> Option<Step> {
    match finish(plan, component) {
        None => None,
        Some(remaining) => Some(Step {
            address: frozen_address(input, component),
            source: frozen_operand(input, component),
            remaining,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const COMPONENTS: [Component; 4] = [
        Component::First,
        Component::Second,
        Component::Third,
        Component::Fourth,
    ];

    #[test]
    fn every_enabled_component_is_an_independent_retirement_choice() {
        for mask in 0..16 {
            let plan = Plan {
                first: mask & 1 != 0,
                second: mask & 2 != 0,
                third: mask & 4 != 0,
                fourth: mask & 8 != 0,
            };
            assert_eq!(retired(plan), mask == 0);
            for component in COMPONENTS {
                assert_eq!(choices(plan).contains(&component), pending(plan, component));
                let next = finish(plan, component);
                assert_eq!(next.is_some(), pending(plan, component));
                if let Some(next) = next {
                    for other in COMPONENTS {
                        assert_eq!(
                            pending(next, other),
                            pending(plan, other) && other != component
                        );
                    }
                    assert_eq!(finish(next, component), None);
                }
            }
        }
    }

    #[test]
    fn vector_widths_and_component_addresses_are_exact() {
        for bytes in 0..=u8::MAX {
            let plan = vector_plan(bytes);
            assert_eq!(plan.is_some(), matches!(bytes, 8 | 16));
            assert_eq!(
                ticket_access_eligible(true, bytes),
                !matches!(bytes, 8 | 16)
            );
            assert!(ticket_access_eligible(false, bytes));
            if let Some(plan) = plan {
                for (index, component) in COMPONENTS.into_iter().enumerate() {
                    assert_eq!(byte_offset(component) as usize, 4 * index);
                    assert_eq!(pending(plan, component), 4 * index < bytes as usize);
                }
            }
        }
    }
    #[test]
    fn selected_component_keeps_frozen_values_and_wrapping_addresses() {
        let operands = [11, 22, 33, 44]
            .map(|bits| crate::symbolic::prepare(bits, Some(bits), bits % 2 != 0, true));
        for address in [0, u64::MAX - 15, u64::MAX - 3, u64::MAX] {
            let input = FrozenOperands {
                address,
                first: operands[0],
                second: operands[1],
                third: operands[2],
                fourth: operands[3],
            };
            let plan = vector_plan(16).unwrap();
            for (index, component) in COMPONENTS.into_iter().enumerate() {
                let selected = begin_component(input, plan, component).unwrap();
                assert_eq!(selected.source, operands[index]);
                assert_eq!(selected.address, address.wrapping_add(4 * index as u64));
                assert_eq!(begin_component(input, selected.remaining, component), None);
            }
        }
    }
}
