/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Frozen symbolic operands and conservative register metadata transfers.

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Metadata {
    pub captured: u32,
    pub history: Option<u32>,
    pub tainted: bool,
    pub defined: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MissingHistory {
    Constant,
    Opaque,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    And,
    Or,
    Xor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RegisterKey {
    pub thread: u32,
    pub file: u8,
    pub index: u16,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SymbolicError {
    NodeIdExhausted,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Control {
    pub enabled: bool,
    pub address_dependent: bool,
    pub guard_dependent: bool,
}

pub fn control(guard: Metadata, address_low: Metadata, address_high: Metadata) -> Control {
    Control {
        enabled: guard.captured != 0,
        address_dependent: address_low.tainted || address_high.tainted,
        guard_dependent: guard.tainted,
    }
}

pub fn register_key(thread: u32, file: u8, index: u16) -> RegisterKey {
    RegisterKey {
        thread,
        file,
        index,
    }
}

pub fn checked_node_id(index: usize) -> Result<u32, SymbolicError> {
    if index > u32::MAX as usize {
        Err(SymbolicError::NodeIdExhausted)
    } else {
        Ok(index as u32)
    }
}

pub fn classify_missing(input_derived: bool) -> MissingHistory {
    if input_derived {
        MissingHistory::Opaque
    } else {
        MissingHistory::Constant
    }
}

pub fn prepare(captured: u32, history: Option<u32>, tainted: bool, defined: bool) -> Metadata {
    Metadata {
        captured,
        history,
        tainted,
        defined,
    }
}

pub fn assign_metadata(
    enabled: bool,
    guard_dependent: bool,
    source: Metadata,
    old: Metadata,
) -> Metadata {
    let mut selected = if enabled { source } else { old };
    if guard_dependent {
        selected.history = None;
        selected.tainted = true;
        selected.defined = false;
    }
    selected
}

pub fn binary_bits(op: BinaryOp, left: u32, right: u32) -> u32 {
    match op {
        BinaryOp::Add => left.wrapping_add(right),
        BinaryOp::And => left & right,
        BinaryOp::Or => left | right,
        BinaryOp::Xor => left ^ right,
    }
}

pub fn binary(op: BinaryOp, left: Metadata, right: Metadata) -> Metadata {
    Metadata {
        captured: binary_bits(op, left.captured, right.captured),
        history: None,
        tainted: left.tainted || right.tainted,
        defined: left.defined && right.defined,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn captured_false_guard_and_high_address_dependencies_are_preserved() {
        let guard = prepare(0, None, true, true);
        let low = prepare(0, None, false, true);
        let high = prepare(0, None, true, true);
        assert_eq!(
            control(guard, low, high),
            Control {
                enabled: false,
                address_dependent: true,
                guard_dependent: true,
            }
        );
    }

    #[test]
    fn captured_zero_keeps_input_dependency() {
        let operand = prepare(0, None, true, true);
        assert_eq!(classify_missing(operand.tainted), MissingHistory::Opaque);
        let constant = prepare(0, None, false, true);
        assert_eq!(classify_missing(constant.tainted), MissingHistory::Constant);
        let result = binary(BinaryOp::Add, operand, constant);
        assert_eq!(result.captured, 0);
        assert!(result.tainted);
    }

    #[test]
    fn prepared_old_operand_survives_memory_overwrite() {
        let old = prepare(0, None, true, true);
        let replacement = prepare(0, None, false, true);
        let memory_after = assign_metadata(true, false, replacement, old);
        assert!(!memory_after.tainted);
        assert!(old.tainted);
        assert_eq!(classify_missing(old.tainted), MissingHistory::Opaque);
    }

    #[test]
    fn captured_false_dependent_guard_invalidates_metadata() {
        let old = prepare(0, Some(256), false, true);
        let source = prepare(1, Some(257), false, true);
        let result = assign_metadata(false, true, source, old);
        assert_eq!(result.captured, 0);
        assert_eq!(result.history, None);
        assert!(result.tainted);
        assert!(!result.defined);
    }

    #[test]
    fn predicate_namespaces_and_full_width_ids_remain_distinct() {
        assert_ne!(register_key(0, 3, 0), register_key(0, 4, 0));
        assert_eq!(checked_node_id(256), Ok(256));
        assert_ne!(checked_node_id(256), checked_node_id(0));
        assert_eq!(checked_node_id(u32::MAX as usize), Ok(u32::MAX));
        if usize::BITS > 32 {
            assert_eq!(
                checked_node_id(u32::MAX as usize + 1),
                Err(SymbolicError::NodeIdExhausted)
            );
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DestinationPlan {
    pub touched: bool,
    pub clear_history: bool,
    pub forget_half: bool,
    pub clear_bits: bool,
    pub active_taint: Option<bool>,
    pub force_taint: bool,
}

pub fn destination_plan(
    active: bool,
    guard_dependent: bool,
    hook_wrote: bool,
    source_tainted: bool,
) -> DestinationPlan {
    let touched = active || guard_dependent;
    DestinationPlan {
        touched,
        clear_history: touched && (guard_dependent || !hook_wrote),
        forget_half: guard_dependent,
        clear_bits: touched,
        active_taint: if active { Some(source_tainted) } else { None },
        force_taint: guard_dependent,
    }
}

#[cfg(test)]
mod destination_plan_tests {
    use super::*;

    #[test]
    fn captured_false_dependent_guard_discards_remembered_history() {
        let plan = destination_plan(false, true, true, false);
        assert!(plan.touched && plan.clear_history && plan.forget_half && plan.clear_bits);
        assert!(plan.force_taint);
        assert_eq!(plan.active_taint, None);
    }

    #[test]
    fn independent_guard_keeps_hook_history_and_unexecuted_registers() {
        let executed = destination_plan(true, false, true, false);
        assert!(!executed.clear_history);
        assert_eq!(executed.active_taint, Some(false));
        assert!(destination_plan(true, false, false, false).clear_history);
        let inactive = destination_plan(false, false, false, true);
        assert!(!inactive.clear_bits);
        assert!(!inactive.touched && !inactive.clear_history && !inactive.force_taint);
        assert_eq!(inactive.active_taint, None);
    }
}
