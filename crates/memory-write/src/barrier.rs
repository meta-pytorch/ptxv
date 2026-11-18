/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::reduction::arrival_pending;
use crate::reduction::arrival_writes_pending;
use crate::reduction::phase_completes;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Counters {
    pub expected: u32,
    pub pending: u32,
    pub transactions: i64,
    pub phase: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TransactionDelta {
    Add(i64),
    Subtract(i64),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fault {
    TransactionOverflow,
    OverArrival,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Transition {
    /// State visible to the release/clock publication adapter before completion.
    pub intermediate: Counters,
    pub after: Counters,
    pub completed: bool,
    pub fault: Option<Fault>,
}

fn rejected(state: Counters, fault: Fault) -> Transition {
    Transition {
        intermediate: state,
        after: state,
        completed: false,
        fault: Some(fault),
    }
}

pub fn transition(mut state: Counters, delta: TransactionDelta, arrivals: u32) -> Transition {
    let difference = match delta {
        TransactionDelta::Add(value) => value as i128,
        TransactionDelta::Subtract(value) => -(value as i128),
    };
    let transactions = state.transactions as i128 + difference;
    if transactions < i64::MIN as i128 || transactions > i64::MAX as i128 {
        return rejected(state, Fault::TransactionOverflow);
    }
    state.transactions = transactions as i64;
    let Some(pending) = arrival_pending(state.pending, arrivals) else {
        return rejected(state, Fault::OverArrival);
    };
    state.pending = pending;
    let completed = phase_completes(state.pending, state.transactions == 0);
    let after = if completed {
        Counters {
            pending: state.expected,
            phase: state.phase.wrapping_add(1),
            ..state
        }
    } else {
        state
    };
    Transition {
        intermediate: state,
        after,
        completed,
        fault: None,
    }
}

pub fn writes_phase_membership(delta: TransactionDelta, arrivals: u32, completed: bool) -> bool {
    arrival_writes_pending(arrivals)
        || completed
        || match delta {
            TransactionDelta::Add(value) | TransactionDelta::Subtract(value) => value != 0,
        }
}

pub fn batch_writes_phase_membership(
    delta: TransactionDelta,
    pending_before: u32,
    arrivals: u32,
    completed: bool,
    faulted: bool,
    prior_same_phase: bool,
) -> bool {
    faulted
        || prior_same_phase
        || arrivals > 1
        || (arrivals != 0 && pending_before <= 1)
        || completed
        || match delta {
            TransactionDelta::Add(value) | TransactionDelta::Subtract(value) => value != 0,
        }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArrivalBatch {
    pending: u32,
    arrivals: u32,
    valid: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArrivalStep {
    pub delta: TransactionDelta,
    pub pending_before: u32,
    pub arrivals: u32,
    pub completed: bool,
    pub faulted: bool,
}

pub fn arrival_batch(pending: u32) -> ArrivalBatch {
    ArrivalBatch {
        pending,
        arrivals: 0,
        valid: true,
    }
}

pub fn invalidate_arrival_batch(mut batch: ArrivalBatch) -> ArrivalBatch {
    batch.valid = false;
    batch
}

pub fn record_arrival_batch(mut batch: ArrivalBatch, step: ArrivalStep) -> ArrivalBatch {
    if !batch.valid {
        return batch;
    }
    if step.completed || step.faulted || step.arrivals == 0 {
        return invalidate_arrival_batch(batch);
    }
    let plain = match step.delta {
        TransactionDelta::Add(value) | TransactionDelta::Subtract(value) => value == 0,
    };
    if !plain {
        return invalidate_arrival_batch(batch);
    }
    let Some(pending) = batch.pending.checked_sub(batch.arrivals) else {
        return invalidate_arrival_batch(batch);
    };
    if pending != step.pending_before {
        return invalidate_arrival_batch(batch);
    }
    let Some(arrivals) = batch.arrivals.checked_add(step.arrivals) else {
        return invalidate_arrival_batch(batch);
    };
    batch.arrivals = arrivals;
    batch
}

pub fn arrival_batch_writes(batch: ArrivalBatch) -> bool {
    // ASSUME(E-CTA-INDEP): only audited plain batches small enough to remain in the
    // same phase in either pairwise order may share a read-only membership key.
    !batch.valid || batch.arrivals == 0 || batch.arrivals > batch.pending / 2
}

pub fn phase_key(offset: u32, phase: u32) -> u64 {
    offset as u64 * 0x10_0000 + (phase % 0x10_0000) as u64
}

pub fn next_arrival_key(offset: u32, outcome: Transition) -> u64 {
    phase_key(offset, outcome.after.phase)
}

pub fn publication_parity(outcome: Transition) -> Option<u32> {
    if outcome.completed {
        Some(outcome.intermediate.phase % 2)
    } else {
        None
    }
}

pub fn poll_release<T>(phase: u32, parity: u32, completed: &[T; 2]) -> Option<&T> {
    let parity = parity % 2;
    if phase % 2 == parity {
        None
    } else if parity == 0 {
        Some(&completed[0])
    } else {
        Some(&completed[1])
    }
}

pub fn wait_can_progress<T, E>(
    result: Result<Option<T>, E>,
    reported: Option<u32>,
    pc: u32,
) -> bool {
    match result {
        Ok(None) => false,
        Ok(Some(_)) => true,
        Err(_) => match reported {
            Some(previous) => previous != pc,
            None => true,
        },
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WaitMode {
    Blocking,
    NonBlocking,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WaitPolicy {
    pub progress: bool,
    pub force_failed: bool,
}

pub fn wait_mode(is_peek: bool, blocking_override: bool) -> WaitMode {
    if is_peek && !blocking_override {
        WaitMode::NonBlocking
    } else {
        WaitMode::Blocking
    }
}

pub fn wait_policy<T, E>(
    result: Result<Option<T>, E>,
    mode: WaitMode,
    reported: Option<u32>,
    pc: u32,
    force_requested: bool,
) -> WaitPolicy {
    let nonblocking = mode == WaitMode::NonBlocking;
    let valid = matches!(result, Ok(_));
    let progress = match result {
        Ok(None) => nonblocking,
        other => wait_can_progress(other, reported, pc),
    };
    WaitPolicy {
        progress,
        force_failed: force_requested && nonblocking && valid,
    }
}

pub fn mark_wait_error(reported: &mut Option<u32>, pc: u32) {
    *reported = Some(pc);
}

pub fn clear_wait_error(reported: &mut Option<u32>) {
    *reported = None;
}

pub fn publish_release<T>(current: &mut T, completed: &mut [T; 2], empty: T, outcome: Transition) {
    match publication_parity(outcome) {
        Some(0) => completed[0] = std::mem::replace(current, empty),
        Some(_) => completed[1] = std::mem::replace(current, empty),
        None => {}
    }
}

pub fn begin_update(
    expected: u32,
    pending: &mut u32,
    transactions: &mut i64,
    phase: u32,
    delta: TransactionDelta,
    arrivals: u32,
) -> Transition {
    let outcome = transition(
        Counters {
            expected,
            pending: *pending,
            transactions: *transactions,
            phase,
        },
        delta,
        arrivals,
    );
    *pending = outcome.intermediate.pending;
    *transactions = outcome.intermediate.transactions;
    outcome
}

pub fn finish_update(
    pending: &mut u32,
    transactions: &mut i64,
    phase: &mut u32,
    outcome: Transition,
) {
    *pending = outcome.after.pending;
    *transactions = outcome.after.transactions;
    *phase = outcome.after.phase;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn initial(pending: u32, transactions: i64, phase: u32) -> Counters {
        Counters {
            expected: 3,
            pending,
            transactions,
            phase,
        }
    }

    #[test]
    fn polling_selects_only_the_observed_completed_parity() {
        let completed = [vec![3], vec![5]];
        for phase in [0, 1, 2, u32::MAX] {
            for parity in 0..4 {
                let result = poll_release(phase, parity, &completed);
                if phase % 2 == parity % 2 {
                    assert!(result.is_none());
                } else {
                    assert_eq!(result, Some(&completed[parity as usize % 2]));
                }
            }
        }
    }

    #[test]
    fn only_a_valid_blocked_wait_cannot_progress() {
        assert!(wait_can_progress::<(), _>(Err("uninitialized"), None, 3));
        assert!(wait_can_progress::<_, ()>(Ok(Some(())), Some(3), 3));
        assert!(!wait_can_progress::<(), ()>(Ok(None), Some(3), 3));
        let mut reported = None;
        mark_wait_error(&mut reported, 3);
        assert!(!wait_can_progress::<(), _>(
            Err("uninitialized"),
            reported,
            3
        ));
        assert!(wait_can_progress::<(), _>(Err("other wait"), reported, 4));
        clear_wait_error(&mut reported);
        assert!(wait_can_progress::<(), _>(
            Err("uninitialized"),
            reported,
            3
        ));
    }

    #[test]
    fn phase_keys_preserve_the_offset_and_wrap_the_phase_bits() {
        for offset in [0, 1, 1024, u32::MAX] {
            for phase in [0, 1, 0xf_ffff, 0x10_0000, u32::MAX] {
                let key = phase_key(offset, phase);
                assert_eq!(key / 0x10_0000, offset as u64);
                assert_eq!(key % 0x10_0000, (phase % 0x10_0000) as u64);
            }
        }
    }

    #[test]
    fn completion_dependency_uses_the_actual_next_phase_key() {
        for phase in [0, 1, 0xf_ffff, u32::MAX] {
            let outcome = transition(initial(1, 0, phase), TransactionDelta::Add(0), 1);
            assert_eq!(
                next_arrival_key(1024, outcome),
                phase_key(1024, phase.wrapping_add(1))
            );
        }
    }

    #[test]
    fn publication_moves_the_release_and_frames_the_other_slot() {
        for phase in [0, 1, u32::MAX] {
            let mut current = vec![3, 5];
            let mut completed = [vec![7], vec![11]];
            let before = completed.clone();
            let outcome = transition(initial(1, 0, phase), TransactionDelta::Add(0), 1);
            publish_release(&mut current, &mut completed, vec![], outcome);
            assert!(current.is_empty());
            assert_eq!(completed[phase as usize % 2], [3, 5]);
            assert_eq!(
                completed[(phase as usize + 1) % 2],
                before[(phase as usize + 1) % 2]
            );
        }
    }

    #[test]
    fn noncompletion_and_fault_leave_release_fields_unchanged() {
        for (pending, transactions, arrivals) in [(2, 0, 1), (1, 1, 1), (1, 0, 2)] {
            let mut current = vec![3, 5];
            let mut completed = [vec![7], vec![11]];
            let outcome = transition(
                initial(pending, transactions, 9),
                TransactionDelta::Add(0),
                arrivals,
            );
            publish_release(&mut current, &mut completed, vec![], outcome);
            assert_eq!(current, [3, 5]);
            assert_eq!(completed, [vec![7], vec![11]]);
        }
    }

    #[test]
    fn live_fields_expose_intermediate_counters_until_publication_finishes() {
        let (mut pending, mut transactions, mut phase) = (3, 0, u32::MAX);
        let outcome = begin_update(
            3,
            &mut pending,
            &mut transactions,
            phase,
            TransactionDelta::Add(0),
            3,
        );
        assert_eq!((pending, transactions, phase), (0, 0, u32::MAX));
        assert_eq!(publication_parity(outcome), Some(1));
        finish_update(&mut pending, &mut transactions, &mut phase, outcome);
        assert_eq!((pending, transactions, phase), (3, 0, 0));
    }

    #[test]
    fn live_field_error_keeps_its_preceding_transaction_update() {
        let (mut pending, mut transactions) = (2, 16);
        let outcome = begin_update(
            3,
            &mut pending,
            &mut transactions,
            8,
            TransactionDelta::Subtract(4),
            3,
        );
        assert_eq!(outcome.fault, Some(Fault::OverArrival));
        assert_eq!((pending, transactions), (2, 12));
        assert_eq!(publication_parity(outcome), None);
    }

    #[test]
    fn phase_membership_covers_both_signs_of_transaction_changes() {
        for value in [-1, 1] {
            for delta in [
                TransactionDelta::Add(value),
                TransactionDelta::Subtract(value),
            ] {
                assert!(writes_phase_membership(delta, 0, false));
            }
        }
        for delta in [TransactionDelta::Add(0), TransactionDelta::Subtract(0)] {
            assert!(!writes_phase_membership(delta, 0, false));
            assert!(writes_phase_membership(delta, 1, false));
            assert!(writes_phase_membership(delta, 0, true));
        }
    }

    #[test]
    fn exact_arrival_completes_and_resets_only_after_publication() {
        let result = transition(initial(3, 0, 8), TransactionDelta::Add(0), 3);
        assert_eq!(result.intermediate, initial(0, 0, 8));
        assert_eq!(result.after, initial(3, 0, 9));
        assert!(result.completed);
        assert_eq!(result.fault, None);
    }

    #[test]
    fn outstanding_transactions_defer_completion_until_their_last_byte() {
        let arrival = transition(initial(3, 16, 8), TransactionDelta::Add(0), 3);
        assert_eq!(arrival.after, initial(0, 16, 8));
        assert!(!arrival.completed);
        let completion = transition(arrival.after, TransactionDelta::Subtract(16), 0);
        assert_eq!(completion.intermediate, initial(0, 0, 8));
        assert_eq!(completion.after, initial(3, 0, 9));
        assert!(completion.completed);
    }

    #[test]
    fn negative_transaction_debt_also_defers_completion() {
        let result = transition(initial(0, -1, 8), TransactionDelta::Add(0), 0);
        assert_eq!(result.after, initial(0, -1, 8));
        assert!(!result.completed);
        assert_eq!(result.fault, None);
        let completion = transition(result.after, TransactionDelta::Add(1), 0);
        assert_eq!(completion.after, initial(3, 0, 9));
        assert!(completion.completed);
    }

    #[test]
    fn overarrival_keeps_the_prior_transaction_update_and_does_not_publish() {
        let result = transition(initial(2, 16, 8), TransactionDelta::Subtract(4), 3);
        assert_eq!(result.fault, Some(Fault::OverArrival));
        assert_eq!(result.intermediate, initial(2, 12, 8));
        assert_eq!(result.after, result.intermediate);
        assert!(!result.completed);
    }

    #[test]
    fn signed_transaction_overflow_is_a_framed_fault() {
        for (value, delta) in [
            (i64::MAX, TransactionDelta::Add(1)),
            (i64::MIN, TransactionDelta::Subtract(1)),
            (0, TransactionDelta::Subtract(i64::MIN)),
        ] {
            let before = initial(0, value, 8);
            let result = transition(before, delta, 0);
            assert_eq!(result.fault, Some(Fault::TransactionOverflow));
            assert_eq!(result.intermediate, before);
            assert_eq!(result.after, before);
            assert!(!result.completed);
        }
    }

    #[test]
    fn phase_wrap_preserves_the_parity_toggle() {
        let result = transition(initial(1, 0, u32::MAX), TransactionDelta::Add(0), 1);
        assert_eq!(result.after.phase, 0);
        assert_eq!(publication_parity(result), Some(1));
        assert_eq!(result.after.phase & 1, (u32::MAX & 1) ^ 1);
        assert_eq!(result.after.pending, 3);
        assert!(result.completed);
    }
}

#[cfg(test)]
mod wait_policy_tests {
    use super::*;

    #[test]
    fn wait_modes_preserve_pending_peeks_and_report_errors_once() {
        for is_peek in [false, true] {
            for blocking_override in [false, true] {
                let mode = wait_mode(is_peek, blocking_override);
                assert_eq!(mode == WaitMode::NonBlocking, is_peek && !blocking_override);
                for reported in [None, Some(6), Some(7)] {
                    for force_requested in [false, true] {
                        for result in [Ok(None), Ok(Some(3)), Err("invalid")] {
                            let policy = wait_policy(result, mode, reported, 7, force_requested);
                            let expected_progress = match result {
                                Ok(None) => mode == WaitMode::NonBlocking,
                                Ok(Some(_)) => true,
                                Err(_) => reported != Some(7),
                            };
                            assert_eq!(policy.progress, expected_progress);
                            assert_eq!(
                                policy.force_failed,
                                force_requested && mode == WaitMode::NonBlocking && result.is_ok()
                            );
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod batch_membership_tests {
    use super::Counters;
    use super::Fault;
    use super::TransactionDelta;
    use super::batch_writes_phase_membership;
    use super::transition;

    #[test]
    fn unit_arrival_read_requires_room_for_both_orders() {
        let state = Counters {
            expected: 2,
            pending: 1,
            transactions: 1,
            phase: 0,
        };
        let first = transition(state, TransactionDelta::Add(0), 1);
        assert_eq!(first.fault, None);
        assert!(!first.completed);
        assert_eq!(
            transition(first.after, TransactionDelta::Add(0), 1).fault,
            Some(Fault::OverArrival)
        );
        assert!(batch_writes_phase_membership(
            TransactionDelta::Add(0),
            state.pending,
            1,
            first.completed,
            false,
            false
        ));
        let roomy = Counters {
            pending: 2,
            ..state
        };
        let first = transition(roomy, TransactionDelta::Add(0), 1);
        let second = transition(first.after, TransactionDelta::Add(0), 1);
        assert_eq!(second.fault, None);
        assert!(!batch_writes_phase_membership(
            TransactionDelta::Add(0),
            roomy.pending,
            1,
            first.completed,
            false,
            false
        ));
    }

    #[test]
    fn multiple_unit_arrivals_in_one_action_remain_a_write() {
        let state = Counters {
            expected: 3,
            pending: 3,
            transactions: 0,
            phase: 0,
        };
        let first = transition(state, TransactionDelta::Add(0), 1);
        let second = transition(first.after, TransactionDelta::Add(0), 1);
        assert!(!first.completed && !second.completed);
        assert!(!batch_writes_phase_membership(
            TransactionDelta::Add(0),
            state.pending,
            1,
            false,
            false,
            false
        ));
        assert!(batch_writes_phase_membership(
            TransactionDelta::Add(0),
            first.after.pending,
            1,
            false,
            false,
            true
        ));
        assert!(batch_writes_phase_membership(
            TransactionDelta::Add(0),
            state.pending,
            2,
            false,
            false,
            false
        ));
    }

    #[test]
    fn phase_changes_transactions_and_rejections_remain_writes() {
        let state = Counters {
            expected: 2,
            pending: 1,
            transactions: 0,
            phase: 0,
        };
        let last = transition(state, TransactionDelta::Add(0), 1);
        assert!(last.completed);
        assert!(batch_writes_phase_membership(
            TransactionDelta::Add(0),
            state.pending,
            1,
            true,
            false,
            false
        ));
        for delta in [TransactionDelta::Add(1), TransactionDelta::Subtract(1)] {
            assert!(batch_writes_phase_membership(
                delta, 2, 0, false, false, false
            ));
        }
        assert!(batch_writes_phase_membership(
            TransactionDelta::Add(0),
            2,
            0,
            false,
            true,
            false
        ));
    }
}

#[cfg(test)]
mod arrival_batch_tests {
    use super::*;

    fn step(pending_before: u32, arrivals: u32) -> ArrivalStep {
        ArrivalStep {
            delta: TransactionDelta::Add(0),
            pending_before,
            arrivals,
            completed: false,
            faulted: false,
        }
    }

    #[test]
    fn grouped_arrivals_consume_one_outer_action_budget() {
        let mut batch = arrival_batch(128);
        for pending in (97..=128).rev() {
            batch = record_arrival_batch(batch, step(pending, 1));
        }
        assert_eq!(batch.arrivals, 32);
        assert!(!arrival_batch_writes(batch));
        assert!(!arrival_batch_writes(record_arrival_batch(
            arrival_batch(64),
            step(64, 32)
        )));
        assert!(arrival_batch_writes(record_arrival_batch(
            arrival_batch(32),
            step(32, 32)
        )));
    }

    #[test]
    fn two_individually_admitted_batches_cannot_over_arrive() {
        assert!(!arrival_batch_writes(record_arrival_batch(
            arrival_batch(4),
            step(4, 2)
        )));
        assert!(arrival_batch_writes(record_arrival_batch(
            arrival_batch(4),
            step(4, 3)
        )));
    }

    #[test]
    fn rejected_and_mixed_actions_remain_writes() {
        let mut rejected = step(128, 1);
        rejected.delta = TransactionDelta::Add(1);
        assert!(arrival_batch_writes(record_arrival_batch(
            arrival_batch(128),
            rejected
        )));
        rejected = step(128, 1);
        rejected.completed = true;
        assert!(arrival_batch_writes(record_arrival_batch(
            arrival_batch(128),
            rejected
        )));
        rejected = step(128, 1);
        rejected.faulted = true;
        assert!(arrival_batch_writes(record_arrival_batch(
            arrival_batch(128),
            rejected
        )));
        assert!(arrival_batch_writes(record_arrival_batch(
            arrival_batch(128),
            step(127, 1)
        )));
        assert!(arrival_batch_writes(record_arrival_batch(
            arrival_batch(128),
            step(128, 0)
        )));
        assert!(arrival_batch_writes(invalidate_arrival_batch(
            record_arrival_batch(arrival_batch(128), step(128, 1))
        )));
    }

    #[test]
    fn overflow_and_invalidity_are_sticky() {
        let batch = record_arrival_batch(arrival_batch(u32::MAX), step(u32::MAX, u32::MAX));
        let batch = record_arrival_batch(batch, step(0, 1));
        assert!(!batch.valid);
        assert!(arrival_batch_writes(batch));
        assert_eq!(record_arrival_batch(batch, step(0, 0)), batch);
    }
}
