/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/// The pending-count update, before transaction completion can advance the phase.
pub fn arrival_pending(pending: u32, arrivals: u32) -> Option<u32> {
    if pending < arrivals {
        None
    } else {
        Some(pending - arrivals)
    }
}

pub fn phase_completes(pending: u32, transaction_count_is_zero: bool) -> bool {
    pending == 0 && transaction_count_is_zero
}

pub fn arrival_writes_pending(arrivals: u32) -> bool {
    arrivals > 0
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationRequest {
    Disabled,
    Invalid,
    Valid { available: bool },
}

/// Invalid requests run so execution can report their diagnostic.
pub fn allocation_ready(request: AllocationRequest) -> bool {
    match request {
        AllocationRequest::Disabled | AllocationRequest::Invalid => true,
        AllocationRequest::Valid { available } => available,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counted_arrival_exposes_the_historical_order_difference() {
        assert_eq!(
            arrival_pending(3, 1).and_then(|p| arrival_pending(p, 3)),
            None
        );
        let completed = arrival_pending(3, 3).unwrap();
        assert!(phase_completes(completed, true));
        assert_eq!(arrival_pending(3, 1), Some(2));
        assert!(arrival_writes_pending(3));
        assert!(arrival_writes_pending(2));
        assert!(arrival_writes_pending(1));
    }

    #[test]
    fn zero_arrivals_and_pending_transaction_preserve_phase_boundaries() {
        assert_eq!(arrival_pending(0, 0), Some(0));
        assert_eq!(arrival_pending(0, 1), None);
        assert!(!arrival_writes_pending(0));
        assert!(!phase_completes(0, false));
        assert!(!phase_completes(1, true));
        assert!(phase_completes(0, true));
    }

    #[test]
    fn disabled_allocation_advances_without_capacity() {
        assert!(allocation_ready(AllocationRequest::Disabled));
        assert!(allocation_ready(AllocationRequest::Invalid));
        assert!(!allocation_ready(AllocationRequest::Valid {
            available: false
        }));
        assert!(allocation_ready(AllocationRequest::Valid {
            available: true
        }));
    }
}
