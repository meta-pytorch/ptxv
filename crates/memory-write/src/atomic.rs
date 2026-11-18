/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/// Numeric histories to forget and numeric written markers for one atomic effect.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtomicFootprint {
    pub history_start: u64,
    pub history_end: u128,
    pub marker_start: u64,
    pub marker_end: u128,
    pub marker_stride: u64,
}

/// A successful write invalidates every intersecting four-byte history. A failed
/// CAS forgets only the numeric history at its address, without changing bytes.
pub fn atomic_footprint(offset: u64, bytes: u8, did_write: bool) -> AtomicFootprint {
    if !did_write {
        return AtomicFootprint {
            history_start: offset,
            history_end: offset as u128 + 1,
            marker_start: offset,
            marker_end: offset as u128 + 1,
            marker_stride: 1,
        };
    }
    if bytes == 0 {
        return AtomicFootprint {
            history_start: offset,
            history_end: offset as u128,
            marker_start: offset,
            marker_end: offset as u128,
            marker_stride: 4,
        };
    }
    let end = offset as u128 + bytes as u128;
    let end = if end > (1u128 << 64) {
        1u128 << 64
    } else {
        end
    };
    AtomicFootprint {
        history_start: if offset < 3 { 0 } else { offset - 3 },
        history_end: end,
        marker_start: offset / 4 * 4,
        marker_end: end,
        marker_stride: 4,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn histories(effect: AtomicFootprint) -> Vec<u64> {
        (effect.history_start as u128..effect.history_end)
            .map(|offset| offset as u64)
            .collect()
    }

    fn markers(effect: AtomicFootprint) -> Vec<u64> {
        (effect.marker_start as u128..effect.marker_end)
            .step_by(effect.marker_stride as usize)
            .map(|offset| offset as u64)
            .collect()
    }

    #[test]
    fn wide_writes_forget_upper_and_intersecting_histories() {
        let effect = atomic_footprint(8, 8, true);
        assert_eq!(histories(effect), (5..16).collect::<Vec<_>>());
        assert_eq!(markers(effect), vec![8, 12]);
        assert_eq!(markers(atomic_footprint(3, 8, true)), vec![0, 4, 8]);
        assert_eq!(
            histories(atomic_footprint(1, 4, true)),
            (0..5).collect::<Vec<_>>()
        );
    }

    #[test]
    fn failed_cas_forgets_only_the_exact_numeric_key() {
        let effect = atomic_footprint(3, 8, false);
        assert_eq!(histories(effect), vec![3]);
        assert_eq!(markers(effect), vec![3]);
    }

    #[test]
    fn empty_and_terminal_spans_do_not_wrap() {
        let empty = atomic_footprint(u64::MAX, 0, true);
        assert!(histories(empty).is_empty());
        assert!(markers(empty).is_empty());
        let terminal = atomic_footprint(u64::MAX, 255, true);
        assert_eq!(
            histories(terminal),
            (u64::MAX - 3..=u64::MAX).collect::<Vec<_>>()
        );
        assert_eq!(markers(terminal), vec![u64::MAX - 3]);
        let failed = atomic_footprint(u64::MAX, 8, false);
        assert_eq!(histories(failed), vec![u64::MAX]);
        assert_eq!(markers(failed), vec![u64::MAX]);
    }
}
