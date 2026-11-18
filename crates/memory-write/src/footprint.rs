/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

pub type Range = (u8, u32, u64, u64);

pub fn shared_register_range(warp: u32) -> Range {
    (17, warp, 0, 1)
}

pub fn convergence_barrier_range(warp: u32, barrier: u8) -> Range {
    let start = u64::from(barrier);
    (19, warp, start, start + 1)
}

pub fn key_before(left: Range, right: Range) -> bool {
    left.0 < right.0 || (left.0 == right.0 && left.1 < right.1)
}

/// Both inputs are sorted by (space, owner, start, end), as in native normalization.
pub fn overlaps(left: &[Range], right: &[Range]) -> bool {
    let mut i = 0;
    let mut j = 0;
    while i < left.len() && j < right.len() {
        let x = left[i];
        let y = right[j];
        if key_before(x, y) {
            i += 1;
        } else if key_before(y, x) {
            j += 1;
        } else if x.3 <= y.2 {
            i += 1;
        } else if y.3 <= x.2 {
            j += 1;
        } else {
            return true;
        }
    }
    false
}

pub fn conflicts(
    left_all: bool,
    left_reads: &[Range],
    left_writes: &[Range],
    right_all: bool,
    right_reads: &[Range],
    right_writes: &[Range],
) -> bool {
    left_all
        || right_all
        || overlaps(left_writes, right_writes)
        || overlaps(left_writes, right_reads)
        || overlaps(right_writes, left_reads)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn oracle(left: &[Range], right: &[Range]) -> bool {
        left.iter().any(|x| {
            right
                .iter()
                .any(|y| x.0 == y.0 && x.1 == y.1 && x.3 > y.2 && y.3 > x.2)
        })
    }

    #[test]
    fn convergence_barriers_use_a_separate_per_warp_namespace() {
        for warp in [0, 1, u32::MAX] {
            for barrier in [0, 1, 15, u8::MAX] {
                let range = convergence_barrier_range(warp, barrier);
                assert_eq!(
                    range,
                    (19, warp, u64::from(barrier), u64::from(barrier) + 1)
                );
                assert!(overlaps(&[range], &[range]));
                assert!(!overlaps(&[range], &[shared_register_range(warp)]));
            }
        }
        assert!(!overlaps(
            &[convergence_barrier_range(0, 0)],
            &[convergence_barrier_range(1, 0)]
        ));
        assert!(!overlaps(
            &[convergence_barrier_range(0, 0)],
            &[convergence_barrier_range(0, 1)]
        ));
    }

    #[test]
    fn shared_registers_conflict_exactly_within_a_warp() {
        for warp in [0, 1, u32::MAX] {
            assert_eq!(shared_register_range(warp), (17, warp, 0, 1));
            assert!(overlaps(
                &[shared_register_range(warp)],
                &[shared_register_range(warp)]
            ));
        }
        assert!(!overlaps(
            &[shared_register_range(0)],
            &[shared_register_range(1)]
        ));
    }

    #[test]
    fn sweep_matches_pairwise_conflicts() {
        let mut seed = 29u64;
        for _ in 0..5000 {
            let mut ranges = || {
                let mut result = Vec::new();
                for _ in 0..12 {
                    seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let lo = (seed >> 16) % 100;
                    result.push((
                        (seed % 4) as u8,
                        ((seed >> 8) % 4) as u32,
                        lo,
                        lo + 1 + (seed >> 24) % 9,
                    ));
                }
                result.sort_unstable();
                result
            };
            let left = ranges();
            let right = ranges();
            assert_eq!(overlaps(&left, &right), oracle(&left, &right));
        }
        assert!(!overlaps(&[], &[(0, 0, 0, 1)]));
        assert!(!overlaps(&[(0, 0, 0, 1)], &[(0, 0, 1, 2)]));
        assert!(overlaps(
            &[(0, 0, 0, u64::MAX)],
            &[(0, 0, u64::MAX - 1, u64::MAX)]
        ));
    }
}
