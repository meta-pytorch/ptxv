/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

fn merge_successors(successors: &[usize], reach: &[u32], initial: u32) -> u32 {
    let mut merged = initial;
    let mut position = 0;
    while position < successors.len() {
        merged |= reach[successors[position]];
        position += 1;
    }
    merged
}

#[expect(
    clippy::ptr_arg,
    reason = "Aeneas tracks the owned vector returned by this mutable borrow"
)]
fn scan(successors: &[Vec<usize>], reach: &mut Vec<u32>) -> bool {
    let mut changed = false;
    let mut pc = reach.len();
    while pc > 0 {
        pc -= 1;
        let merged = merge_successors(&successors[pc], reach, reach[pc]);
        if merged != reach[pc] {
            reach[pc] = merged;
            changed = true;
        }
    }
    changed
}

pub fn propagate(successors: &[Vec<usize>], mut reach: Vec<u32>) -> Vec<u32> {
    let mut changed = true;
    while changed {
        changed = scan(successors, &mut reach);
    }
    reach
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reverse_order_cycle_requires_multiple_passes() {
        let successors = vec![vec![], vec![3, 0], vec![1], vec![2]];
        assert_eq!(propagate(&successors, vec![1, 0, 0, 0]), vec![1, 1, 1, 1]);
    }

    #[test]
    fn joins_preserve_all_barriers_and_unreachable_components() {
        let successors = vec![vec![1, 2], vec![3], vec![3], vec![], vec![4]];
        let seeds = vec![0, 2, 1 << 31, 1, 0];
        assert_eq!(
            propagate(&successors, seeds),
            vec![0x80000003, 3, 0x80000001, 1, 0]
        );
    }

    #[test]
    fn empty_graph_is_a_fixed_point() {
        assert!(propagate(&[], vec![]).is_empty());
    }
}
