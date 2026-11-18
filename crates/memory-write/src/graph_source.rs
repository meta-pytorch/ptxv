/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::barrier_reach;
use crate::cache_graph::MayWriteGraph;

pub fn build(successors: Vec<Vec<usize>>, seeds: Vec<u32>) -> MayWriteGraph {
    let reach = barrier_reach::propagate(&successors, seeds);
    MayWriteGraph {
        reach,
        succ: successors,
        writes: Vec::new(),
        register_count: 0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn graph_construction_preserves_edges_and_owned_allocations() {
        let successors = vec![vec![1, 2], vec![2], vec![]];
        let seeds = vec![0, 0, 4];
        let successor_allocation = successors.as_ptr();
        let reach_allocation = seeds.as_ptr();
        let graph = build(successors, seeds);
        assert_eq!(graph.succ.as_ptr(), successor_allocation);
        assert_eq!(graph.reach.as_ptr(), reach_allocation);
        assert_eq!(graph.succ, vec![vec![1, 2], vec![2], vec![]]);
        assert_eq!(graph.reach, vec![4, 4, 4]);
        assert!(graph.writes.is_empty());
        assert_eq!(graph.register_count, 0);
    }
}
