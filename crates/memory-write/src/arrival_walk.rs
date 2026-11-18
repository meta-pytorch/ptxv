/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

pub const DEFAULT_BUDGET: usize = 1 << 20;

pub struct Frontier<State> {
    slots: Vec<Option<State>>,
    len: usize,
}

impl<State> Frontier<State> {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            len: 0,
        }
    }

    pub fn push(&mut self, state: State) -> bool {
        if self.len == usize::MAX {
            return false;
        }
        if self.len < self.slots.len() {
            self.slots[self.len] = Some(state);
        } else {
            self.slots.push(Some(state));
        }
        self.len += 1;
        true
    }

    fn pop(&mut self) -> Option<State> {
        if self.len == 0 {
            None
        } else {
            self.len -= 1;
            self.slots[self.len].take()
        }
    }
}

pub enum Expansion<State> {
    Miss,
    Hit,
    Next(State),
    Children(Frontier<State>),
}

pub trait Walker<State> {
    fn begin(&mut self) -> State;
    fn fresh(&mut self, state: &State) -> bool;
    fn expand(&mut self, state: State) -> Expansion<State>;
    fn retry(&mut self) -> bool;
}

pub trait ReadDependencies {
    fn count(&self) -> usize;
    fn conflict(&self, index: usize) -> bool;
}

pub fn has_conflict<D: ReadDependencies>(dependencies: &D) -> bool {
    let count = dependencies.count();
    let mut index = 0;
    while index < count {
        if dependencies.conflict(index) {
            return true;
        }
        index += 1;
    }
    false
}

fn push_children<State>(pending: &mut Frontier<State>, mut children: Frontier<State>) -> bool {
    while let Some(state) = children.pop() {
        if !pending.push(state) {
            return false;
        }
    }
    true
}

fn restart<State, W: Walker<State>>(walker: &mut W, pending: &mut Frontier<State>) -> bool {
    let initial = walker.begin();
    pending.push(initial)
}

/// Children are supplied in reverse traversal order. Exhaustion is conservative.
pub fn may_arrive<State, W: Walker<State>>(walker: &mut W, mut budget: usize) -> bool {
    let mut pending = Frontier::new();
    let mut started = false;
    while budget > 0 {
        budget -= 1;
        match pending.pop() {
            None => {
                if started && !walker.retry() {
                    return false;
                }
                if !restart(walker, &mut pending) {
                    return true;
                }
                started = true;
            }
            Some(state) => {
                if !walker.fresh(&state) {
                    continue;
                }
                match walker.expand(state) {
                    Expansion::Miss => {}
                    Expansion::Hit => return true,
                    Expansion::Next(state) => {
                        if !pending.push(state) {
                            return true;
                        }
                    }
                    Expansion::Children(children) => {
                        if !push_children(&mut pending, children) {
                            return true;
                        }
                    }
                }
            }
        }
    }
    true
}

impl<State> Default for Frontier<State> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::collections::VecDeque;

    use super::Expansion;
    use super::Frontier;
    use super::ReadDependencies;
    use super::Walker;
    use super::has_conflict;
    use super::may_arrive;

    struct Probe {
        edges: Vec<Vec<usize>>,
        hits: u8,
        start: usize,
        seen: HashSet<usize>,
        visited: Vec<usize>,
        attempts: usize,
        retries: usize,
    }

    impl Walker<usize> for Probe {
        fn begin(&mut self) -> usize {
            self.seen.clear();
            self.start
        }
        fn fresh(&mut self, state: &usize) -> bool {
            self.seen.insert(*state)
        }
        fn expand(&mut self, state: usize) -> Expansion<usize> {
            self.visited.push(state);
            if self.hits & (1 << state) != 0 {
                return Expansion::Hit;
            }
            match self.edges[state].as_slice() {
                [] => Expansion::Miss,
                &[next] => Expansion::Next(next),
                children => {
                    let mut frontier = Frontier::new();
                    for &child in children.iter().rev() {
                        assert!(frontier.push(child));
                    }
                    Expansion::Children(frontier)
                }
            }
        }
        fn retry(&mut self) -> bool {
            self.attempts += 1;
            self.attempts <= self.retries
        }
    }

    fn fresh_probe(edges: Vec<Vec<usize>>, hits: u8, start: usize) -> Probe {
        Probe {
            edges,
            hits,
            start,
            seen: HashSet::new(),
            visited: Vec::new(),
            attempts: 0,
            retries: 0,
        }
    }

    fn reachable_hit(edges: &[Vec<usize>], hits: u8, start: usize) -> bool {
        let mut queue = VecDeque::from([start]);
        let mut reached = vec![false; edges.len()];
        while let Some(state) = queue.pop_front() {
            if reached[state] {
                continue;
            }
            reached[state] = true;
            if hits & (1 << state) != 0 {
                return true;
            }
            queue.extend(edges[state].iter().copied());
        }
        false
    }

    #[test]
    fn dependency_scan_covers_every_used_position() {
        struct Dependencies(Vec<bool>);
        impl ReadDependencies for Dependencies {
            fn count(&self) -> usize {
                self.0.len()
            }
            fn conflict(&self, index: usize) -> bool {
                self.0[index]
            }
        }
        for len in 0..=8 {
            for mask in 0..256 {
                let conflicts: Vec<_> = (0..len).map(|index| mask & (1 << index) != 0).collect();
                let expected = conflicts.contains(&true);
                assert_eq!(has_conflict(&Dependencies(conflicts)), expected);
            }
        }
    }

    #[test]
    fn completed_walk_agrees_with_independent_breadth_first_reachability() {
        for bits in 0..512usize {
            let edges: Vec<Vec<usize>> = (0..3)
                .map(|source| {
                    (0..3)
                        .filter(|target| bits & (1 << (source * 3 + target)) != 0)
                        .collect()
                })
                .collect();
            for hits in 0..8 {
                for start in 0..3 {
                    let expected = reachable_hit(&edges, hits, start);
                    assert_eq!(
                        may_arrive(&mut fresh_probe(edges.clone(), hits, start), 64),
                        expected
                    );
                    for budget in 0..8 {
                        let possible =
                            may_arrive(&mut fresh_probe(edges.clone(), hits, start), budget);
                        assert!(
                            possible || !expected,
                            "budget {budget}, graph {bits}, hits {hits}, start {start}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn driver_preserves_depth_first_order_and_retry_observations() {
        let mut probe = fresh_probe(vec![vec![1, 2], vec![], vec![1]], 0, 0);
        probe.retries = 1;
        assert!(!may_arrive(&mut probe, 64));
        assert_eq!(probe.visited, vec![0, 2, 1, 0, 2, 1]);
        assert_eq!(probe.attempts, 2);
    }

    #[test]
    fn retry_cycle_ends_with_a_conservative_budget_result() {
        let mut probe = fresh_probe(vec![vec![]], 0, 0);
        probe.retries = usize::MAX;
        assert!(may_arrive(&mut probe, 8));
        assert!(probe.attempts < 8);
    }

    #[test]
    fn exhausted_walk_keeps_a_reachable_hit_possible() {
        let edges = vec![vec![1], vec![2], vec![]];
        for budget in 0..4 {
            assert!(may_arrive(
                &mut fresh_probe(edges.clone(), 1 << 2, 0),
                budget
            ));
        }
    }
}
