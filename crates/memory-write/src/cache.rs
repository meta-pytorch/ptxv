/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::hash::Hash;
use std::hash::Hasher;

/// Advances a program identity without ever recycling an exhausted counter.
pub fn next_generation(current: u64) -> Option<u64> {
    if current == u64::MAX {
        None
    } else {
        Some(current + 1)
    }
}

#[derive(Clone, Copy, Debug)]
pub struct CacheKey {
    pub generation: u64,
    pub pc: u32,
    pub barrier: u32,
}

pub fn make_key(generation: u64, pc: u32, barrier: u32) -> CacheKey {
    CacheKey {
        generation,
        pc,
        barrier,
    }
}

pub fn same_key(left: CacheKey, right: CacheKey) -> bool {
    left.generation == right.generation && left.pc == right.pc && left.barrier == right.barrier
}

impl PartialEq for CacheKey {
    fn eq(&self, other: &Self) -> bool {
        same_key(*self, *other)
    }
}

impl Eq for CacheKey {}

impl Hash for CacheKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.generation.hash(state);
        self.pc.hash(state);
        self.barrier.hash(state);
    }
}

/// Failed reads are observations too: unknown and known zero are distinct.
pub fn same_observation(recorded: Option<u64>, current: Option<u64>) -> bool {
    match (recorded, current) {
        (None, None) => true,
        (Some(left), Some(right)) => left == right,
        _ => false,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Replay {
    matches: bool,
}

pub fn begin_replay(recorded: CacheKey, current: CacheKey) -> Replay {
    Replay {
        matches: same_key(recorded, current),
    }
}

pub fn observe(replay: Replay, recorded: Option<u64>, current: Option<u64>) -> Replay {
    Replay {
        matches: replay.matches && same_observation(recorded, current),
    }
}

pub fn replay_possible(replay: Replay) -> bool {
    replay.matches
}

pub fn finish_replay(replay: Replay, cached: bool) -> Option<bool> {
    if replay.matches { Some(cached) } else { None }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identities_do_not_wrap_or_stall() {
        assert_eq!(next_generation(0), Some(1));
        assert_eq!(next_generation(u64::MAX - 1), Some(u64::MAX));
        assert_eq!(next_generation(u64::MAX), None);
    }

    #[test]
    fn every_semantic_key_field_participates() {
        let base = make_key(19, 7, 4);
        assert!(same_key(base, base));
        assert!(!same_key(base, make_key(20, 7, 4)));
        assert!(!same_key(base, make_key(19, 8, 4)));
        assert!(!same_key(base, make_key(19, 7, 8)));
    }

    #[test]
    fn failed_read_is_not_known_zero() {
        for (recorded, current) in [
            (None, None),
            (None, Some(0)),
            (Some(0), None),
            (Some(0), Some(0)),
            (Some(1), Some(0)),
            (Some(u64::MAX), Some(u64::MAX)),
        ] {
            assert_eq!(same_observation(recorded, current), recorded == current);
        }
    }

    #[test]
    fn replay_never_forgets_an_earlier_mismatch() {
        let key = make_key(1, 9, 2);
        let matching = begin_replay(key, key);
        assert_eq!(finish_replay(matching, false), Some(false));
        assert_eq!(finish_replay(matching, true), Some(true));
        let different_program = begin_replay(key, make_key(2, 9, 2));
        let different_program = observe(different_program, Some(7), Some(7));
        assert_eq!(finish_replay(different_program, true), None);
        let different_read = observe(matching, Some(7), Some(8));
        let different_read = observe(different_read, Some(9), Some(9));
        assert_eq!(finish_replay(different_read, true), None);
        let failed_read = observe(matching, None, Some(0));
        assert_eq!(finish_replay(failed_read, true), None);
    }
}
