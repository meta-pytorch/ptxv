/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

pub type Entry = (u32, u32);

pub trait OwnEntry {
    fn own_entry(&self) -> Entry;
}

pub trait KeepEntry {
    fn keep_entry(&self, entry: Entry) -> bool;
}

#[derive(Debug, PartialEq, Eq)]
pub enum JoinPlan {
    Keep,
    ShareRight,
    Replace(Vec<Entry>),
}

#[inline]
pub fn empty() -> (Vec<Entry>, Entry) {
    (Vec::new(), (0, 0))
}

#[inline]
pub fn single(actor: u32, clock: u32) -> (Vec<Entry>, Entry) {
    let mut base = Vec::with_capacity(1);
    base.push((actor, clock));
    (base, (0, 0))
}

#[inline]
pub fn relocate(base: &[Entry], own: Entry, actor: u32) -> Option<Vec<Entry>> {
    if own.1 != 0 && own.0 != actor {
        Some(entries(base, own))
    } else {
        None
    }
}

#[inline]
pub fn install_relocation<T>(base: &mut T, own: &mut Entry, next: T) {
    *base = next;
    *own = (0, 0);
}

#[inline]
pub fn tick_own(base: &[Entry], own: &mut Entry, actor: u32) {
    let next = get(base, *own, actor) + 1;
    *own = (actor, next);
}

/// Finds an actor, or the position where it belongs in a sorted, distinct base.
#[inline]
pub fn locate(base: &[Entry], actor: u32) -> (usize, bool) {
    let mut low = 0;
    let mut high = base.len();
    while low < high {
        let middle = low + (high - low) / 2;
        let entry = base[middle];
        if entry.0 == actor {
            return (middle, true);
        }
        if entry.0 < actor {
            low = middle + 1;
        } else {
            high = middle;
        }
    }
    (low, false)
}

#[inline]
pub fn lookup(base: &[Entry], actor: u32) -> u32 {
    let (index, found) = locate(base, actor);
    if found { base[index].1 } else { 0 }
}

#[inline]
pub fn get(base: &[Entry], own: Entry, actor: u32) -> u32 {
    let clock = lookup(base, actor);
    if own.0 == actor {
        clock.max(own.1)
    } else {
        clock
    }
}

#[inline]
pub fn overlay_entry(result: &mut Vec<Entry>, actor: u32, stamp: u32) {
    if stamp != 0 {
        let (index, found) = locate(result, actor);
        if found {
            result[index].1 = result[index].1.max(stamp);
        } else if index == result.len() {
            result.push((actor, stamp));
        } else {
            result.insert(index, (actor, stamp));
        }
    }
}

pub fn entries(base: &[Entry], own: Entry) -> Vec<Entry> {
    let mut result = base.to_vec();
    overlay_entry(&mut result, own.0, own.1);
    result
}

pub fn join_shared<C: OwnEntry>(base: &[Entry], clocks: &[C]) -> (Vec<Entry>, Entry) {
    let mut result = base.to_vec();
    let mut index = 0;
    while index < clocks.len() {
        let (actor, stamp) = clocks[index].own_entry();
        overlay_entry(&mut result, actor, stamp);
        index += 1;
    }
    (result, (0, 0))
}

pub fn filter_entries<K: KeepEntry>(entries: &[Entry], keep: &K) -> Vec<Entry> {
    let mut result = Vec::new();
    let mut index = 0;
    while index < entries.len() {
        let entry = entries[index];
        if keep.keep_entry(entry) {
            result.push(entry);
        }
        index += 1;
    }
    result
}

struct Uncovered<'a> {
    base: &'a [Entry],
    own: Entry,
}

impl KeepEntry for Uncovered<'_> {
    fn keep_entry(&self, (actor, stamp): Entry) -> bool {
        get(self.base, self.own, actor) < stamp
    }
}

pub fn uncovered_clock(
    base: &[Entry],
    own: Entry,
    seen: &[Entry],
    seen_own: Entry,
) -> Option<(Vec<Entry>, Entry)> {
    let entries = entries(base, own);
    let keep = Uncovered {
        base: seen,
        own: seen_own,
    };
    let selected = filter_entries(&entries, &keep);
    let selected_len = selected.len();
    if selected_len == 0 {
        None
    } else {
        Some(from_sorted(selected))
    }
}

pub fn tail(base: &[Entry], own: Entry, lower: u32) -> Vec<Entry> {
    let (start, _) = locate(base, lower);
    let mut result = base[start..base.len()].to_vec();
    if own.0 >= lower {
        overlay_entry(&mut result, own.0, own.1);
    }
    result
}

#[inline]
pub fn from_sorted(entries: Vec<Entry>) -> (Vec<Entry>, Entry) {
    (entries, (0, 0))
}

pub const READ_BIT: u32 = 1 << 31;

#[inline]
pub fn read_actor(actor: u32) -> u32 {
    if actor < READ_BIT {
        actor + READ_BIT
    } else {
        actor
    }
}

pub fn read_release_clock(
    base: &[Entry],
    own: Entry,
    lower: u32,
    reader: u32,
    done: u32,
) -> (Vec<Entry>, Entry) {
    let source = tail(base, own, lower);
    let mut fresh = Vec::new();
    let mut inherited = Vec::new();
    let mut index = 0;
    while index < source.len() {
        let (actor, stamp) = source[index];
        if actor < READ_BIT {
            fresh.push((read_actor(actor), stamp));
        } else {
            inherited.push((actor, stamp));
        }
        index += 1;
    }
    let mut result = merge_entries(&fresh, &inherited);
    overlay_entry(&mut result, read_actor(reader), done);
    from_sorted(result)
}

pub fn meet_entries(left: &[Entry], right: &[Entry]) -> Vec<Entry> {
    let mut result = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < left.len() && j < right.len() {
        let a = left[i];
        let b = right[j];
        if a.0 < b.0 {
            i += 1;
        } else if b.0 < a.0 {
            j += 1;
        } else {
            let stamp = a.1.min(b.1);
            if stamp > 0 {
                result.push((a.0, stamp));
            }
            i += 1;
            j += 1;
        }
    }
    result
}

pub fn meet_clock(
    left: &[Entry],
    left_own: Entry,
    right: &[Entry],
    right_own: Entry,
) -> (Vec<Entry>, Entry) {
    let left = entries(left, left_own);
    let right = entries(right, right_own);
    (meet_entries(&left, &right), (0, 0))
}

#[inline]
pub fn push_coalesced(result: &mut Vec<Entry>, entry: Entry) {
    let length = result.len();
    if length > 0 {
        let last = length - 1;
        if result[last].0 == entry.0 {
            result[last].1 = result[last].1.max(entry.1);
            return;
        }
    }
    result.push(entry);
}

pub fn append_sorted_suffix(result: &mut Vec<Entry>, suffix: &[Entry]) {
    let length = suffix.len();
    if length > 0 {
        push_coalesced(result, suffix[0]);
        result.extend_from_slice(&suffix[1..length]);
    }
}

pub fn merge_sorted_updates(base: &[Entry], updates: &[Entry]) -> Vec<Entry> {
    let mut result = Vec::with_capacity(base.len() + updates.len());
    let (mut i, mut j) = (0, 0);
    while i < base.len() || j < updates.len() {
        if i < base.len() && j < updates.len() {
            let a = base[i];
            let b = updates[j];
            if a.0 == b.0 {
                push_coalesced(&mut result, (a.0, a.1.max(b.1)));
                i += 1;
                j += 1;
            } else if a.0 < b.0 {
                push_coalesced(&mut result, a);
                i += 1;
            } else {
                push_coalesced(&mut result, b);
                j += 1;
            }
        } else if i < base.len() {
            append_sorted_suffix(&mut result, &base[i..base.len()]);
            return result;
        } else {
            push_coalesced(&mut result, updates[j]);
            j += 1;
        }
    }
    result
}

pub fn join_sorted_updates(base: &[Entry], own: Entry, updates: &[Entry]) -> (Vec<Entry>, Entry) {
    let base = entries(base, own);
    (merge_sorted_updates(&base, updates), (0, 0))
}

pub fn merge_entries(left: &[Entry], right: &[Entry]) -> Vec<Entry> {
    let mut result = Vec::with_capacity(left.len().max(right.len()));
    let mut i = 0;
    let mut j = 0;
    while i < left.len() || j < right.len() {
        if i < left.len() && j < right.len() {
            let a = left[i];
            let b = right[j];
            if a.0 == b.0 {
                result.push((a.0, a.1.max(b.1)));
                i += 1;
                j += 1;
            } else if a.0 < b.0 {
                result.push(a);
                i += 1;
            } else {
                result.push(b);
                j += 1;
            }
        } else if i < left.len() {
            result.push(left[i]);
            i += 1;
        } else {
            result.push(right[j]);
            j += 1;
        }
    }
    result
}

#[inline]
pub fn merge_clock(left: &[Entry], right: &[Entry], right_own: Entry) -> Vec<Entry> {
    if right_own.1 == 0 {
        merge_entries(left, right)
    } else {
        merge_entries(left, &entries(right, right_own))
    }
}

pub fn entry_covered(base: &[Entry], entry: Entry) -> bool {
    let (index, found) = locate(base, entry.0);
    found && base[index].1 >= entry.1
}

pub fn base_covers_sparse(left: &[Entry], right: &[Entry]) -> bool {
    let mut covered = true;
    let mut index = 0;
    while covered && index < right.len() {
        let entry = right[index];
        covered = entry_covered(left, entry);
        index += 1;
    }
    covered
}

pub fn base_covers_linear(left: &[Entry], right: &[Entry]) -> bool {
    let mut covered = true;
    let (mut i, mut j) = (0, 0);
    while covered && j < right.len() {
        if i < left.len() {
            let a = left[i];
            let b = right[j];
            if a.0 < b.0 {
                i += 1;
            } else {
                covered = a.0 == b.0 && a.1 >= b.1;
                i += 1;
                j += 1;
            }
        } else {
            covered = false;
        }
    }
    covered
}

pub fn base_covers(left: &[Entry], right: &[Entry]) -> bool {
    if right.len() < left.len() / 16 {
        base_covers_sparse(left, right)
    } else {
        base_covers_linear(left, right)
    }
}

/// `shared_base` means the two slices are the same immutable base.
#[inline]
pub fn join_plan(
    left: &[Entry],
    left_own: Entry,
    right: &[Entry],
    right_own: Entry,
    shared_base: bool,
) -> JoinPlan {
    if right.is_empty() && right_own.1 == 0 {
        return JoinPlan::Keep;
    }
    if shared_base && (right_own.1 == 0 || get(left, left_own, right_own.0) >= right_own.1) {
        return JoinPlan::Keep;
    }
    if (right_own.1 == 0 || lookup(left, right_own.0) >= right_own.1) && base_covers(left, right) {
        return JoinPlan::Keep;
    }
    if right_own.1 == 0 && base_covers(right, left) {
        return JoinPlan::ShareRight;
    }
    JoinPlan::Replace(merge_clock(left, right, right_own))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense(base: &[Entry], own: Entry) -> [u32; 5] {
        let mut result = [0; 5];
        for &(actor, clock) in base.iter().chain([own].iter()) {
            result[actor as usize] = result[actor as usize].max(clock);
        }
        result
    }

    #[test]
    fn read_release_projection_matches_ordered_map_with_actor_aliases() {
        let actors = [0, 1, READ_BIT - 1, READ_BIT, READ_BIT + 1, u32::MAX];
        for actor in actors {
            assert_eq!(read_actor(actor), actor | READ_BIT);
        }
        for mask in 0..64 {
            let base: Vec<_> = actors
                .iter()
                .enumerate()
                .filter(|(index, _)| mask & (1 << index) != 0)
                .map(|(index, &actor)| (actor, if index == 2 { 0 } else { index as u32 + 2 }))
                .collect();
            for &owner in &actors {
                for lower in [0, 1, 2, READ_BIT - 1, READ_BIT, READ_BIT + 2, u32::MAX] {
                    for &reader in &actors {
                        for done in [0, 3, u32::MAX] {
                            let own = (owner, 7);
                            let mut expected = std::collections::BTreeMap::<u32, u32>::new();
                            let mut logical = std::collections::BTreeMap::<u32, u32>::new();
                            for &(actor, stamp) in base.iter().chain([own].iter()) {
                                let cell = logical.entry(actor).or_default();
                                *cell = (*cell).max(stamp);
                            }
                            for (actor, stamp) in logical {
                                if actor >= lower {
                                    let cell = expected.entry(actor | READ_BIT).or_default();
                                    *cell = (*cell).max(stamp);
                                }
                            }
                            if done > 0 {
                                let cell = expected.entry(reader | READ_BIT).or_default();
                                *cell = (*cell).max(done);
                            }
                            let (result, cleared) =
                                read_release_clock(&base, own, lower, reader, done);
                            assert_eq!(cleared, (0, 0));
                            assert_eq!(result, expected.into_iter().collect::<Vec<_>>());
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn sorted_updates_coalesce_duplicate_actors_and_match_dense_join() {
        for mask in 0..32 {
            let base: Vec<_> = (0..5)
                .filter(|actor| mask & (1 << actor) != 0)
                .map(|actor| (actor, if actor == 2 { 0 } else { actor + 2 }))
                .collect();
            for encoded in 0..256_u32 {
                let mut updates: Vec<_> = (0..4)
                    .map(|index| {
                        let item = (encoded >> (index * 2)) & 3;
                        (item, if index == 0 { 0 } else { 2 * index + item })
                    })
                    .collect();
                updates.sort_unstable();
                let output = merge_sorted_updates(&base, &updates);
                let mut expected = dense(&base, (0, 0));
                for &(actor, stamp) in &updates {
                    expected[actor as usize] = expected[actor as usize].max(stamp);
                }
                assert!(output.windows(2).all(|pair| pair[0].0 < pair[1].0));
                assert_eq!(dense(&output, (0, 0)), expected);
                let (joined, own) = join_sorted_updates(&base, (4, 9), &updates);
                expected[4] = expected[4].max(9);
                assert_eq!(own, (0, 0));
                assert!(joined.windows(2).all(|pair| pair[0].0 < pair[1].0));
                assert_eq!(dense(&joined, own), expected);
            }
        }
        assert_eq!(merge_sorted_updates(&[], &[]), []);
        assert_eq!(
            merge_sorted_updates(&[(0, 2), (u32::MAX, 0)], &[(0, u32::MAX), (0, 1)]),
            [(0, u32::MAX), (u32::MAX, 0)]
        );
    }

    #[test]
    fn sorted_lookup_and_overlay_match_dense_clocks() {
        for mask in 0..32 {
            let base: Vec<_> = (0..5)
                .filter(|actor| mask & (1 << actor) != 0)
                .map(|actor| (actor, if actor == 2 { 0 } else { 3 + actor }))
                .collect();
            for actor in 0..5 {
                for clock in [0, 1, 6, u32::MAX] {
                    let own = (actor, clock);
                    let expected = dense(&base, own);
                    let result = entries(&base, own);
                    assert!(result.windows(2).all(|pair| pair[0].0 < pair[1].0));
                    assert_eq!(dense(&result, (0, 0)), expected);
                    for query in 0..5 {
                        assert_eq!(get(&base, own, query), expected[query as usize]);
                    }
                }
            }
        }
        assert_eq!(lookup(&[(u32::MAX, u32::MAX)], u32::MAX), u32::MAX);
        assert_eq!(lookup(&[(u32::MAX, 7)], 0), 0);
        assert_eq!(locate(&[(0, 1), (u32::MAX, 2)], 1), (1, false));
    }

    #[test]
    fn covered_joins_reuse_existing_storage_plans() {
        assert!(!entry_covered(&[], (1, 0)));
        assert!(entry_covered(&[(1, 0)], (1, 0)));
        assert_eq!(
            join_plan(&[(1, 4), (2, 8)], (3, 2), &[(1, 3)], (2, 7), false),
            JoinPlan::Keep
        );
        assert_eq!(
            join_plan(&[(1, 3)], (3, 9), &[(1, 4), (2, 8)], (0, 0), false),
            JoinPlan::ShareRight
        );
        assert!(matches!(
            join_plan(&[(1, 3)], (3, 9), &[(2, 8)], (0, 0), false),
            JoinPlan::Replace(_)
        ));
    }

    #[test]
    fn every_join_plan_preserves_pointwise_maximum() {
        let bases: Vec<Vec<Entry>> = (0..32)
            .map(|mask| {
                (0..5)
                    .filter(|actor| mask & (1 << actor) != 0)
                    .map(|actor| (actor, if actor == 2 { 0 } else { 2 + actor }))
                    .collect()
            })
            .collect();
        for left in &bases {
            for right in &bases {
                for left_own in [(0, 0), (2, 9), (4, 1)] {
                    for right_own in [(0, 0), (0, 8), (2, 3), (4, u32::MAX)] {
                        for shared in [false, true] {
                            if shared && left != right {
                                continue;
                            }
                            let plan = join_plan(left, left_own, right, right_own, shared);
                            let output = match plan {
                                JoinPlan::Keep => left.clone(),
                                JoinPlan::ShareRight => right.clone(),
                                JoinPlan::Replace(output) => output,
                            };
                            assert!(output.windows(2).all(|pair| pair[0].0 < pair[1].0));
                            let a = dense(left, left_own);
                            let b = dense(right, right_own);
                            let expected = std::array::from_fn(|i| a[i].max(b[i]));
                            assert_eq!(dense(&output, left_own), expected);
                        }
                    }
                }
            }
        }
        let (base, own) = empty();
        assert_eq!(dense(&base, own), [0; 5]);
    }
}

#[cfg(test)]
mod tick_tests {
    use super::*;

    #[test]
    fn relocation_and_tick_match_dense_increment() {
        for mask in 0..32 {
            let base: Vec<_> = (0..5)
                .filter(|actor| mask & (1 << actor) != 0)
                .map(|actor| (actor, actor + 1))
                .collect();
            for own in [(0, 0), (2, 9), (4, 1)] {
                for actor in 0..5 {
                    let before: Vec<_> = (0..5).map(|query| get(&base, own, query)).collect();
                    let mut base = base.clone();
                    let mut own = own;
                    if let Some(next) = relocate(&base, own, actor) {
                        install_relocation(&mut base, &mut own, next);
                    }
                    tick_own(&base, &mut own, actor);
                    assert!(base.windows(2).all(|pair| pair[0].0 < pair[1].0));
                    for query in 0..5 {
                        assert_eq!(
                            get(&base, own, query),
                            before[query as usize] + u32::from(query == actor)
                        );
                    }
                }
            }
        }
        assert_eq!(single(u32::MAX, 0), (vec![(u32::MAX, 0)], (0, 0)));
    }

    #[test]
    #[should_panic(expected = "attempt to add with overflow")]
    fn tick_rejects_overflow() {
        tick_own(&[], &mut (3, u32::MAX), 3);
    }
}

#[cfg(test)]
mod shared_join_tests {
    use super::*;

    impl OwnEntry for Entry {
        fn own_entry(&self) -> Entry {
            *self
        }
    }

    #[test]
    fn shared_join_matches_a_dense_maximum_for_unsorted_repeated_owns() {
        let owns = [(4, 3), (1, 7), (2, 0), (4, 9), (0, u32::MAX), (1, 2)];
        for mask in 0..32 {
            let base: Vec<_> = (0..5)
                .filter(|actor| mask & (1 << actor) != 0)
                .map(|actor| (actor, actor + 1))
                .collect();
            for count in 0..=owns.len() {
                let (result, own) = join_shared(&base, &owns[..count]);
                assert_eq!(own, (0, 0));
                assert!(result.windows(2).all(|pair| pair[0].0 < pair[1].0));
                for actor in 0..5 {
                    let expected = owns[..count]
                        .iter()
                        .filter(|entry| entry.0 == actor)
                        .map(|entry| entry.1)
                        .fold(lookup(&base, actor), u32::max);
                    assert_eq!(get(&result, own, actor), expected);
                }
            }
        }
    }
}

#[cfg(test)]
mod meet_tests {
    use super::*;

    #[test]
    fn meet_matches_a_dense_minimum_with_missing_zero_and_own_entries() {
        let owns = [(0, 0), (0, 8), (2, 1), (4, 9)];
        for left_mask in 0..32_u32 {
            for right_mask in 0..32_u32 {
                let left: Vec<_> = (0..5)
                    .filter(|actor| left_mask & (1 << actor) != 0)
                    .map(|actor| (actor, (left_mask + 3 * actor) % 9))
                    .collect();
                let right: Vec<_> = (0..5)
                    .filter(|actor| right_mask & (1 << actor) != 0)
                    .map(|actor| (actor, (right_mask + 5 * actor) % 11))
                    .collect();
                for &left_own in &owns {
                    for &right_own in &owns {
                        let mut dense_left = [0; 5];
                        let mut dense_right = [0; 5];
                        for &(actor, stamp) in &left {
                            dense_left[actor as usize] = stamp;
                        }
                        for &(actor, stamp) in &right {
                            dense_right[actor as usize] = stamp;
                        }
                        dense_left[left_own.0 as usize] =
                            dense_left[left_own.0 as usize].max(left_own.1);
                        dense_right[right_own.0 as usize] =
                            dense_right[right_own.0 as usize].max(right_own.1);
                        let expected: Vec<_> = (0..5_u32)
                            .map(|actor| {
                                (
                                    actor,
                                    dense_left[actor as usize].min(dense_right[actor as usize]),
                                )
                            })
                            .filter(|&(_, stamp)| stamp != 0)
                            .collect();
                        let (actual, own) = meet_clock(&left, left_own, &right, right_own);
                        assert_eq!(actual, expected);
                        assert_eq!(own, (0, 0));
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod filter_tests {
    use std::collections::BTreeMap;

    use super::*;

    impl KeepEntry for u32 {
        fn keep_entry(&self, (actor, _): Entry) -> bool {
            self & (1 << actor) != 0
        }
    }

    #[test]
    fn filtering_keeps_original_actor_order_and_sorted_import_keeps_entries() {
        for mask in 0..32_u32 {
            let entries: Vec<_> = (0..5).map(|actor| (actor, actor * 3)).collect();
            let actual = filter_entries(&entries, &mask);
            let expected: Vec<_> = entries
                .iter()
                .copied()
                .filter(|&(actor, _)| mask & (1 << actor) != 0)
                .collect();
            assert_eq!(actual, expected);
            assert!(actual.windows(2).all(|pair| pair[0].0 < pair[1].0));
            let (imported, own) = from_sorted(actual.clone());
            assert_eq!(imported, actual);
            assert_eq!(own, (0, 0));
        }
    }

    #[test]
    fn tail_matches_an_independent_ordered_map_with_zero_and_own_entries() {
        for mask in 0..32_u32 {
            let base: Vec<_> = (0..5)
                .filter(|actor| mask & (1 << actor) != 0)
                .map(|actor| (actor, (mask + actor) % 7))
                .collect();
            for own in [(0, 0), (0, 9), (2, 7), (5, 3), (5, u32::MAX)] {
                let mut expected: BTreeMap<_, _> = base.iter().copied().collect();
                if own.1 != 0 {
                    expected
                        .entry(own.0)
                        .and_modify(|stamp| *stamp = (*stamp).max(own.1))
                        .or_insert(own.1);
                }
                for lower in 0..7 {
                    let expected: Vec<_> = expected
                        .range(lower..)
                        .map(|(&actor, &stamp)| (actor, stamp))
                        .collect();
                    assert_eq!(tail(&base, own, lower), expected);
                }
            }
        }
    }
}

#[cfg(test)]
mod uncovered_tests {
    use std::collections::BTreeMap;

    use super::*;

    #[test]
    fn uncovered_clock_matches_an_independent_ordered_filter_and_empty_branch() {
        for incoming_mask in 0..32_u32 {
            for seen_mask in 0..32_u32 {
                let incoming: Vec<_> = (0..5)
                    .filter(|actor| incoming_mask & (1 << actor) != 0)
                    .map(|actor| (actor, (incoming_mask + 3 * actor) % 9))
                    .collect();
                let seen: Vec<_> = (0..5)
                    .filter(|actor| seen_mask & (1 << actor) != 0)
                    .map(|actor| (actor, (seen_mask + 5 * actor) % 11))
                    .collect();
                for incoming_own in [(0, 0), (1, 7)] {
                    for seen_own in [(0, 0), (2, 8)] {
                        let mut incoming_map: BTreeMap<_, _> = incoming.iter().copied().collect();
                        let mut seen_map: BTreeMap<_, _> = seen.iter().copied().collect();
                        for (map, own) in
                            [(&mut incoming_map, incoming_own), (&mut seen_map, seen_own)]
                        {
                            if own.1 != 0 {
                                map.entry(own.0)
                                    .and_modify(|stamp| *stamp = (*stamp).max(own.1))
                                    .or_insert(own.1);
                            }
                        }
                        let expected: Vec<_> = incoming_map
                            .into_iter()
                            .filter(|&(actor, stamp)| {
                                stamp > seen_map.get(&actor).copied().unwrap_or(0)
                            })
                            .collect();
                        let actual = uncovered_clock(&incoming, incoming_own, &seen, seen_own);
                        if expected.is_empty() {
                            assert_eq!(actual, None);
                        } else {
                            assert_eq!(actual, Some((expected, (0, 0))));
                        }
                    }
                }
            }
        }
    }
}
