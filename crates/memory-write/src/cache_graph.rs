/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#[derive(Debug, Default)]
pub struct MayWriteGraph {
    pub reach: Vec<u32>,
    pub succ: Vec<Vec<usize>>,
    pub writes: Vec<Vec<usize>>,
    pub register_count: usize,
}

#[derive(Debug)]
pub struct PcWrites {
    pub pc: usize,
    pub words: Vec<u64>,
}

#[derive(Debug)]
pub struct MayWriteSets {
    pub rows: Vec<PcWrites>,
}

pub fn valid_indices(indices: &[usize], bound: usize) -> bool {
    let mut index = 0;
    let mut valid = true;
    while index < indices.len() && valid {
        valid = indices[index] < bound;
        index += 1;
    }
    valid
}

pub fn valid_graph(graph: &MayWriteGraph) -> bool {
    let count = graph.reach.len();
    if graph.succ.len() != count || graph.writes.len() != count {
        return false;
    }
    let mut pc = 0;
    let mut valid = true;
    while pc < count && valid {
        valid = valid_indices(&graph.succ[pc], count)
            && valid_indices(&graph.writes[pc], graph.register_count);
        pc += 1;
    }
    valid
}

pub fn zero_words(count: usize) -> Vec<u64> {
    let mut words = Vec::new();
    let mut index = 0;
    while index < count {
        words.push(0);
        index += 1;
    }
    words
}

pub fn copy_words(source: &[u64]) -> Vec<u64> {
    let mut words = Vec::new();
    let mut index = 0;
    while index < source.len() {
        words.push(source[index]);
        index += 1;
    }
    words
}

pub fn join_words(target: &mut [u64], source: &[u64]) -> bool {
    let mut changed = false;
    let mut index = 0;
    while index < source.len() {
        let joined = target[index] | source[index];
        if target[index] != joined {
            target[index] = joined;
            changed = true;
        }
        index += 1;
    }
    changed
}

pub fn push_work(work: &mut Vec<usize>, pending: usize, pc: usize) -> usize {
    if pending == work.len() {
        work.push(pc);
    } else {
        work[pending] = pc;
    }
    pending + 1
}

const PAGE_SIZE: usize = 64;

#[derive(Clone, Copy)]
struct PageIndex {
    page: usize,
    storage: usize,
}

struct WorkSets {
    directory: Vec<PageIndex>,
    pages: Vec<Vec<Option<Vec<u64>>>>,
}

fn page_position(directory: &[PageIndex], page: usize) -> usize {
    let mut left = 0;
    let mut right = directory.len();
    while left < right {
        let middle = left + (right - left) / 2;
        if directory[middle].page < page {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    left
}

fn empty_page() -> Vec<Option<Vec<u64>>> {
    let mut slots = Vec::with_capacity(PAGE_SIZE);
    let mut offset = 0;
    while offset < PAGE_SIZE {
        slots.push(None);
        offset += 1;
    }
    slots
}

fn insert_page(directory: &mut Vec<PageIndex>, position: usize, entry: PageIndex) {
    if position == directory.len() {
        directory.push(entry);
    } else {
        directory.insert(position, entry);
    }
}

fn add_page(sets: &mut WorkSets, page: usize, position: usize) -> usize {
    let storage = sets.pages.len();
    sets.pages.push(empty_page());
    insert_page(&mut sets.directory, position, PageIndex { page, storage });
    storage
}

fn ensure_page(sets: &mut WorkSets, page: usize) -> usize {
    let position = page_position(&sets.directory, page);
    if position < sets.directory.len() && sets.directory[position].page == page {
        sets.directory[position].storage
    } else {
        add_page(sets, page, position)
    }
}

fn ensure_pc(sets: &mut WorkSets, pc: usize, word_count: usize) -> (usize, usize, bool) {
    let storage = ensure_page(sets, pc / PAGE_SIZE);
    let offset = pc % PAGE_SIZE;
    let fresh = sets.pages[storage][offset].is_none();
    if fresh {
        sets.pages[storage][offset] = Some(zero_words(word_count));
    }
    (storage, offset, fresh)
}

fn merge_row(sets: &mut WorkSets, storage: usize, offset: usize, source: &[u64]) -> bool {
    match &mut sets.pages[storage][offset] {
        Some(target) => join_words(target, source),
        None => panic!("initialized may-write successor has no set"),
    }
}

fn copy_incoming(sets: &WorkSets, pc: usize) -> Vec<u64> {
    let page = pc / PAGE_SIZE;
    let position = page_position(&sets.directory, page);
    assert!(
        position < sets.directory.len() && sets.directory[position].page == page,
        "may-write worklist pc has no page"
    );
    let storage = sets.directory[position].storage;
    match &sets.pages[storage][pc % PAGE_SIZE] {
        Some(incoming) => copy_words(incoming),
        None => panic!("may-write worklist pc has no incoming set"),
    }
}

fn propagate_successor(
    sets: &mut WorkSets,
    work: &mut Vec<usize>,
    pending: usize,
    successor: usize,
    words: usize,
    out: &[u64],
) -> usize {
    let (storage, offset, fresh) = ensure_pc(sets, successor, words);
    let changed = merge_row(sets, storage, offset, out);
    if fresh || changed {
        push_work(work, pending, successor)
    } else {
        pending
    }
}

fn compact(sets: &WorkSets) -> MayWriteSets {
    let mut rows = Vec::new();
    let mut page = 0;
    while page < sets.directory.len() {
        let entry = sets.directory[page];
        let mut offset = 0;
        while offset < PAGE_SIZE {
            if let Some(words) = &sets.pages[entry.storage][offset] {
                rows.push(PcWrites {
                    pc: entry.page * PAGE_SIZE + offset,
                    words: copy_words(words),
                });
            }
            offset += 1;
        }
        page += 1;
    }
    MayWriteSets { rows }
}

pub fn may_write_sets(graph: &MayWriteGraph, start: usize, bit: u32) -> MayWriteSets {
    let mut sets = WorkSets {
        directory: Vec::new(),
        pages: Vec::new(),
    };
    let words = graph.register_count / 64 + usize::from(!graph.register_count.is_multiple_of(64));
    let mut work = Vec::new();
    let mut pending = 0;
    if start < graph.reach.len() {
        ensure_pc(&mut sets, start, words);
        pending = push_work(&mut work, pending, start);
    }
    while pending != 0 {
        pending -= 1;
        let pc = work[pending];
        if graph.reach[pc] & bit == 0 {
            continue;
        }
        let mut out = copy_incoming(&sets, pc);
        let mut index = 0;
        while index < graph.writes[pc].len() {
            let register = graph.writes[pc][index];
            out[register / 64] |= 1 << (register % 64);
            index += 1;
        }
        index = 0;
        while index < graph.succ[pc].len() {
            let successor = graph.succ[pc][index];
            pending = propagate_successor(&mut sets, &mut work, pending, successor, words, &out);
            index += 1;
        }
    }
    compact(&sets)
}

pub fn find_pc(rows: &[PcWrites], pc: usize) -> Option<usize> {
    let mut left = 0;
    let mut right = rows.len();
    while left < right {
        let middle = left + (right - left) / 2;
        if rows[middle].pc < pc {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    if left < rows.len() && rows[left].pc == pc {
        Some(left)
    } else {
        None
    }
}

pub fn may_at(sets: &MayWriteSets, pc: usize, register: Option<usize>) -> bool {
    let Some(index) = find_pc(&sets.rows, pc) else {
        return true;
    };
    let words = &sets.rows[index].words;
    let Some(register) = register else {
        return false;
    };
    if register / 64 >= words.len() {
        return false;
    }
    words[register / 64] >> (register % 64) & 1 == 1
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;
    use std::collections::HashMap;

    use super::*;

    fn fresh_sets(
        graph: &MayWriteGraph,
        start: usize,
        bit: u32,
    ) -> HashMap<usize, BTreeSet<usize>> {
        let mut at: HashMap<usize, BTreeSet<usize>> = HashMap::new();
        let mut work = vec![start];
        at.insert(start, BTreeSet::new());
        while let Some(pc) = work.pop() {
            if pc >= graph.reach.len() || graph.reach[pc] & bit == 0 {
                continue;
            }
            let mut out = at[&pc].clone();
            out.extend(graph.writes[pc].iter().copied());
            for &successor in &graph.succ[pc] {
                let fresh = !at.contains_key(&successor);
                let incoming = at.entry(successor).or_default();
                let old = incoming.len();
                incoming.extend(out.iter().copied());
                if fresh || incoming.len() != old {
                    work.push(successor);
                }
            }
        }
        at
    }

    #[test]
    fn worklist_matches_set_semantics_on_cycles_and_unreachable_pcs() {
        for topology in 0..512 {
            for pattern in 0usize..8 {
                let graph = MayWriteGraph {
                    reach: (0..3)
                        .map(|pc| (1 + ((pattern >> pc) & 1)) as u32)
                        .collect(),
                    succ: (0..3)
                        .map(|pc| {
                            (0..3)
                                .filter(|&q| topology >> (pc * 3 + q) & 1 != 0)
                                .collect()
                        })
                        .collect(),
                    writes: (0..3)
                        .map(|pc| {
                            [0, 63, 64, 69]
                                .into_iter()
                                .filter(|&reg| (pattern + pc + reg) % 3 == 0)
                                .collect()
                        })
                        .collect(),
                    register_count: 70,
                };
                assert!(valid_graph(&graph));
                for start in 0..3 {
                    for bit in [1, 2, 3] {
                        let actual = may_write_sets(&graph, start, bit);
                        let expected = fresh_sets(&graph, start, bit);
                        for pc in 0..3 {
                            for register in [0, 1, 63, 64, 69] {
                                assert_eq!(
                                    may_at(&actual, pc, Some(register)),
                                    expected.get(&pc).is_none_or(|set| set.contains(&register)),
                                    "topology={topology} pattern={pattern} start={start} bit={bit} pc={pc} register={register}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn graph_bounds_and_unknown_queries_are_explicit() {
        let graph = MayWriteGraph {
            reach: vec![1],
            succ: vec![vec![]],
            writes: vec![vec![]],
            register_count: 0,
        };
        assert!(valid_graph(&graph));
        let sets = may_write_sets(&graph, 1, 1);
        assert!(may_at(&sets, 0, Some(0)));
        assert!(may_at(&sets, 1, None));
        assert!(!valid_graph(&MayWriteGraph {
            succ: vec![],
            ..graph
        }));
        assert!(!valid_graph(&MayWriteGraph {
            reach: vec![1],
            succ: vec![vec![1]],
            writes: vec![vec![]],
            register_count: 0
        }));
        assert!(!valid_graph(&MayWriteGraph {
            reach: vec![1],
            succ: vec![vec![]],
            writes: vec![vec![0]],
            register_count: 0
        }));
    }

    #[test]
    fn sparse_pages_preserve_arbitrary_discovery_order_and_holes() {
        let pcs = [0, 64, 129];
        for topology in 0..512 {
            let mut graph = MayWriteGraph {
                reach: vec![0; 130],
                succ: vec![vec![]; 130],
                writes: vec![vec![]; 130],
                register_count: 70,
            };
            for (index, &pc) in pcs.iter().enumerate() {
                graph.reach[pc] = 1;
                graph.succ[pc] = pcs
                    .iter()
                    .enumerate()
                    .filter_map(|(target, &q)| {
                        (topology >> (index * 3 + target) & 1 != 0).then_some(q)
                    })
                    .collect();
                graph.writes[pc] = vec![[0, 64, 69][index]];
            }
            assert!(valid_graph(&graph));
            for start in pcs {
                let actual = may_write_sets(&graph, start, 1);
                let expected = fresh_sets(&graph, start, 1);
                assert!(actual.rows.windows(2).all(|rows| rows[0].pc < rows[1].pc));
                for pc in [0, 1, 63, 64, 65, 128, 129, 130, usize::MAX] {
                    for register in [0, 1, 63, 64, 69] {
                        assert_eq!(
                            may_at(&actual, pc, Some(register)),
                            expected.get(&pc).is_none_or(|set| set.contains(&register))
                        );
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod query_progress_tests {
    use super::*;

    #[test]
    fn missing_pc_search_progresses() {
        let rows = vec![PcWrites {
            pc: 0,
            words: vec![0],
        }];
        assert_eq!(find_pc(&rows, 1), None);
    }

    #[test]
    fn high_register_query_has_a_valid_shift() {
        let sets = MayWriteSets {
            rows: vec![PcWrites {
                pc: 0,
                words: vec![0, 1],
            }],
        };
        assert!(may_at(&sets, 0, Some(64)));
    }
}
