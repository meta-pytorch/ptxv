/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::cache_graph::MayWriteGraph;
use crate::symbolic::RegisterKey;

const PAGE_SIZE: usize = 64;

#[derive(Debug)]
struct RegisterPage {
    key: u64,
    slots: Vec<Option<usize>>,
}

#[derive(Debug, Default)]
pub struct RegisterIndex {
    pages: Vec<RegisterPage>,
    count: usize,
}

fn page_key(key: RegisterKey) -> u64 {
    u64::from(key.thread) * 262144 + u64::from(key.file) * 1024 + u64::from(key.index / 64)
}

fn page_position(pages: &[RegisterPage], key: u64) -> usize {
    let mut left = 0;
    let mut right = pages.len();
    while left < right {
        let middle = left + (right - left) / 2;
        if pages[middle].key < key {
            left = middle + 1;
        } else {
            right = middle;
        }
    }
    left
}

fn empty_page(key: u64) -> RegisterPage {
    let mut slots = Vec::with_capacity(PAGE_SIZE);
    let mut offset = 0;
    while offset < PAGE_SIZE {
        slots.push(None);
        offset += 1;
    }
    RegisterPage { key, slots }
}

fn insert_page(pages: &mut Vec<RegisterPage>, position: usize, key: u64) {
    let page = empty_page(key);
    if position == pages.len() {
        pages.push(page);
    } else {
        pages.insert(position, page);
    }
}

fn ensure_page(pages: &mut Vec<RegisterPage>, key: u64) -> usize {
    let position = page_position(pages, key);
    if position == pages.len() || pages[position].key != key {
        insert_page(pages, position, key);
    }
    position
}

impl RegisterIndex {
    pub fn new() -> Self {
        Self {
            pages: Vec::new(),
            count: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    pub fn get(&self, key: RegisterKey) -> Option<usize> {
        let page = page_key(key);
        let position = page_position(&self.pages, page);
        if position < self.pages.len() && self.pages[position].key == page {
            self.pages[position].slots[usize::from(key.index % 64)]
        } else {
            None
        }
    }

    pub fn intern(&mut self, key: RegisterKey) -> usize {
        let page = page_key(key);
        let position = ensure_page(&mut self.pages, page);
        let offset = usize::from(key.index % 64);
        if let Some(index) = self.pages[position].slots[offset] {
            index
        } else {
            let index = self.count;
            self.count += 1;
            self.pages[position].slots[offset] = Some(index);
            index
        }
    }
}

pub fn build_writes(destinations: &[Vec<RegisterKey>]) -> (RegisterIndex, Vec<Vec<usize>>) {
    let mut index = RegisterIndex::new();
    let mut writes = Vec::new();
    let mut pc = 0;
    while pc < destinations.len() {
        let mut row = Vec::new();
        let mut position = 0;
        while position < destinations[pc].len() {
            row.push(index.intern(destinations[pc][position]));
            position += 1;
        }
        writes.push(row);
        pc += 1;
    }
    (index, writes)
}

pub fn publish_graph(
    mut graph: MayWriteGraph,
    destinations: &[Vec<RegisterKey>],
) -> (RegisterIndex, MayWriteGraph) {
    let (index, writes) = build_writes(destinations);
    graph.writes = writes;
    graph.register_count = index.len();
    (index, graph)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::symbolic::register_key;

    #[test]
    fn publication_preserves_graph_allocations_and_builds_exact_register_rows() {
        let local = register_key(0, 0, 7);
        let high = register_key(0, 4, u16::MAX);
        let destinations = vec![vec![local, high], vec![local], vec![]];
        let graph = MayWriteGraph {
            reach: vec![1, 1, 0],
            succ: vec![vec![1], vec![2], vec![]],
            writes: vec![vec![usize::MAX]],
            register_count: usize::MAX,
        };
        let reach_storage = graph.reach.as_ptr();
        let successor_storage = graph.succ.as_ptr();
        let (index, graph) = publish_graph(graph, &destinations);
        assert_eq!(graph.reach, vec![1, 1, 0]);
        assert_eq!(graph.succ, vec![vec![1], vec![2], vec![]]);
        assert_eq!(graph.reach.as_ptr(), reach_storage);
        assert_eq!(graph.succ.as_ptr(), successor_storage);
        assert_eq!(graph.register_count, 2);
        assert_eq!(graph.register_count, index.len());
        assert!(crate::cache_graph::valid_graph(&graph));
        for (keys, row) in destinations.iter().zip(&graph.writes) {
            assert_eq!(keys.len(), row.len());
            for (key, id) in keys.iter().zip(row) {
                assert_eq!(index.get(*key), Some(*id));
            }
        }
    }

    #[test]
    fn full_register_identity_survives_sparse_pages_and_dense_ids() {
        let mut index = RegisterIndex::new();
        let mut expected = HashMap::new();
        for offset in 0..16384u32 {
            let key = register_key(
                offset.wrapping_mul(0x9e3779b9),
                (offset.wrapping_mul(17) >> 5) as u8,
                offset.wrapping_mul(257) as u16,
            );
            assert_eq!(index.get(key), expected.get(&key).copied());
            let next = expected.len();
            let identity = *expected.entry(key).or_insert(next);
            assert_eq!(index.intern(key), identity);
            assert_eq!(index.get(key), Some(identity));
        }
        for (&key, &identity) in &expected {
            assert_eq!(index.get(key), Some(identity));
            assert_eq!(index.intern(key), identity);
        }
        assert_eq!(index.len(), expected.len());
        assert!(!index.is_empty());
    }

    #[test]
    fn graph_write_producer_and_lookup_share_full_keys() {
        let mut destinations = Vec::new();
        for file in 0..5u8 {
            for register in [0, 1, 63, 64, 65, 32768, 65535] {
                destinations.push(vec![
                    register_key(0, file, register),
                    register_key(u32::MAX, file, register),
                    register_key(0, file, register),
                ]);
            }
        }
        let (index, writes) = build_writes(&destinations);
        assert_eq!(index.len(), 70);
        for (destinations, row) in destinations.iter().zip(&writes) {
            assert_eq!(row[0], row[2]);
            assert_ne!(row[0], row[1]);
            for (&key, &identity) in destinations.iter().zip(row) {
                assert_eq!(index.get(key), Some(identity));
                assert!(identity < index.len());
            }
        }
        assert_eq!(index.get(register_key(1, 0, 0)), None);
        assert_eq!(index.get(register_key(0, 5, 0)), None);
    }
}
