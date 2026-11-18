/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

pub mod arrival_walk;
pub mod atomic;
pub mod atomic_bytes;
pub mod atomic_components;
pub mod atomic_confluence;
pub mod atomic_exec;
pub mod atomic_integer;
pub mod atomic_packed;
pub mod atomic_scalar;
pub mod barrier;
pub mod barrier_reach;
pub mod cache;
pub mod cache_graph;
pub mod checked;
pub mod clock;
pub mod definedness;
pub mod effects;
pub mod engine;
pub mod execution;
pub mod graph_source;
pub mod machine_action;
pub mod machine_control;
pub mod machine_footprint;
pub mod machine_recorder;
pub mod machine_registers;
pub mod machine_warp;
pub mod numeric;
pub mod reduction;
pub mod register_index;
pub mod source_rows;
pub mod store;
pub mod symbolic;
pub mod words;

/// A byte address qualified by address space and allocation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Key {
    pub space: u8,
    pub allocation: u64,
    pub offset: u64,
}

/// Concrete byte state and the symbolic histories that start at its address.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Origin {
    Zero,
    E { alloc: u64, elem: u64, b: u8 },
    N { node: u32, b: u8 },
    E7 { alloc: u64, elem: u64 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WordHistory {
    Known(u32),
    Opaque(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cell {
    pub key: Key,
    pub value: u8,
    pub origin: Option<Origin>,
    pub tainted: bool,
    pub defined: bool,
    pub written: bool,
    /// A four-byte history starting at this cell's address.
    pub word: Option<WordHistory>,
}

/// One byte of an immutable payload prepared before any destination changes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PayloadByte {
    pub value: u8,
    pub origin: Option<Origin>,
    pub tainted: bool,
    pub defined: bool,
    /// Used only when the write contains this entire four-byte word.
    pub word: Option<WordHistory>,
}

/// Updates a byte and invalidates or replaces every history intersected by the write.
pub fn cell_after_write(mut cell: Cell, write: Key, payload: &[PayloadByte]) -> Cell {
    if cell.key.space != write.space || cell.key.allocation != write.allocation {
        return cell;
    }
    let start = write.offset as u128;
    let end = start + payload.len() as u128;
    let address = cell.key.offset as u128;
    if !payload.is_empty() && address < end && start < address + 4 {
        cell.word = None;
    }
    if start <= address && address < end {
        let index = (address - start) as usize;
        let byte = payload[index];
        cell.value = byte.value;
        cell.origin = byte.origin;
        cell.tainted = byte.tainted;
        cell.defined = byte.defined;
        cell.written = true;
        if address + 4 <= end {
            cell.word = byte.word;
        }
    }
    cell
}

/// Applies the prepared write to all entries, including histories starting before it.
pub fn apply_write_effect(cells: &mut [Cell], write: Key, payload: &[PayloadByte]) {
    let mut index = 0;
    while index < cells.len() {
        cells[index] = cell_after_write(cells[index], write, payload);
        index += 1;
    }
}

pub fn snapshot_cells(source: &[Cell]) -> Vec<PayloadByte> {
    let mut payload = Vec::new();
    let mut index = 0;
    while index < source.len() {
        let cell = source[index];
        payload.push(PayloadByte {
            value: cell.value,
            origin: cell.origin,
            tainted: cell.tainted,
            defined: cell.defined,
            word: cell.word,
        });
        index += 1;
    }
    payload
}

pub fn apply_copy_effect(cells: &mut [Cell], write: Key, source: &[Cell]) {
    let payload = snapshot_cells(source);
    apply_write_effect(cells, write, &payload);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cell(offset: u64) -> Cell {
        Cell {
            key: Key {
                space: 1,
                allocation: 0,
                offset,
            },
            value: offset as u8,
            origin: Some(Origin::E {
                alloc: 0,
                elem: offset / 2,
                b: (offset % 2) as u8,
            }),
            tainted: true,
            defined: true,
            written: false,
            word: Some(WordHistory::Known(offset as u32)),
        }
    }

    fn unknown(count: usize) -> Vec<PayloadByte> {
        vec![
            PayloadByte {
                value: 0,
                origin: None,
                tainted: true,
                defined: true,
                word: None,
            };
            count
        ]
    }

    #[test]
    fn byte_intersections_determine_updates() {
        for start in (0..12).chain([u64::MAX - 3, u64::MAX]) {
            for offset in (0..16).chain([u64::MAX - 3, u64::MAX]) {
                for width in [0, 1, 2, 4, 8, 16, 255] {
                    for space in [0, 1] {
                        for allocation in [0, 1] {
                            let before = cell(offset);
                            let write = Key {
                                space,
                                allocation,
                                offset: start,
                            };
                            let payload: Vec<_> = (0..width)
                                .map(|i| PayloadByte {
                                    value: (i + 20) as u8,
                                    origin: Some(Origin::N {
                                        node: 1000 + i as u32,
                                        b: 0,
                                    }),
                                    tainted: false,
                                    defined: true,
                                    word: Some(WordHistory::Known(2000 + i as u32)),
                                })
                                .collect();
                            let after = cell_after_write(before, write, &payload);
                            let position = |address: u128| {
                                (space == 1 && allocation == 0)
                                    .then(|| {
                                        (0..width).find(|&i| start as u128 + i as u128 == address)
                                    })
                                    .flatten()
                            };
                            let lanes: Vec<_> =
                                (0..4).map(|lane| position(offset as u128 + lane)).collect();
                            let expected_word = if lanes.iter().all(Option::is_some) {
                                payload[lanes[0].unwrap()].word
                            } else if lanes.iter().any(Option::is_some) {
                                None
                            } else {
                                before.word
                            };
                            assert_eq!(after.word, expected_word, "{write:?} at {offset}");
                            assert_eq!(after.key, before.key);
                            match position(offset as u128) {
                                Some(i) => {
                                    assert_eq!(after.value, payload[i].value);
                                    assert_eq!(after.origin, payload[i].origin);
                                    assert_eq!(after.tainted, payload[i].tainted);
                                    assert_eq!(after.defined, payload[i].defined);
                                    assert!(after.written);
                                }
                                None => assert_eq!(
                                    after,
                                    Cell {
                                        word: expected_word,
                                        ..before
                                    }
                                ),
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn wide_write_and_offset_two_clobber_every_intersecting_history() {
        for (offset, width, removed) in [(0, 8, vec![0, 4]), (2, 2, vec![0])] {
            let mut cells: Vec<_> = (0..12).map(cell).collect();
            let write = Key {
                offset,
                ..cells[0].key
            };
            apply_write_effect(&mut cells, write, &unknown(width));
            for index in [0, 4, 8] {
                assert_eq!(
                    cells[index].word,
                    (!removed.contains(&index)).then_some(WordHistory::Known(index as u32))
                );
            }
            for (index, byte) in cells.iter().enumerate() {
                let written = index >= offset as usize && index < offset as usize + width;
                assert_eq!(byte.written, written);
                if written {
                    assert_eq!(byte.origin, None);
                    assert!(byte.tainted, "captured zero must remain input dependent");
                }
            }
        }
    }

    #[test]
    fn copy_uses_a_snapshot_when_source_and_destination_overlap() {
        let mut cells: Vec<_> = (0..20).map(cell).collect();
        let before = cells.clone();
        let snapshot = snapshot_cells(&cells[4..12]);
        let write = Key {
            offset: 6,
            ..cells[0].key
        };
        apply_write_effect(&mut cells, write, &snapshot);
        for index in 0..8 {
            assert_eq!(cells[6 + index].value, before[4 + index].value);
            assert_eq!(cells[6 + index].origin, before[4 + index].origin);
        }
        assert_eq!(cells[6].word, before[4].word);
        assert_eq!(cells[10].word, before[8].word);
        assert_eq!(cells[11].word, None);
        assert_eq!(cells[14], before[14]);
    }

    #[test]
    fn delayed_copy_keeps_issue_time_metadata_after_source_changes() {
        let mut cells: Vec<_> = (0..20).map(cell).collect();
        cells[5].defined = false;
        let issued = cells.clone();
        let snapshot = snapshot_cells(&cells[4..12]);
        let source = cells[4].key;
        apply_write_effect(&mut cells, source, &unknown(8));
        assert_ne!(cells[4].origin, issued[4].origin);
        let write = Key {
            offset: 10,
            ..source
        };
        apply_write_effect(&mut cells, write, &snapshot);
        for index in 0..8 {
            let before = issued[4 + index];
            let after = cells[10 + index];
            assert_eq!(after.value, before.value);
            assert_eq!(after.origin, before.origin);
            assert_eq!(after.tainted, before.tainted);
            assert_eq!(after.defined, before.defined);
            assert_eq!(after.word, if index + 4 <= 8 { before.word } else { None });
        }
        let mut direct = issued.clone();
        apply_copy_effect(&mut direct, write, &issued[4..12]);
        assert_eq!(&direct[10..18], &cells[10..18]);
        assert!(snapshot_cells(&[]).is_empty());
    }

    #[test]
    fn empty_write_is_identity_and_undefined_payload_stays_unknown() {
        let mut cells: Vec<_> = (0..8).map(cell).collect();
        let before = cells.clone();
        let write = cells[0].key;
        apply_write_effect(&mut cells, write, &[]);
        assert_eq!(cells, before);
        let mut payload = unknown(4);
        payload[1].defined = false;
        apply_write_effect(&mut cells, write, &payload);
        assert!(!cells[1].defined);
        assert_eq!(cells[0].word, None);
        assert_eq!(cells[1].origin, None);
    }
}

pub mod footprint;
