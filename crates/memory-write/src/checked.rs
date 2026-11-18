/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::Cell;
use crate::Key;
use crate::PayloadByte;
use crate::apply_write_effect;
use crate::snapshot_cells;
use crate::words::WordPatch;
use crate::words::install_words;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WriteClass {
    Store,
    LocalStore,
    TensorStore,
    SharedTransfer,
    GlobalTransfer,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EffectError {
    InputDependentAddress,
    InputDependentGuard,
    WrongSpace,
    OutOfBounds,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreparedWrite {
    pub class: WriteClass,
    pub destination: Key,
    pub payload: Vec<PayloadByte>,
    pub words: Vec<WordPatch>,
    pub enabled: bool,
    pub address_dependent: bool,
    pub guard_dependent: bool,
}

pub fn class_allows(class: WriteClass, space: u8) -> bool {
    match class {
        WriteClass::Store => space == 0 || space == 1,
        WriteClass::LocalStore => space == 4,
        WriteClass::TensorStore => space == 3,
        WriteClass::SharedTransfer => space == 0,
        WriteClass::GlobalTransfer => space == 1,
    }
}

pub fn same_key(left: Key, right: Key) -> bool {
    left.space == right.space && left.allocation == right.allocation && left.offset == right.offset
}

pub fn find_cell(cells: &[Cell], key: Key) -> Option<Cell> {
    let mut index = 0;
    while index < cells.len() {
        let cell = cells[index];
        if same_key(cell.key, key) {
            return Some(cell);
        }
        index += 1;
    }
    None
}

/// Checks both the nonwrapping address range and every byte's live inventory entry.
pub fn span_fits(cells: &[Cell], start: Key, width: usize) -> bool {
    if start.offset as u128 + width as u128 > u64::MAX as u128 + 1 {
        return false;
    }
    let mut index = 0;
    while index < width {
        let key = Key {
            space: start.space,
            allocation: start.allocation,
            offset: start.offset + index as u64,
        };
        if find_cell(cells, key).is_none() {
            return false;
        }
        index += 1;
    }
    true
}

/// Freezes source bits and metadata together, before any overlapping destination write.
pub fn capture_span(cells: &[Cell], start: Key, width: usize) -> Option<Vec<PayloadByte>> {
    if !span_fits(cells, start, width) {
        return None;
    }
    let mut source = Vec::new();
    let mut index = 0;
    while index < width {
        let key = Key {
            space: start.space,
            allocation: start.allocation,
            offset: start.offset + index as u64,
        };
        match find_cell(cells, key) {
            Some(cell) => source.push(cell),
            None => return None,
        }
        index += 1;
    }
    Some(snapshot_cells(&source))
}

/// An error leaves the complete byte and history state untouched.
pub fn checked_write(cells: &mut [Cell], write: &PreparedWrite) -> Result<(), EffectError> {
    if write.address_dependent {
        return Err(EffectError::InputDependentAddress);
    }
    if write.guard_dependent {
        return Err(EffectError::InputDependentGuard);
    }
    if !write.enabled {
        return Ok(());
    }
    if !class_allows(write.class, write.destination.space) {
        return Err(EffectError::WrongSpace);
    }
    if !span_fits(cells, write.destination, write.payload.len()) {
        return Err(EffectError::OutOfBounds);
    }
    apply_write_effect(cells, write.destination, &write.payload);
    install_words(cells, &write.words);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Origin;
    use crate::WordHistory;

    fn state() -> Vec<Cell> {
        (0..12)
            .map(|offset| Cell {
                key: Key {
                    space: 1,
                    allocation: 9,
                    offset,
                },
                value: offset as u8,
                origin: Some(Origin::N {
                    node: 7,
                    b: (offset % 4) as u8,
                }),
                tainted: true,
                defined: offset != 3,
                written: false,
                word: Some(WordHistory::Known(7)),
            })
            .collect()
    }

    fn request(cells: &[Cell]) -> PreparedWrite {
        PreparedWrite {
            class: WriteClass::Store,
            destination: cells[4].key,
            payload: capture_span(cells, cells[0].key, 4).expect("live source span"),
            words: Vec::new(),
            enabled: true,
            address_dependent: false,
            guard_dependent: false,
        }
    }

    #[test]
    fn rejected_writes_leave_every_plane_unchanged() {
        let before = state();
        let valid = request(&before);
        let mut cases = Vec::new();
        let mut dependent = valid.clone();
        dependent.address_dependent = true;
        cases.push((dependent, EffectError::InputDependentAddress));
        let mut guarded = valid.clone();
        guarded.guard_dependent = true;
        cases.push((guarded.clone(), EffectError::InputDependentGuard));
        guarded.enabled = false;
        cases.push((guarded, EffectError::InputDependentGuard));
        let mut wrong_space = valid.clone();
        wrong_space.class = WriteClass::LocalStore;
        cases.push((wrong_space, EffectError::WrongSpace));
        let mut missing = valid.clone();
        missing.destination.offset = 10;
        cases.push((missing, EffectError::OutOfBounds));
        let mut wrapping = valid;
        wrapping.destination.offset = u64::MAX;
        cases.push((wrapping, EffectError::OutOfBounds));
        for (write, error) in cases {
            let mut cells = before.clone();
            assert_eq!(checked_write(&mut cells, &write), Err(error));
            assert_eq!(cells, before);
        }
    }

    #[test]
    fn capture_is_ordered_and_frozen_across_clobber() {
        let mut cells = state();
        cells[0].word = Some(WordHistory::Opaque(42));
        cells.reverse();
        let source = Key {
            space: 1,
            allocation: 9,
            offset: 0,
        };
        let payload = capture_span(&cells, source, 8).expect("permuted live inventory");
        assert_eq!(
            payload.iter().map(|byte| byte.value).collect::<Vec<_>>(),
            (0..8).collect::<Vec<_>>()
        );
        let write = PreparedWrite {
            class: WriteClass::Store,
            destination: Key {
                offset: 2,
                ..source
            },
            payload,
            words: Vec::new(),
            enabled: true,
            address_dependent: false,
            guard_dependent: false,
        };
        checked_write(&mut cells, &write).expect("overlapping checked write");
        let copied = capture_span(&cells, write.destination, 8).expect("destination");
        assert_eq!(copied[0].value, 0);
        assert_eq!(copied[3].value, 3);
        assert!(!copied[3].defined);
        assert!(copied[0].tainted);
        assert_eq!(copied[0].origin, Some(Origin::N { node: 7, b: 0 }));
        assert_eq!(copied[0].word, Some(WordHistory::Opaque(42)));
        assert_eq!(copied[4].word, Some(WordHistory::Known(7)));
        assert_eq!(copied[5].word, None);
    }

    #[test]
    fn missing_source_and_wrapping_ranges_are_rejected() {
        let mut cells = state();
        let start = cells[0].key;
        cells.remove(2);
        assert!(capture_span(&cells, start, 4).is_none());
        assert!(
            capture_span(
                &cells,
                Key {
                    offset: u64::MAX,
                    ..start
                },
                2
            )
            .is_none()
        );
        assert_eq!(capture_span(&cells, start, 0), Some(Vec::new()));
        let mut disabled = request(&state());
        disabled.enabled = false;
        disabled.class = WriteClass::LocalStore;
        let before = cells.clone();
        assert_eq!(checked_write(&mut cells, &disabled), Ok(()));
        assert_eq!(cells, before);
    }

    #[test]
    fn postword_updates_follow_byte_invalidation_and_respect_rejection() {
        let before = state();
        let mut write = request(&before);
        write.words = vec![
            WordPatch {
                key: before[6].key,
                history: Some(WordHistory::Opaque(42)),
            },
            WordPatch {
                key: before[1].key,
                history: None,
            },
        ];
        let mut cells = before.clone();
        checked_write(&mut cells, &write).expect("checked write with postword metadata");
        assert_eq!(cells[6].word, Some(WordHistory::Opaque(42)));
        assert_eq!(cells[1].word, None);
        assert_eq!(cells[1].value, before[1].value);
        assert_eq!(cells[1].origin, before[1].origin);
        assert_eq!(cells[1].written, before[1].written);
        write.enabled = false;
        let mut disabled = before.clone();
        checked_write(&mut disabled, &write).expect("disabled write");
        assert_eq!(disabled, before);
        write.enabled = true;
        write.address_dependent = true;
        assert_eq!(
            checked_write(&mut disabled, &write),
            Err(EffectError::InputDependentAddress)
        );
        assert_eq!(disabled, before);
    }
}
