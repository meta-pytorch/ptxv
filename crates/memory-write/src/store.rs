/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::Cell;
use crate::Key;
use crate::PayloadByte;
use crate::WordHistory;
use crate::checked::EffectError;
use crate::checked::PreparedWrite;
use crate::checked::WriteClass;
use crate::engine::Engine;
use crate::engine::apply;
use crate::symbolic::Metadata;
use crate::symbolic::control;

pub fn materialize_opaque(mut cell: Cell, id: u32) -> Cell {
    if cell.word.is_none() {
        cell.word = Some(WordHistory::Opaque(id));
    }
    cell
}

/// Operand and control metadata frozen before the instruction changes memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WordStoreInput {
    pub space: u8,
    pub allocation: u64,
    pub source: Metadata,
    pub guard: Metadata,
    pub address_low: Metadata,
    pub address_high: Metadata,
}

fn payload_byte(value: u8, source: Metadata, word: Option<WordHistory>) -> PayloadByte {
    PayloadByte {
        value,
        origin: None,
        tainted: source.tainted,
        defined: source.defined,
        word,
    }
}

/// Constructs a four-byte effect from the same frozen data and control operands.
pub fn prepare_word(input: WordStoreInput) -> PreparedWrite {
    let source = input.source;
    let decision = control(input.guard, input.address_low, input.address_high);
    let bytes = source.captured.to_le_bytes();
    let word = match source.history {
        Some(id) => Some(WordHistory::Known(id)),
        None => None,
    };
    let mut payload = Vec::new();
    payload.push(payload_byte(bytes[0], source, word));
    payload.push(payload_byte(bytes[1], source, None));
    payload.push(payload_byte(bytes[2], source, None));
    payload.push(payload_byte(bytes[3], source, None));
    PreparedWrite {
        class: WriteClass::Store,
        destination: Key {
            space: input.space,
            allocation: input.allocation,
            offset: input.address_low.captured as u64
                | ((input.address_high.captured as u64) << 32),
        },
        payload,
        words: Vec::new(),
        enabled: decision.enabled,
        address_dependent: decision.address_dependent,
        guard_dependent: decision.guard_dependent,
    }
}

/// Commits the prepared effect and its independent fallback markers together.
pub fn commit_word(state: &mut Engine, input: WordStoreInput) -> Result<(), EffectError> {
    let write = prepare_word(input);
    apply(state, &write)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cell;
    use crate::Origin;
    use crate::symbolic::prepare;

    fn state() -> Engine {
        Engine {
            cells: (0..12)
                .map(|offset| Cell {
                    key: Key {
                        space: 1,
                        allocation: 9,
                        offset,
                    },
                    value: offset as u8,
                    origin: Some(Origin::Zero),
                    tainted: false,
                    defined: true,
                    written: false,
                    word: Some(WordHistory::Known(100 + offset as u32)),
                })
                .collect(),
            numeric_written: Vec::new(),
            half_written: Vec::new(),
            track_numeric: true,
            track_half: true,
        }
    }

    #[test]
    fn opaque_materialization_does_not_claim_a_symbolic_history() {
        let cell = state().cells[0];
        assert_eq!(materialize_opaque(cell, 256), cell);
        let missing = Cell { word: None, ..cell };
        assert_eq!(
            materialize_opaque(missing, 256),
            Cell {
                word: Some(WordHistory::Opaque(256)),
                ..missing
            }
        );
    }

    #[test]
    fn frozen_word_commits_all_planes_and_intersecting_histories() {
        let mut memory = state();
        let source = prepare(0x4433_2211, Some(256), true, true);
        commit_word(
            &mut memory,
            WordStoreInput {
                space: 1,
                allocation: 9,
                source,
                guard: prepare(1, None, false, true),
                address_low: prepare(4, None, false, true),
                address_high: prepare(0, None, false, true),
            },
        )
        .expect("independent store to live cells");
        assert_eq!(source.captured, 0x4433_2211);
        assert_eq!(
            memory.cells[4..8]
                .iter()
                .map(|cell| cell.value)
                .collect::<Vec<_>>(),
            [0x11, 0x22, 0x33, 0x44]
        );
        assert!(
            memory.cells[4..8].iter().all(|cell| {
                cell.origin.is_none() && cell.tainted && cell.defined && cell.written
            })
        );
        assert_eq!(memory.cells[0].word, Some(WordHistory::Known(100)));
        for index in [1, 2, 3, 5, 6, 7] {
            assert_eq!(memory.cells[index].word, None);
        }
        assert_eq!(memory.cells[4].word, Some(WordHistory::Known(256)));
        assert_eq!(memory.cells[8].word, Some(WordHistory::Known(108)));
        assert_eq!(memory.numeric_written.len(), 4);
        assert_eq!(memory.half_written.len(), 4);
    }

    #[test]
    fn zero_payload_and_both_address_halves_keep_dependencies() {
        let source = prepare(0, None, true, true);
        let guard = prepare(1, None, false, true);
        let low = prepare(0x89ab_cdef, None, false, true);
        let high = prepare(0x0123_4567, None, true, true);
        let write = prepare_word(WordStoreInput {
            space: 1,
            allocation: 9,
            source,
            guard,
            address_low: low,
            address_high: high,
        });
        assert_eq!(write.destination.offset, 0x0123_4567_89ab_cdef);
        assert!(write.address_dependent);
        assert!(
            write
                .payload
                .iter()
                .all(|byte| byte.value == 0 && byte.tainted)
        );
        assert!(write.payload.iter().all(|byte| byte.word.is_none()));
    }

    #[test]
    fn rejection_and_independent_disabled_store_preserve_whole_state() {
        let before = state();
        for (enabled, guard_dependent, address_dependent, expected) in [
            (0, true, false, Err(EffectError::InputDependentGuard)),
            (1, false, true, Err(EffectError::InputDependentAddress)),
            (0, false, false, Ok(())),
        ] {
            let mut memory = before.clone();
            let result = commit_word(
                &mut memory,
                WordStoreInput {
                    space: 1,
                    allocation: 9,
                    source: prepare(0, None, true, true),
                    guard: prepare(enabled, None, guard_dependent, true),
                    address_low: prepare(4, None, false, true),
                    address_high: prepare(0, None, address_dependent, true),
                },
            );
            assert_eq!(result, expected);
            assert_eq!(memory, before);
        }
    }
}
