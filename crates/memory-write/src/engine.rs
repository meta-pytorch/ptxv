/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::Cell;
use crate::Key;
use crate::checked::EffectError;
use crate::checked::PreparedWrite;
use crate::checked::checked_write;

/// Owns the byte state and the two independently enabled fallback suppression sets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Engine {
    pub cells: Vec<Cell>,
    pub numeric_written: Vec<Key>,
    pub half_written: Vec<Key>,
    pub track_numeric: bool,
    pub track_half: bool,
}

pub fn record_write(
    numeric: &mut Vec<Key>,
    half: &mut Vec<Key>,
    track_numeric: bool,
    track_half: bool,
    destination: Key,
    width: usize,
) {
    if destination.space != 1 {
        return;
    }
    let mut index = 0;
    while index < width {
        let offset = destination.offset.wrapping_add(index as u64);
        if track_numeric {
            numeric.push(Key {
                offset: offset / 4 * 4,
                ..destination
            });
        }
        if track_half {
            half.push(Key {
                offset,
                ..destination
            });
        }
        index += 1;
    }
}

/// Couples an accepted write with both fallback markers in one transition.
pub fn apply(state: &mut Engine, write: &PreparedWrite) -> Result<(), EffectError> {
    match checked_write(&mut state.cells, write) {
        Ok(()) => {}
        Err(error) => return Err(error),
    }
    if write.enabled {
        record_write(
            &mut state.numeric_written,
            &mut state.half_written,
            state.track_numeric,
            state.track_half,
            write.destination,
            write.payload.len(),
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::checked::WriteClass;
    use crate::checked::capture_span;

    #[test]
    fn fallback_markers_track_each_written_byte_independently() {
        let cells: Vec<_> = (0..12)
            .map(|offset| Cell {
                key: Key {
                    space: 1,
                    allocation: 5,
                    offset,
                },
                value: offset as u8,
                origin: None,
                tainted: true,
                defined: true,
                written: false,
                word: None,
            })
            .collect();
        let write = PreparedWrite {
            class: WriteClass::Store,
            destination: cells[3].key,
            payload: capture_span(&cells, cells[0].key, 6).expect("source"),
            words: Vec::new(),
            enabled: true,
            address_dependent: false,
            guard_dependent: false,
        };
        for track_numeric in [false, true] {
            for track_half in [false, true] {
                let mut state = Engine {
                    cells: cells.clone(),
                    numeric_written: Vec::new(),
                    half_written: Vec::new(),
                    track_numeric,
                    track_half,
                };
                apply(&mut state, &write).expect("valid checked effect");
                assert_eq!(
                    state.numeric_written.len(),
                    if track_numeric { 6 } else { 0 }
                );
                assert_eq!(state.half_written.len(), if track_half { 6 } else { 0 });
                if track_numeric {
                    assert_eq!(
                        state
                            .numeric_written
                            .iter()
                            .map(|key| key.offset)
                            .collect::<Vec<_>>(),
                        vec![0, 4, 4, 4, 4, 8]
                    );
                }
                if track_half {
                    assert_eq!(
                        state
                            .half_written
                            .iter()
                            .map(|key| key.offset)
                            .collect::<Vec<_>>(),
                        vec![3, 4, 5, 6, 7, 8]
                    );
                }
                assert!(state.cells[3..9].iter().all(|cell| cell.written));
                let before = state.clone();
                let mut rejected = write.clone();
                rejected.guard_dependent = true;
                assert_eq!(
                    apply(&mut state, &rejected),
                    Err(EffectError::InputDependentGuard)
                );
                assert_eq!(state, before);
            }
        }
    }
}
