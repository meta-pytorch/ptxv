/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::Key;
use crate::engine::Engine;
use crate::words::WordPatch;
use crate::words::install_word;

/// Applies the numeric shadow update of a failed CAS without changing byte memory.
pub fn forget_numeric(state: &mut Engine, key: Key) {
    install_word(&mut state.cells, WordPatch { key, history: None });
    if state.track_numeric && key.space == 1 {
        state.numeric_written.push(key);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Cell;
    use crate::Origin;
    use crate::WordHistory;

    #[test]
    fn forgetting_marks_only_the_exact_numeric_key() {
        let cells: Vec<_> = (0..8)
            .map(|offset| Cell {
                key: Key {
                    space: 1,
                    allocation: 9,
                    offset,
                },
                value: offset as u8,
                origin: Some(Origin::Zero),
                tainted: offset % 2 == 0,
                defined: true,
                written: false,
                word: Some(WordHistory::Opaque(offset as u32)),
            })
            .collect();
        for track_numeric in [false, true] {
            for track_half in [false, true] {
                let mut state = Engine {
                    cells: cells.clone(),
                    numeric_written: vec![cells[0].key],
                    half_written: vec![cells[7].key],
                    track_numeric,
                    track_half,
                };
                let mut expected = state.clone();
                let key = cells[3].key;
                expected.cells[3].word = None;
                if track_numeric {
                    expected.numeric_written.push(key);
                }
                forget_numeric(&mut state, key);
                assert_eq!(state, expected);
            }
        }
    }

    #[test]
    fn non_global_and_absent_keys_preserve_marker_planes() {
        let key = Key {
            space: 0,
            allocation: 2,
            offset: 4,
        };
        let mut state = Engine {
            cells: vec![Cell {
                key,
                value: 42,
                origin: None,
                tainted: true,
                defined: false,
                written: true,
                word: Some(WordHistory::Known(3)),
            }],
            numeric_written: Vec::new(),
            half_written: Vec::new(),
            track_numeric: true,
            track_half: true,
        };
        forget_numeric(&mut state, key);
        assert_eq!(state.cells[0].word, None);
        assert!(state.numeric_written.is_empty());
        assert!(state.half_written.is_empty());
        let before = state.clone();
        forget_numeric(&mut state, Key { offset: 5, ..key });
        assert_eq!(state, before);
    }
}
