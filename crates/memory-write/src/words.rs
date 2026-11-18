/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::Cell;
use crate::Key;
use crate::WordHistory;
use crate::checked::same_key;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WordPatch {
    pub key: Key,
    pub history: Option<WordHistory>,
}

pub fn patch_cell(mut cell: Cell, patch: WordPatch) -> Cell {
    if same_key(cell.key, patch.key) {
        cell.word = patch.history;
    }
    cell
}

pub fn install_word(cells: &mut [Cell], patch: WordPatch) {
    let mut index = 0;
    while index < cells.len() {
        cells[index] = patch_cell(cells[index], patch);
        index += 1;
    }
}

pub fn install_words(cells: &mut [Cell], patches: &[WordPatch]) {
    let mut index = 0;
    while index < patches.len() {
        install_word(cells, patches[index]);
        index += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_patches_replace_only_matching_histories() {
        let key = Key {
            space: 1,
            allocation: 7,
            offset: 3,
        };
        let byte = Cell {
            key,
            value: 41,
            origin: None,
            tainted: true,
            defined: false,
            written: false,
            word: Some(WordHistory::Known(8)),
        };
        let mut cells = vec![
            byte,
            Cell {
                key: Key { offset: 4, ..key },
                ..byte
            },
            Cell {
                key: Key { space: 0, ..key },
                ..byte
            },
            Cell {
                key: Key {
                    allocation: 9,
                    ..key
                },
                ..byte
            },
            byte,
        ];
        let before = cells.clone();
        install_words(
            &mut cells,
            &[
                WordPatch {
                    key,
                    history: Some(WordHistory::Known(12)),
                },
                WordPatch { key, history: None },
                WordPatch {
                    key,
                    history: Some(WordHistory::Opaque(23)),
                },
            ],
        );
        for index in [0, 4] {
            assert_eq!(
                cells[index],
                Cell {
                    word: Some(WordHistory::Opaque(23)),
                    ..before[index]
                }
            );
        }
        assert_eq!(&cells[1..4], &before[1..4]);
        install_word(&mut cells, WordPatch { key, history: None });
        assert_eq!(cells[0].word, None);
        assert_eq!(cells[4].word, None);
        let after = cells.clone();
        install_words(&mut cells, &[]);
        assert_eq!(cells, after);
    }
}
