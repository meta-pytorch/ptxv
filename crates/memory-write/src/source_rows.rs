/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::symbolic::RegisterKey;

trait RegisterRows {
    fn row_count(&self) -> usize;
    fn row_len(&self, pc: usize) -> usize;
    fn key(&self, pc: usize, position: usize) -> RegisterKey;
}

fn collect<R: RegisterRows>(source: &R) -> Vec<Vec<RegisterKey>> {
    let count = source.row_count();
    let mut rows = Vec::with_capacity(count);
    let mut pc = 0;
    while pc < count {
        let length = source.row_len(pc);
        let mut row = Vec::with_capacity(length);
        let mut position = 0;
        while position < length {
            row.push(source.key(pc, position));
            position += 1;
        }
        rows.push(row);
        pc += 1;
    }
    rows
}

struct RawRows {
    rows: Vec<Vec<(u32, u8, u16)>>,
}

impl RegisterRows for RawRows {
    fn row_count(&self) -> usize {
        self.rows.len()
    }

    fn row_len(&self, pc: usize) -> usize {
        self.rows[pc].len()
    }

    fn key(&self, pc: usize, position: usize) -> RegisterKey {
        let (owner, file, index) = self.rows[pc][position];
        crate::symbolic::register_key(owner, file, index)
    }
}

pub fn from_raw(rows: Vec<Vec<(u32, u8, u16)>>) -> Vec<Vec<RegisterKey>> {
    collect(&RawRows { rows })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_rows_keep_owner_file_and_full_register_index() {
        let rows = vec![
            vec![(0, 0, 7), (u32::MAX, u8::MAX, u16::MAX)],
            vec![],
            vec![(19, 4, 64), (19, 4, 32768)],
        ];
        let converted = from_raw(rows.clone());
        assert_eq!(converted.len(), rows.len());
        for (actual, expected) in converted.iter().zip(&rows) {
            assert_eq!(actual.len(), expected.len());
            for (key, &(owner, file, index)) in actual.iter().zip(expected) {
                assert_eq!((key.thread, key.file, key.index), (owner, file, index));
            }
        }
    }
}
