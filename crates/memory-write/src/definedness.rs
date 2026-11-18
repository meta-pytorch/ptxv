/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

/// Whether a consumed bit is undefined; bits past the 64-bit mask are defined.
#[inline]
pub fn selected_undefined(mask: u64, start: u32, width: u32) -> bool {
    if start >= 64 {
        return false;
    }
    let selected = mask >> start;
    if width >= 64 {
        selected != 0
    } else {
        selected & ((1u64 << width) - 1) != 0
    }
}

/// One undefinedness bit per source byte, for at most eight bytes.
#[inline]
pub fn byte_mask(mask: u64, count: u32) -> u64 {
    let mut result = 0;
    let mut byte = 0;
    while byte < count && byte < 8 {
        if selected_undefined(mask, 8 * byte, 8) {
            result |= 1 << byte;
        }
        byte += 1;
    }
    result
}

/// Add a register's byte mask at its destination offset, clipping at 64 bytes.
pub fn merge_byte_mask(previous: u64, source: u64, first: u32, count: u32) -> u64 {
    if first >= 64 {
        previous
    } else {
        previous | (byte_mask(source, count) << first)
    }
}

/// An unordered old write can still leave the byte undefined after a defined write.
pub fn written_undefined(source: bool, previous: bool, unordered: bool) -> bool {
    source || unordered && previous
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selections_ignore_unconsumed_bits() {
        for start in 0..=65 {
            for width in 0..=65 {
                for bit in 0..64 {
                    assert_eq!(
                        selected_undefined(1 << bit, start, width),
                        start <= bit && bit - start < width,
                        "start={start}, width={width}, bit={bit}"
                    );
                }
            }
        }
        assert!(!selected_undefined(0xffff_0000, 0, 16));
        assert!(!selected_undefined(0x0000_ffff, 16, 16));
        assert!(!selected_undefined(0x0000_ff00, 0, 8));
        assert!(!selected_undefined(0xff00_0000, 16, 8));
    }

    #[test]
    fn each_store_byte_uses_only_its_source_bits() {
        for count in 0..=65 {
            for bit in 0..64 {
                let expected = if bit / 8 < count { 1 << (bit / 8) } else { 0 };
                assert_eq!(byte_mask(1 << bit, count), expected);
            }
        }
        assert_eq!(byte_mask(0xffff_ffff_0000_0000, 4), 0);
        assert_eq!(byte_mask(0x0000_ffff_0000_ffff, 8), 0b0011_0011);
    }

    #[test]
    fn ordered_writes_define_bytes_independently_of_old_state() {
        assert!(!written_undefined(false, true, false));
        assert!(written_undefined(false, true, true));
        assert!(written_undefined(true, false, false));
    }

    #[test]
    fn vector_registers_only_poison_the_bytes_they_supply() {
        for first in 0..=65 {
            for count in 0..=9 {
                for bit in 0..64 {
                    let mut expected = 1;
                    if bit / 8 < count && first < 64 && first + bit / 8 < 64 {
                        expected |= 1 << (first + bit / 8);
                    }
                    assert_eq!(merge_byte_mask(1, 1 << bit, first, count), expected);
                }
            }
        }
        let first = merge_byte_mask(0, 0, 0, 4);
        let second = merge_byte_mask(first, u32::MAX as u64, 4, 4);
        assert_eq!(second, 0xf0);
        assert_eq!(merge_byte_mask(second, 0, 8, 4), 0xf0);
    }
}
