/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::atomic_integer::AtomicError;

pub fn decode(bytes: &[u8]) -> Result<u64, AtomicError> {
    match bytes.len() {
        2 => Ok(u16::from_le_bytes([bytes[0], bytes[1]]) as u64),
        4 => Ok(u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as u64),
        8 => Ok(u64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ])),
        _ => Err(AtomicError::UnsupportedWidth),
    }
}

pub fn encode(value: u64) -> [u8; 8] {
    value.to_le_bytes()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn atomic_words_use_little_endian_bytes_without_sign_extension() {
        let bytes = [0xef, 0xcd, 0xab, 0x89, 0x67, 0x45, 0x23, 0x81];
        assert_eq!(decode(&bytes[..2]), Ok(0xcdef));
        assert_eq!(decode(&bytes[..4]), Ok(0x89ab_cdef));
        assert_eq!(decode(&bytes), Ok(0x8123_4567_89ab_cdef));
        assert_eq!(encode(0x8123_4567_89ab_cdef), bytes);
        for width in [0, 1, 3, 5, 6, 7] {
            assert_eq!(decode(&bytes[..width]), Err(AtomicError::UnsupportedWidth));
        }
    }
}
