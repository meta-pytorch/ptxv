/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ptxv_ir::FCmp;

#[inline]
fn classify(bits: u32, flush: bool) -> (i32, bool) {
    let negative = bits >= 0x8000_0000;
    let magnitude = bits & 0x7fff_ffff;
    let magnitude = if flush && magnitude < 0x0080_0000 {
        0
    } else {
        magnitude
    };
    let unordered = magnitude > 0x7f80_0000;
    let magnitude = magnitude as i32;
    let key = if negative { -magnitude } else { magnitude };
    (key, unordered)
}

#[inline]
pub fn compare(first: u32, second: u32, mode: FCmp, flush: bool) -> bool {
    let (first, first_nan) = classify(first, flush);
    let (second, second_nan) = classify(second, flush);
    let unordered = first_nan || second_nan;
    match mode {
        FCmp::F => false,
        FCmp::T => true,
        FCmp::Lt => !unordered && first < second,
        FCmp::Eq => !unordered && first == second,
        FCmp::Le => !unordered && first <= second,
        FCmp::Gt => !unordered && first > second,
        FCmp::Ne => !unordered && first != second,
        FCmp::Ge => !unordered && first >= second,
        FCmp::Num => !unordered,
        FCmp::Nan => unordered,
        FCmp::Ltu => unordered || first < second,
        FCmp::Equ => unordered || first == second,
        FCmp::Leu => unordered || first <= second,
        FCmp::Gtu => unordered || first > second,
        FCmp::Neu => unordered || first != second,
        FCmp::Geu => unordered || first >= second,
    }
}

#[cfg(test)]
mod tests {
    use ptxv_ir::FCmp::*;

    use super::*;

    const MODES: [FCmp; 16] = [
        F, T, Lt, Eq, Le, Gt, Ne, Ge, Num, Nan, Ltu, Equ, Leu, Gtu, Neu, Geu,
    ];

    fn legacy(first: u32, second: u32, mode: FCmp, flush: bool) -> bool {
        let fl = |u: u32| if flush { crate::fp::f32_ftz(u) } else { u };
        let (x, y) = (f32::from_bits(fl(first)), f32::from_bits(fl(second)));
        let un = x.is_nan() || y.is_nan();
        match mode {
            F => false,
            T => true,
            Lt => x < y,
            Eq => x == y,
            Le => x <= y,
            Gt => x > y,
            Ne => !un && x != y,
            Ge => x >= y,
            Num => !un,
            Nan => un,
            Ltu => un || x < y,
            Equ => un || x == y,
            Leu => un || x <= y,
            Gtu => un || x > y,
            Neu => un || x != y,
            Geu => un || x >= y,
        }
    }

    fn check(first: u32, second: u32) {
        for mode in MODES {
            for flush in [false, true] {
                let expected = legacy(first, second, mode, flush);
                assert_eq!(
                    compare(first, second, mode, flush),
                    expected,
                    "first={first:08x}, second={second:08x}, mode={mode:?}, FTZ={flush}"
                );
                assert_eq!(crate::fp::f32_cmp(first, second, mode, flush), expected);
            }
        }
    }

    #[test]
    fn ieee_classes_match_original_host_comparison() {
        let mut values = Vec::new();
        for sign in [0, 0x8000_0000] {
            for exponent in [0, 1, 126, 127, 128, 253, 254, 255] {
                for fraction in [0, 1, 0x3f_ffff, 0x7f_fffe, 0x7f_ffff] {
                    values.push(sign | (exponent << 23) | fraction);
                }
            }
        }
        for &first in &values {
            for &second in &values {
                check(first, second);
            }
        }
    }

    #[test]
    fn arbitrary_words_match_original_host_comparison() {
        let mut state = 0x1234_5678_9abc_def0_u64;
        let random = |state: &mut u64| {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            *state as u32
        };
        for _ in 0..200_000 {
            check(random(&mut state), random(&mut state));
        }
    }

    #[test]
    fn signed_zero_ftz_nan_and_infinity_have_ieee_predicates() {
        assert!(compare(0, 0x8000_0000, Eq, false));
        assert!(!compare(0x8000_0000, 0, Lt, false));
        assert!(!compare(0, 0x8000_0000, Gt, false));
        assert!(compare(0x8000_0001, 0, Lt, false));
        assert!(compare(0x8000_0001, 0, Eq, true));
        assert!(compare(1, 0x8000_0001, Eq, true));
        assert!(compare(0xff80_0000, 0x7f80_0000, Lt, false));
        assert!(compare(0x7f80_0000, 0x7f80_0000, Num, false));
        for nan in [0x7f80_0001, 0x7fc0_0000, 0xffff_ffff] {
            assert!(compare(nan, 0, Nan, false));
            assert!(!compare(nan, 0, Ne, false));
            assert!(compare(nan, 0, Neu, false));
            assert!(!compare(nan, nan, Eq, false));
            assert!(compare(nan, nan, T, true));
        }
    }
}
