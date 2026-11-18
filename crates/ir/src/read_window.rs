/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::FFmt;
use crate::Op;
use crate::Val;

/// Narrow conversion source and the interval of source bits it consumes.
pub fn narrow(op: &Op) -> Option<(&Val, u32, u32)> {
    match op {
        Op::FCvt {
            a,
            from: FFmt::F16 | FFmt::Bf16,
            hi,
            ..
        }
        | Op::CvtF32F16 { a, hi, .. } => Some((a, if *hi { 16 } else { 0 }, 16)),
        Op::I2F { a, bits, hi, .. } if *bits == 8 || *bits == 16 => {
            Some((a, if *hi { 16 } else { 0 }, *bits as u32))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::File;
    use crate::Reg;
    use crate::Rnd;

    fn r(index: u16) -> Reg {
        Reg {
            file: File::R,
            idx: index,
        }
    }

    #[test]
    fn narrow_conversions_name_the_selected_source_bits() {
        let value = Val::Reg(r(7), 1);
        for high in [false, true] {
            let start = if high { 16 } else { 0 };
            for format in [FFmt::F16, FFmt::Bf16, FFmt::F32, FFmt::F64] {
                let operation = Op::FCvt {
                    d: r(1),
                    a: value.clone(),
                    from: format,
                    to: FFmt::F64,
                    rnd: Rnd::Rn,
                    ftz: false,
                    int: false,
                    hi: high,
                };
                let expected = match format {
                    FFmt::F16 | FFmt::Bf16 => Some((&value, start, 16)),
                    FFmt::F32 | FFmt::F64 => None,
                };
                assert_eq!(narrow(&operation), expected);
            }
            for bits in 0..=255 {
                let operation = Op::I2F {
                    d: r(1),
                    a: value.clone(),
                    signed: true,
                    rnd: Rnd::Rn,
                    bits,
                    hi: high,
                    f64: false,
                };
                let expected = if bits == 8 || bits == 16 {
                    Some((&value, start, bits as u32))
                } else {
                    None
                };
                assert_eq!(narrow(&operation), expected);
            }
            let operation = Op::CvtF32F16 {
                d: r(1),
                a: value.clone(),
                hi: high,
            };
            assert_eq!(narrow(&operation), Some((&value, start, 16)));
        }
    }
}
