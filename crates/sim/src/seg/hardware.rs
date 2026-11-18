/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ptxv_ir::FCmp;
use ptxv_ir::comparison_select::Comparator;
use ptxv_ir::comparison_select::Plan;

struct IeeeComparator;

impl Comparator for IeeeComparator {
    #[inline]
    fn compare(&self, comparison: FCmp, first: u32, second: u32) -> bool {
        crate::fp::f32_cmp(first, second, comparison, false)
    }
}

#[inline]
pub fn evaluate(plan: Plan, first: u32, second: u32) -> u32 {
    ptxv_ir::comparison_select::evaluate(plan, first, second, &IeeeComparator)
}

#[cfg(test)]
mod tests {
    use ptxv_ir::FCmp::*;

    use super::*;
    use crate::seg::ComparisonContext;
    use crate::seg::SProg;
    use crate::seg::comparison_program_parts;

    #[test]
    fn produced_hardware_bits_match_actual_segment_execution() {
        let values = [
            0,
            0x8000_0000,
            1,
            0x8000_0001,
            0x007f_ffff,
            0x807f_ffff,
            0x0080_0000,
            0x8080_0000,
            0x3f80_0000,
            0xbf80_0000,
            0x7f7f_ffff,
            0xff7f_ffff,
            0x7f80_0000,
            0xff80_0000,
            0x7f80_0001,
            0xff80_0001,
            0x7fc0_0017,
            0xffc0_0023,
        ];
        for comparison in [
            F, T, Lt, Eq, Le, Gt, Ne, Ge, Num, Nan, Ltu, Equ, Leu, Gtu, Neu, Geu,
        ] {
            for negate in [false, true] {
                for swap in [false, true] {
                    let plan = Plan {
                        comparison,
                        negate,
                        swap,
                    };
                    let (args, ops, out) = comparison_program_parts(ComparisonContext {
                        plan,
                        left: 11,
                        right: 23,
                        comparison_pc: 17,
                        selection_pc: 19,
                    });
                    assert_eq!(args, [11, 23]);
                    let program = SProg {
                        args,
                        ops,
                        out,
                        extra_pcs: Default::default(),
                    };
                    let mut scratch = vec![0xdead_beef, u64::MAX, 123];
                    for first in values {
                        for second in values {
                            assert_eq!(
                                program.eval_with(&[first, second], &mut scratch).unwrap(),
                                evaluate(plan, first, second),
                                "mode={comparison:?}, negate={negate}, swap={swap}, first={first:08x}, second={second:08x}"
                            );
                        }
                    }
                }
            }
        }
    }
}
