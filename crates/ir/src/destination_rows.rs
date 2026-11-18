/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::Inst;
use crate::Op;
use crate::Reg;
use crate::register_effect;

pub trait LegacyDestinations {
    fn row(&self, operation: &Op) -> Vec<Reg>;
}

pub fn collect<F: LegacyDestinations>(code: &[Inst], fallback: &F) -> Vec<Vec<Reg>> {
    let mut rows = Vec::with_capacity(code.len());
    let mut pc = 0;
    while pc < code.len() {
        let operation = &code[pc].op;
        let row = match register_effect::memory_writes(operation) {
            Some(registers) => registers,
            None => fallback.row(operation),
        };
        rows.push(row);
        pc += 1;
    }
    rows
}

pub fn keys(rows: &[Vec<Reg>]) -> Vec<Vec<(u32, u8, u16)>> {
    let mut result = Vec::with_capacity(rows.len());
    let mut pc = 0;
    while pc < rows.len() {
        let mut row = Vec::with_capacity(rows[pc].len());
        let mut position = 0;
        while position < rows[pc].len() {
            row.push(crate::register_owner::canonical_key(
                0,
                &[],
                rows[pc][position],
            ));
            position += 1;
        }
        result.push(row);
        pc += 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Addr;
    use crate::File;
    use crate::Pred;
    use crate::Space;

    struct Legacy;

    impl LegacyDestinations for Legacy {
        fn row(&self, operation: &Op) -> Vec<Reg> {
            assert!(matches!(operation, Op::Nop));
            vec![Reg {
                file: File::UP,
                idx: u16::MAX,
            }]
        }
    }

    #[test]
    fn selected_memory_rows_ignore_legacy_and_instruction_guards() {
        let destination = Reg {
            file: File::UR,
            idx: 1023,
        };
        let load = Op::Ld {
            d: destination,
            n: 1,
            space: Space::Global,
            bytes: 4,
            addr: Addr {
                base: None,
                base2: None,
                off: 0,
            },
            sync: None,
            nc: false,
            sext: false,
        };
        let instruction = |op, guard| Inst {
            op,
            guard,
            uniform: false,
            loc: 0,
            text: String::new(),
        };
        let code = vec![
            instruction(load.clone(), Pred::TRUE),
            instruction(
                load,
                Pred {
                    reg: Some(Reg {
                        file: File::P,
                        idx: 4,
                    }),
                    neg: true,
                },
            ),
            instruction(Op::Nop, Pred::TRUE),
        ];
        assert_eq!(
            keys(&collect(&code, &Legacy)),
            vec![
                vec![(0, File::UR as u8, 1023)],
                vec![(0, File::UR as u8, 1023)],
                vec![(0, File::UP as u8, u16::MAX)],
            ]
        );
        assert_eq!(
            collect(&code, &Legacy),
            vec![
                vec![destination],
                vec![destination],
                vec![Reg {
                    file: File::UP,
                    idx: u16::MAX
                }],
            ]
        );
    }
}
