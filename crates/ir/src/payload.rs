/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::File;
use crate::Front;
use crate::Op;
use crate::Reg;

pub trait RegisterReader {
    type Error;

    fn read(&self, register: Reg) -> Result<u64, Self::Error>;
}

fn assemble<R: RegisterReader>(
    windows: &[(Reg, u32, u32)],
    total: u8,
    reader: &R,
) -> Result<Vec<u8>, R::Error> {
    let mut output = Vec::new();
    let mut index = 0;
    while index < total {
        output.push(0);
        index += 1;
    }
    let mut index = 0;
    while index < windows.len() {
        let (register, first, count) = windows[index];
        let value = match reader.read(register) {
            Ok(value) => value,
            Err(error) => return Err(error),
        };
        let mut byte = 0;
        while byte < count {
            output[(first + byte) as usize] = (value >> (8 * byte)) as u8;
            byte += 1;
        }
        index += 1;
    }
    Ok(output)
}

fn front_supported(front: Front, file: File) -> bool {
    match front {
        Front::Ptx => true,
        Front::Sass => file != File::V,
    }
}

fn width_supported(file: File, elements: u8, bytes: u8, exchange: bool) -> bool {
    if exchange {
        bytes == 4 || bytes == 8
    } else if file == File::V && elements >= 2 {
        bytes % elements == 0 && bytes / elements <= 8
    } else {
        bytes <= 8 || bytes % 4 == 0
    }
}

fn supported_total(front: Front, operation: &Op) -> Option<u8> {
    let (first, elements, bytes, exchange) = match crate::store_window::descriptor(operation) {
        Some(descriptor) => descriptor,
        None => return None,
    };
    if front_supported(front, first.file) && width_supported(first.file, elements, bytes, exchange)
    {
        Some(bytes)
    } else {
        None
    }
}

/// `None` selects the executor's existing path for unsupported operand layouts.
pub fn from_operation<R: RegisterReader>(
    front: Front,
    operation: &Op,
    reader: &R,
) -> Option<Result<Vec<u8>, R::Error>> {
    match supported_total(front, operation) {
        Some(total) => {
            let windows = crate::store_window::sources(front, operation);
            Some(assemble(&windows, total, reader))
        }
        None => None,
    }
}

fn st256_sources(operation: &Op) -> Option<Vec<(Reg, u32, u32)>> {
    let Op::St256 { lo, hi, .. } = operation else {
        return None;
    };
    let mut windows = Vec::new();
    let mut index = 0u16;
    while index < 8 {
        let first = if index < 4 { *lo } else { *hi };
        windows.push((
            Reg {
                file: first.file,
                idx: first.idx + index % 4,
            },
            4 * index as u32,
            4,
        ));
        index += 1;
    }
    Some(windows)
}

pub fn from_st256<R: RegisterReader>(
    operation: &Op,
    reader: &R,
) -> Option<Result<Vec<u8>, R::Error>> {
    let windows = match st256_sources(operation) {
        Some(windows) => windows,
        None => return None,
    };
    Some(assemble(&windows, 32, reader))
}

#[cfg(test)]
mod tests {
    use std::cell::RefCell;

    use super::*;
    use crate::AtomOp;
    use crate::Space;
    use crate::Val;

    struct Fixture {
        reads: RefCell<Vec<Reg>>,
    }

    impl RegisterReader for Fixture {
        type Error = ();

        fn read(&self, register: Reg) -> Result<u64, ()> {
            self.reads.borrow_mut().push(register);
            let value = 0x8123_4567_89ab_cdefu64.wrapping_mul(register.idx as u64 + 1);
            Ok(if register.file == File::V {
                value
            } else {
                value as u32 as u64
            })
        }
    }

    fn value(reader: &Fixture, first: Reg, elements: u8) -> u64 {
        let low = reader.read(first).unwrap();
        if first.file != File::V && elements >= 2 {
            let high = reader
                .read(Reg {
                    idx: first.idx + 1,
                    ..first
                })
                .unwrap();
            low | (high << 32)
        } else {
            low
        }
    }

    fn legacy_store(reader: &Fixture, first: Reg, elements: u8, bytes: u8) -> Vec<u8> {
        if first.file == File::V && elements >= 2 {
            let width = (bytes / elements) as usize;
            (0..elements)
                .flat_map(|index| {
                    reader
                        .read(Reg {
                            idx: first.idx + index as u16,
                            ..first
                        })
                        .unwrap()
                        .to_le_bytes()[..width]
                        .to_vec()
                })
                .collect()
        } else if bytes > 8 {
            (0..bytes / 4)
                .flat_map(|index| {
                    (reader
                        .read(Reg {
                            idx: first.idx + index as u16,
                            ..first
                        })
                        .unwrap() as u32)
                        .to_le_bytes()
                })
                .collect()
        } else {
            value(reader, first, elements).to_le_bytes()[..bytes as usize].to_vec()
        }
    }

    #[test]
    fn unsupported_layouts_select_the_existing_executor_path() {
        for (front, elements, bytes) in [(Front::Sass, 1, 4), (Front::Ptx, 2, 32)] {
            let operation = Op::St {
                space: Space::Shared,
                bytes,
                addr: crate::Addr {
                    base: None,
                    base2: None,
                    off: 0,
                },
                v: Val::Reg(
                    Reg {
                        file: File::V,
                        idx: 16,
                    },
                    elements,
                ),
                sync: None,
            };
            let reader = Fixture {
                reads: RefCell::new(Vec::new()),
            };
            assert!(from_operation(front, &operation, &reader).is_none());
            assert!(reader.reads.borrow().is_empty());
        }
    }

    #[test]
    fn payload_plan_preserves_existing_store_values_and_reads() {
        for front in [Front::Ptx, Front::Sass] {
            for file in [File::V, File::R, File::UR] {
                for elements in [0, 1, 2, 4, 16] {
                    for bytes in [0, 1, 2, 4, 8, 16, 32] {
                        let first = Reg { file, idx: 16 };
                        let operation = Op::St {
                            space: Space::Shared,
                            bytes,
                            addr: crate::Addr {
                                base: None,
                                base2: None,
                                off: 0,
                            },
                            v: Val::Reg(first, elements),
                            sync: None,
                        };
                        let actual = Fixture {
                            reads: RefCell::new(Vec::new()),
                        };
                        let Some(result) = from_operation(front, &operation, &actual) else {
                            continue;
                        };
                        let expected = Fixture {
                            reads: RefCell::new(Vec::new()),
                        };
                        assert_eq!(
                            result.unwrap(),
                            legacy_store(&expected, first, elements, bytes),
                            "{front:?} {file:?} {elements} {bytes}"
                        );
                        assert_eq!(
                            actual.reads.into_inner(),
                            expected.reads.into_inner(),
                            "{front:?} {file:?} {elements} {bytes}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn exchange_payload_preserves_the_selected_scalar_bytes() {
        for file in [File::V, File::R, File::UR] {
            for elements in [0, 1, 2, 4] {
                for bytes in [4, 8] {
                    let first = Reg { file, idx: 16 };
                    let operation = Op::Atom {
                        op: AtomOp::Exch,
                        space: Space::Shared,
                        bytes,
                        addr: crate::Addr {
                            base: None,
                            base2: None,
                            off: 0,
                        },
                        v: Val::Reg(first, elements),
                        d: None,
                        sync: crate::MemSync {
                            sem: crate::Sem::Relaxed,
                            scope: crate::Scope::Cta,
                        },
                        p: None,
                        new: None,
                    };
                    let actual = Fixture {
                        reads: RefCell::new(Vec::new()),
                    };
                    let expected = Fixture {
                        reads: RefCell::new(Vec::new()),
                    };
                    let output = from_operation(Front::Ptx, &operation, &actual)
                        .unwrap()
                        .unwrap();
                    assert_eq!(
                        output,
                        value(&expected, first, elements).to_le_bytes()[..bytes as usize]
                    );
                    assert_eq!(actual.reads.into_inner(), expected.reads.into_inner());
                }
            }
        }
    }
}

#[cfg(test)]
mod error_tests {
    use super::*;

    struct FailAt(u16);

    impl RegisterReader for FailAt {
        type Error = u16;

        fn read(&self, register: Reg) -> Result<u64, u16> {
            if register.idx == self.0 {
                Err(register.idx)
            } else {
                Ok(0)
            }
        }
    }

    #[test]
    fn payload_preserves_errors_from_consumed_registers_without_output_bytes() {
        let operation = Op::St {
            space: crate::Space::Shared,
            bytes: 1,
            addr: crate::Addr {
                base: None,
                base2: None,
                off: 0,
            },
            v: crate::Val::Reg(
                Reg {
                    file: File::R,
                    idx: 16,
                },
                2,
            ),
            sync: None,
        };
        assert_eq!(
            from_operation(Front::Sass, &operation, &FailAt(17)),
            Some(Err(17))
        );
    }
}

#[cfg(test)]
mod st256_tests {
    use std::cell::RefCell;

    use super::*;

    struct Reader {
        calls: RefCell<Vec<Reg>>,
        fail_at: usize,
    }

    impl RegisterReader for Reader {
        type Error = usize;

        fn read(&self, register: Reg) -> Result<u64, usize> {
            let mut calls = self.calls.borrow_mut();
            let index = calls.len();
            calls.push(register);
            if index == self.fail_at {
                Err(index)
            } else {
                Ok(0xa9b8_c7d6_0000_0000 | (register.idx as u64 * 0x1234567 + 0x76543210))
            }
        }
    }

    fn operation(lo: Reg, hi: Reg) -> Op {
        Op::St256 {
            lo,
            hi,
            addr: crate::Addr {
                base: None,
                base2: None,
                off: 0,
            },
            sync: None,
        }
    }

    fn sequence(lo: Reg, hi: Reg) -> Vec<Reg> {
        [lo, hi]
            .into_iter()
            .flat_map(|first| {
                (0..4).map(move |index| Reg {
                    idx: first.idx + index,
                    ..first
                })
            })
            .collect()
    }

    #[test]
    fn st256_uses_two_bases_in_order_and_four_bytes_per_register() {
        for low_file in [File::V, File::R, File::UR, File::P, File::UP] {
            for high_file in [File::V, File::R, File::UR, File::P, File::UP] {
                for high_index in [16, 32] {
                    let lo = Reg {
                        file: low_file,
                        idx: 16,
                    };
                    let hi = Reg {
                        file: high_file,
                        idx: high_index,
                    };
                    let reader = Reader {
                        calls: RefCell::new(Vec::new()),
                        fail_at: usize::MAX,
                    };
                    let actual = from_st256(&operation(lo, hi), &reader).unwrap().unwrap();
                    let expected: Vec<_> = sequence(lo, hi)
                        .iter()
                        .flat_map(|register| {
                            (register.idx as u32 * 0x1234567 + 0x76543210).to_le_bytes()
                        })
                        .collect();
                    assert_eq!(actual, expected);
                    assert_eq!(*reader.calls.borrow(), sequence(lo, hi));
                }
            }
        }
    }

    #[test]
    fn st256_stops_at_each_failed_read_with_the_original_prefix() {
        let lo = Reg {
            file: File::R,
            idx: 16,
        };
        let hi = Reg {
            file: File::R,
            idx: 32,
        };
        for fail_at in 0..8 {
            let reader = Reader {
                calls: RefCell::new(Vec::new()),
                fail_at,
            };
            assert_eq!(from_st256(&operation(lo, hi), &reader), Some(Err(fail_at)));
            assert_eq!(*reader.calls.borrow(), sequence(lo, hi)[..=fail_at]);
        }
        let reader = Reader {
            calls: RefCell::new(Vec::new()),
            fail_at: 0,
        };
        assert_eq!(from_st256(&Op::Exit, &reader), None);
        assert!(reader.calls.borrow().is_empty());
    }
}
