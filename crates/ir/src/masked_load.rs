/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::Op;
use crate::Reg;

pub trait RegisterWriter {
    fn write(&mut self, destination: Reg, value: u32, source_byte: u8);
}

fn word(data: &[u8; 32], source: u8) -> u32 {
    let first = source as usize * 4;
    u32::from_le_bytes([
        data[first],
        data[first + 1],
        data[first + 2],
        data[first + 3],
    ])
}

fn half<W: RegisterWriter>(first: Reg, mask: u8, start: u8, data: &[u8; 32], writer: &mut W) {
    let mut rank = 0u16;
    let mut index = 0u8;
    while index < 4 {
        let source = start + index;
        if (mask >> source) & 1 != 0 {
            let destination = Reg {
                file: first.file,
                idx: first.idx + rank,
            };
            rank += 1;
            let value = word(data, source);
            writer.write(destination, value, source * 4);
        }
        index += 1;
    }
}

pub fn execute<W: RegisterWriter>(operation: &Op, data: &[u8; 32], writer: &mut W) {
    if let Op::Ld256 { lo, hi, mask, .. } = operation {
        half(*lo, *mask, 0, data, writer);
        half(*hi, *mask, 4, data, writer);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Addr;
    use crate::File;

    #[derive(Default)]
    struct Capture(Vec<(Reg, u32, u8)>);

    impl RegisterWriter for Capture {
        fn write(&mut self, destination: Reg, value: u32, source_byte: u8) {
            self.0.push((destination, value, source_byte));
        }
    }

    #[test]
    fn selected_words_use_their_memory_position_and_compacted_destination() {
        let data = std::array::from_fn(|index| (index as u8).wrapping_mul(37));
        for file in [File::V, File::R, File::UR, File::P, File::UP] {
            for mask in 0..=255u8 {
                let operation = Op::Ld256 {
                    hi: Reg { file, idx: 32 },
                    lo: Reg { file, idx: 16 },
                    addr: Addr {
                        base: None,
                        base2: None,
                        off: 0,
                    },
                    sync: None,
                    nc: false,
                    mask,
                };
                let mut captured = Capture::default();
                execute(&operation, &data, &mut captured);
                let expected: Vec<_> = (0..8u8)
                    .filter(|index| mask & (1 << index) != 0)
                    .map(|index| {
                        let half = if index < 4 { 15 } else { 240 };
                        let previous = (mask as u16) & ((1u16 << index) - 1) & half;
                        let destination = Reg {
                            file,
                            idx: if index < 4 { 16 } else { 32 } + previous.count_ones() as u16,
                        };
                        let first = index as usize * 4;
                        (
                            destination,
                            u32::from_le_bytes(data[first..first + 4].try_into().unwrap()),
                            index * 4,
                        )
                    })
                    .collect();
                assert_eq!(captured.0, expected, "{file:?}, {mask:#x}");
            }
        }
    }

    #[test]
    fn another_operation_performs_no_callback() {
        let mut captured = Capture::default();
        execute(&Op::Exit, &[0; 32], &mut captured);
        assert!(captured.0.is_empty());
    }

    #[test]
    fn a_failing_callback_preserves_the_completed_prefix() {
        struct FailAt {
            calls: Vec<u8>,
            byte: u8,
        }
        impl RegisterWriter for FailAt {
            fn write(&mut self, _destination: Reg, _value: u32, source_byte: u8) {
                self.calls.push(source_byte);
                assert_ne!(source_byte, self.byte, "expected callback failure");
            }
        }
        let operation = Op::Ld256 {
            hi: Reg {
                file: File::R,
                idx: 32,
            },
            lo: Reg {
                file: File::R,
                idx: 16,
            },
            addr: Addr {
                base: None,
                base2: None,
                off: 0,
            },
            sync: None,
            nc: false,
            mask: 0x3c,
        };
        let mut writer = FailAt {
            calls: Vec::new(),
            byte: 16,
        };
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            execute(&operation, &[0; 32], &mut writer);
        }));
        assert!(result.is_err());
        assert_eq!(writer.calls, [8, 12, 16]);
    }
}
