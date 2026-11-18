/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::AtomOp;
use crate::File;
use crate::Front;
use crate::Op;
use crate::Reg;
use crate::Space;
use crate::Val;
use crate::register_effect::scalar_count;
use crate::register_effect::store_count;

fn stride(file: File, bytes: u8, elements: u8, exchange: bool) -> u32 {
    if file == File::V {
        if exchange {
            bytes as u32
        } else if elements >= 2 {
            (bytes / elements) as u32
        } else if bytes > 8 {
            4
        } else {
            bytes as u32
        }
    } else {
        4
    }
}

fn clipped_size(total: u32, start: u32, stride: u32) -> u32 {
    if start < total {
        (total - start).min(stride)
    } else {
        0
    }
}

fn append_windows(first: Reg, count: u16, stride: u32, total: u32) -> Vec<(Reg, u32, u32)> {
    let mut result = Vec::new();
    let mut index = 0;
    while index < count {
        let start = index as u32 * stride;
        let size = clipped_size(total, start, stride);
        result.push((
            Reg {
                file: first.file,
                idx: first.idx + index,
            },
            start,
            size,
        ));
        index += 1;
    }
    result
}

pub(crate) fn descriptor(operation: &Op) -> Option<(Reg, u8, u8, bool)> {
    match operation {
        Op::St {
            space: Space::Shared | Space::SharedCluster | Space::Global | Space::Generic,
            bytes,
            v: Val::Reg(first, elements),
            ..
        } => Some((*first, *elements, *bytes, false)),
        Op::Atom {
            op: AtomOp::Exch,
            space: Space::Shared | Space::SharedCluster | Space::Global | Space::Generic,
            bytes,
            v: Val::Reg(first, elements),
            ..
        } => Some((*first, *elements, *bytes, true)),
        _ => None,
    }
}

fn repeated_windows(first: Reg, bytes: u8) -> Vec<(Reg, u32, u32)> {
    let total = bytes as u32;
    let count = (total + 7) / 8;
    let mut result = Vec::new();
    let mut index = 0u32;
    while index < count {
        let start = index * 8;
        let size = clipped_size(total, start, 8);
        result.push((first, start, size));
        index += 1;
    }
    result
}

/// Source registers and the destination-byte interval each supplies to a store.
pub fn sources(front: Front, operation: &Op) -> Vec<(Reg, u32, u32)> {
    let Some((first, elements, bytes, exchange)) = descriptor(operation) else {
        return Vec::new();
    };
    if first.file == File::V {
        if let Front::Sass = front {
            if !exchange
                && crate::register_effect::repeated_virtual(front, first.file, elements, bytes)
                && bytes > 8
            {
                // The SASS discard lowering repeats one synthetic virtual value over the line.
                return repeated_windows(first, bytes);
            }
            return Vec::new();
        }
    }
    assemble(first, elements, bytes, exchange)
}

fn assemble(first: Reg, elements: u8, bytes: u8, exchange: bool) -> Vec<(Reg, u32, u32)> {
    let count = if exchange {
        scalar_count(first.file, elements)
    } else {
        store_count(first.file, bytes, elements, false)
    };
    let stride = stride(first.file, bytes, elements, exchange);
    append_windows(first, count, stride, bytes as u32)
}

fn has_register(registers: &[Reg], needle: Reg) -> bool {
    let mut found = false;
    let mut index = 0;
    while index < registers.len() {
        let candidate = registers[index];
        found = found || (candidate.file == needle.file && candidate.idx == needle.idx);
        index += 1;
    }
    found
}

fn deferred_registers(windows: &[(Reg, u32, u32)], control: &[Reg]) -> Vec<Reg> {
    let mut result = Vec::new();
    let mut index = 0;
    while index < windows.len() {
        let register = windows[index].0;
        if !has_register(control, register) {
            result.push(register);
        }
        index += 1;
    }
    result
}

fn narrow_descriptor(operation: &Op) -> Option<(Reg, u32)> {
    match operation {
        Op::St {
            bytes,
            v: Val::Reg(first, _),
            ..
        } => {
            if *bytes == 1 {
                Some((*first, 255))
            } else if *bytes == 2 {
                Some((*first, 65535))
            } else {
                None
            }
        }
        _ => None,
    }
}

fn narrow_sink(operation: &Op, control: &[Reg]) -> Option<(Reg, u32)> {
    let Some((register, mask)) = narrow_descriptor(operation) else {
        return None;
    };
    if register.file == File::V || has_register(control, register) {
        None
    } else {
        Some((register, mask))
    }
}

/// Preserve address uses when a register also supplies deferred store data.
pub fn metadata_sources(
    front: Front,
    operation: &Op,
) -> (Vec<(Reg, u32, u32)>, Vec<Reg>, Option<(Reg, u32)>) {
    let windows = sources(front, operation);
    let control = match crate::register_effect::memory_control_reads(operation) {
        Some(control) => control,
        None => Vec::new(),
    };
    let deferred = deferred_registers(&windows, &control);
    let narrow = narrow_sink(operation, &control);
    (windows, deferred, narrow)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reg(file: File, idx: u16) -> Reg {
        Reg { file, idx }
    }

    fn store(file: File, elements: u8, bytes: u8) -> Op {
        Op::St {
            space: Space::Global,
            bytes,
            addr: crate::Addr {
                base: None,
                base2: None,
                off: 0,
            },
            v: Val::Reg(reg(file, 16), elements),
            sync: None,
        }
    }

    #[test]
    fn store_windows_follow_payload_assembly() {
        let expected: Vec<_> = (0..4)
            .map(|index| (reg(File::R, 16 + index), 4 * index as u32, 4))
            .collect();
        assert_eq!(sources(Front::Sass, &store(File::R, 1, 16)), expected);
        assert_eq!(
            sources(Front::Ptx, &store(File::V, 2, 16)),
            vec![(reg(File::V, 16), 0, 8), (reg(File::V, 17), 8, 8)]
        );
        assert_eq!(
            sources(Front::Ptx, &store(File::V, 1, 8)),
            vec![(reg(File::V, 16), 0, 8)]
        );
        assert!(sources(Front::Sass, &store(File::V, 2, 16)).is_empty());
    }

    #[test]
    fn sass_synthetic_discard_publishes_every_repeated_byte_window() {
        let source = reg(File::V, 255);
        let mut operation = store(File::V, 1, 128);
        if let Op::St { v, .. } = &mut operation {
            *v = Val::Reg(source, 1);
        }
        assert_eq!(
            crate::register_effect::memory_reads_for_front(Front::Sass, &operation),
            Some(vec![source])
        );
        let (windows, deferred, narrow) = metadata_sources(Front::Sass, &operation);
        assert_eq!(
            windows,
            (0..16)
                .map(|index| (source, index * 8, 8))
                .collect::<Vec<_>>()
        );
        assert_eq!(deferred, vec![source; 16]);
        assert_eq!(narrow, None);
        if let Op::St { addr, .. } = &mut operation {
            addr.base = Some(Val::Reg(source, 1));
        }
        assert!(metadata_sources(Front::Sass, &operation).1.is_empty());
    }

    #[test]
    fn exchange_windows_use_scalar_value_layout() {
        let operation = Op::Atom {
            op: AtomOp::Exch,
            space: Space::Global,
            bytes: 8,
            addr: crate::Addr {
                base: None,
                base2: None,
                off: 0,
            },
            v: Val::Reg(reg(File::V, 16), 2),
            d: None,
            sync: crate::MemSync {
                sem: crate::Sem::Relaxed,
                scope: crate::Scope::Gpu,
            },
            p: None,
            new: None,
        };
        assert_eq!(
            sources(Front::Ptx, &operation),
            vec![(reg(File::V, 16), 0, 8)]
        );
    }
    #[test]
    fn address_registers_are_never_deferred_as_store_data() {
        let source = reg(File::R, 16);
        let mut operation = store(File::R, 1, 1);
        if let Op::St { addr, .. } = &mut operation {
            addr.base = Some(Val::Reg(source, 1));
        }
        let (windows, deferred, narrow) = metadata_sources(Front::Sass, &operation);
        assert_eq!(windows, vec![(source, 0, 1)]);
        assert!(deferred.is_empty());
        assert!(narrow.is_none());
        let (_, deferred, narrow) = metadata_sources(Front::Sass, &store(File::R, 1, 1));
        assert_eq!(deferred, vec![source]);
        assert_eq!(narrow, Some((source, 255)));
        let (_, _, narrow) = metadata_sources(Front::Sass, &store(File::R, 1, 2));
        assert_eq!(narrow, Some((source, 65535)));
    }
}
