/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::Addr;
use crate::AtomOp;
use crate::File;
use crate::Front;
use crate::Op;
use crate::Reg;
use crate::Val;

fn append(result: &mut Vec<Reg>, first: Reg, count: u16) {
    let mut index = 0;
    while index < count {
        result.push(Reg {
            file: first.file,
            idx: first.idx + index,
        });
        index += 1;
    }
}

pub(crate) fn scalar_count(file: File, count: u8) -> u16 {
    if file == File::V {
        1
    } else {
        (count as u16).max(1).min(2)
    }
}

fn slots(result: &mut Vec<Reg>, value: &Val) {
    if let Val::Reg(first, count) = value {
        let first = *first;
        let count = scalar_count(first.file, *count);
        append(result, first, count);
    }
}

fn address(result: &mut Vec<Reg>, addr: &Addr) {
    if let Some(value) = &addr.base {
        slots(result, value);
    }
    if let Some(value) = &addr.base2 {
        slots(result, value);
    }
}

fn vector(result: &mut Vec<Reg>, value: &Val) {
    match value {
        Val::Reg(first, count) if first.file == File::V => {
            let first = *first;
            let count = *count;
            append(result, first, (count as u16).max(1));
        }
        _ => slots(result, value),
    }
}

/// All register operands read by supported memory operations; the instruction guard is separate.
pub fn memory_reads(op: &Op) -> Option<Vec<Reg>> {
    let mut result = Vec::new();
    match op {
        Op::Ld { addr, .. } | Op::Ld256 { addr, .. } => address(&mut result, addr),
        Op::St { addr, v, bytes, .. } => {
            address(&mut result, addr);
            store_sources(&mut result, v, *bytes, false);
        }
        Op::StAsync {
            addr,
            mbar,
            v,
            bytes,
        } => {
            slots(&mut result, addr);
            slots(&mut result, mbar);
            store_sources(&mut result, v, *bytes, true);
        }
        Op::St256 { addr, lo, hi, .. } => {
            address(&mut result, addr);
            append(&mut result, *lo, 4);
            append(&mut result, *hi, 4);
        }
        Op::Atom {
            addr,
            v,
            new,
            op,
            bytes,
            ..
        } => {
            address(&mut result, addr);
            atomic_sources(&mut result, *op, *bytes, v, new);
        }
        _ => return None,
    }
    Some(result)
}

pub(crate) fn repeated_virtual(front: Front, file: File, elements: u8, bytes: u8) -> bool {
    front == Front::Sass && file == File::V && elements == 1 && bytes > 8
}

/// Account for the SASS discard lowering's repeated virtual value; other layouts retain their sources.
pub fn memory_reads_for_front(front: Front, operation: &Op) -> Option<Vec<Reg>> {
    if let Op::St {
        space:
            crate::Space::Shared
            | crate::Space::SharedCluster
            | crate::Space::Global
            | crate::Space::Generic,
        bytes,
        addr,
        v: Val::Reg(first, elements),
        ..
    } = operation
    {
        if repeated_virtual(front, first.file, *elements, *bytes) {
            let mut result = Vec::new();
            address(&mut result, addr);
            result.push(*first);
            return Some(result);
        }
    }
    memory_reads(operation)
}

/// Every register a supported memory operation may write, independent of its guard.
pub fn memory_writes(op: &Op) -> Option<Vec<Reg>> {
    let mut result = Vec::new();
    match op {
        Op::Ld { d, bytes, n, .. } => {
            let count = load_count(d.file, *bytes, *n);
            append(&mut result, *d, count);
        }
        Op::Atom {
            d, p, op, bytes, ..
        } => {
            if let Some(first) = d {
                let first = *first;
                let op = *op;
                let bytes = *bytes;
                let count = atom_count(op, first.file, bytes);
                append(&mut result, first, count);
            }
            if let Some(pred) = p {
                result.push(*pred);
            }
        }
        Op::Ld256 { lo, hi, mask, .. } => {
            let (low, high) = selected_halves(*mask);
            append(&mut result, *lo, low);
            append(&mut result, *hi, high);
        }
        Op::St { .. } | Op::StAsync { .. } | Op::St256 { .. } => (),
        _ => return None,
    }
    Some(result)
}

fn load_count(file: File, bytes: u8, n: u8) -> u16 {
    if file == File::V {
        if n > 1 {
            n as u16
        } else {
            (bytes as u16 / 8).max(1)
        }
    } else {
        (bytes as u16 / 4).max(1)
    }
}
fn cas_count(file: File, bytes: u8) -> u16 {
    if bytes == 8 && file != File::V { 2 } else { 1 }
}
fn atom_count(op: AtomOp, file: File, bytes: u8) -> u16 {
    let vector = op == AtomOp::FAddFtz && (bytes == 8 || bytes == 16);
    if op == AtomOp::CasSpin || (file == File::V && !vector) {
        1
    } else {
        word_count(bytes)
    }
}

fn floating_sources(result: &mut Vec<Reg>, value: &Val, bytes: u8) {
    if bytes > 4 {
        match value {
            Val::Reg(first, _) => append(result, *first, (bytes / 4) as u16),
            _ => slots(result, value),
        }
    } else {
        slots(result, value);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemSync;
    use crate::Scope;
    use crate::Sem;
    use crate::Space;

    fn r(index: u16) -> Reg {
        Reg {
            file: File::R,
            idx: index,
        }
    }

    #[test]
    fn implicit_cas_replacement_sources_are_included() {
        for bytes in [4, 8] {
            let operation = Op::Atom {
                op: AtomOp::Cas,
                space: Space::Global,
                bytes,
                addr: Addr {
                    base: None,
                    base2: None,
                    off: 0,
                },
                v: Val::Reg(r(20), bytes / 4),
                d: Some(r(10)),
                sync: MemSync {
                    sem: Sem::Relaxed,
                    scope: Scope::Gpu,
                },
                p: None,
                new: None,
            };
            let expected: Vec<_> = (20..20 + 2 * (bytes / 4) as u16).map(r).collect();
            assert_eq!(operation.src_regs(), expected);
        }
    }

    #[test]
    fn masked_load_destinations_pack_each_half_without_unused_registers() {
        for mask in 1u16..256 {
            let operation = Op::Ld256 {
                hi: r(40),
                lo: r(20),
                mask: mask as u8,
                addr: Addr {
                    base: None,
                    base2: None,
                    off: 0,
                },
                sync: None,
                nc: false,
            };
            let mut expected = Vec::new();
            for (start, base) in [(0, 20), (4, 40)] {
                let selected = (start..start + 4)
                    .filter(|bit| mask >> bit & 1 != 0)
                    .count();
                expected.extend((base..base + selected as u16).map(r));
            }
            assert_eq!(operation.dst_regs(), expected, "mask={mask:#x}");
        }
    }
}

fn atomic_sources(result: &mut Vec<Reg>, op: AtomOp, bytes: u8, v: &Val, new: &Option<Val>) {
    atomic_value_sources(result, op, bytes, v);
    match (op, new) {
        (AtomOp::Cas | AtomOp::CasSpin, None) => {
            implicit_cas_sources(result, bytes, v);
        }
        (_, Some(value)) => slots(result, value),
        _ => (),
    }
}

/// Data operands, including explicit and implicit compare-and-swap replacements.
pub fn memory_data_reads(op: &Op) -> Option<Vec<Reg>> {
    let mut result = Vec::new();
    match op {
        Op::Ld { .. } | Op::Ld256 { .. } => (),
        Op::St { v, bytes, .. } => store_sources(&mut result, v, *bytes, false),
        Op::StAsync { v, bytes, .. } => store_sources(&mut result, v, *bytes, true),
        Op::St256 { lo, hi, .. } => {
            append(&mut result, *lo, 4);
            append(&mut result, *hi, 4);
        }
        Op::Atom {
            op, bytes, v, new, ..
        } => atomic_sources(&mut result, *op, *bytes, v, new),
        _ => return None,
    }
    Some(result)
}

/// Address and barrier operands that determine where a memory operation acts.
pub fn memory_control_reads(op: &Op) -> Option<Vec<Reg>> {
    let mut result = Vec::new();
    match op {
        Op::Ld { addr, .. }
        | Op::Ld256 { addr, .. }
        | Op::St { addr, .. }
        | Op::St256 { addr, .. }
        | Op::Atom { addr, .. } => address(&mut result, addr),
        Op::StAsync { addr, mbar, .. } => {
            slots(&mut result, addr);
            slots(&mut result, mbar);
        }
        _ => return None,
    }
    Some(result)
}

fn implicit_cas_sources(result: &mut Vec<Reg>, bytes: u8, compare: &Val) {
    if let Val::Reg(first, _) = compare {
        let first = *first;
        let count = cas_count(first.file, bytes);
        append(
            result,
            Reg {
                file: first.file,
                idx: first.idx + count,
            },
            count,
        );
    }
}

fn selected_half(mask: u8) -> u16 {
    ((mask & 1) != 0) as u16
        + ((mask & 2) != 0) as u16
        + ((mask & 4) != 0) as u16
        + ((mask & 8) != 0) as u16
}

fn selected_halves(mask: u8) -> (u16, u16) {
    (selected_half(mask), selected_half(mask >> 4))
}

fn word_count(bytes: u8) -> u16 {
    (bytes as u16 / 4).max(1)
}

fn atomic_value_sources(result: &mut Vec<Reg>, op: AtomOp, bytes: u8, v: &Val) {
    match op {
        AtomOp::Exch => vector(result, v),
        AtomOp::FAddFtz => floating_sources(result, v, bytes),
        _ => slots(result, v),
    }
}

pub(crate) fn store_count(file: File, bytes: u8, elements: u8, asynchronous: bool) -> u16 {
    if file == File::V && elements >= 2 {
        elements as u16
    } else if bytes > if asynchronous { 4 } else { 8 } {
        if asynchronous && file == File::V {
            1
        } else {
            word_count(bytes)
        }
    } else {
        scalar_count(file, elements)
    }
}

fn store_sources(result: &mut Vec<Reg>, value: &Val, bytes: u8, asynchronous: bool) {
    if let Val::Reg(first, elements) = value {
        let first = *first;
        let count = store_count(first.file, bytes, *elements, asynchronous);
        append(result, first, count);
    }
}

#[cfg(test)]
mod store_width_tests {
    use super::*;
    use crate::Space;

    fn address() -> Addr {
        Addr {
            base: None,
            base2: None,
            off: 0,
        }
    }

    #[test]
    fn store_sources_follow_actual_width_and_operand_layout() {
        for file in [File::R, File::UR, File::V] {
            for bytes in [4, 8, 16, 32] {
                for elements in [0, 1, 2, 4, 16] {
                    let first = Reg { file, idx: 16 };
                    let value = Val::Reg(first, elements);
                    let operation = Op::St {
                        space: Space::Global,
                        bytes,
                        addr: address(),
                        v: value.clone(),
                        sync: None,
                    };
                    let count = if file == File::V && elements >= 2 {
                        elements
                    } else if bytes > 8 {
                        bytes / 4
                    } else if file == File::V {
                        1
                    } else {
                        elements.clamp(1, 2)
                    };
                    let expected: Vec<_> = (0..count)
                        .map(|offset| Reg {
                            file,
                            idx: first.idx + offset as u16,
                        })
                        .collect();
                    assert_eq!(
                        operation.src_regs(),
                        expected,
                        "file={file:?}, bytes={bytes}, elements={elements}"
                    );
                    assert_eq!(memory_data_reads(&operation), Some(expected));
                    let asynchronous = Op::StAsync {
                        addr: Val::Imm(0),
                        mbar: Val::Imm(0),
                        v: value,
                        bytes,
                    };
                    let count = if file == File::V && elements >= 2 {
                        elements
                    } else if file != File::V && bytes > 4 {
                        bytes / 4
                    } else if file == File::V {
                        1
                    } else {
                        elements.clamp(1, 2)
                    };
                    let expected: Vec<_> = (0..count)
                        .map(|offset| Reg {
                            file,
                            idx: first.idx + offset as u16,
                        })
                        .collect();
                    assert_eq!(
                        asynchronous.src_regs(),
                        expected,
                        "async file={file:?}, bytes={bytes}, elements={elements}"
                    );
                    assert_eq!(memory_data_reads(&asynchronous), Some(expected));
                }
            }
        }
    }
}
