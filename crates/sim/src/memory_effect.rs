/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ptxv_ir::File;
use ptxv_ir::Inst;
use ptxv_ir::Op;
use ptxv_ir::Reg;
use ptxv_ir::Val;
use ptxv_memory_write::Key;
use ptxv_memory_write::engine::Engine;
use ptxv_memory_write::store::WordStoreInput;
use ptxv_memory_write::store::commit_word;
use ptxv_memory_write::symbolic::Metadata;

use crate::exec::Loc;
use crate::machine::Machine;
use crate::machine::Mem;
use crate::machine::SMEM_RESERVED;

pub(crate) struct WordEffect {
    pc: u32,
    input: WordStoreInput,
    state: Engine,
}

#[cfg(test)]
thread_local! {
    static COMMIT_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

fn register_parts(r: Reg, slots: u8) -> impl Iterator<Item = Reg> {
    let slots = if r.file == File::V { 1 } else { slots.max(1) };
    (0..slots as u16).map(move |i| Reg {
        idx: r.idx + i,
        ..r
    })
}

fn memory(m: &Machine, loc: Loc) -> Option<(&Mem, Key)> {
    match loc {
        Loc::Shared { cta, off } => Some((
            &m.smem[cta as usize],
            Key {
                space: 0,
                allocation: cta as u64,
                offset: off as u64,
            },
        )),
        Loc::Global { alloc, off } => Some((
            &m.gmem[alloc],
            Key {
                space: 1,
                allocation: alloc as u64,
                offset: off,
            },
        )),
        Loc::Param { .. } => None,
    }
}

impl Machine {
    pub(crate) fn prepare_word_effect(
        &mut self,
        t: usize,
        inst: &Inst,
        loc: Loc,
        data: &[u8],
    ) -> Option<WordEffect> {
        let Op::St {
            bytes: 4, addr, v, ..
        } = &inst.op
        else {
            return None;
        };
        if data.len() != 4 || self.wundef != 0 || self.pred_undef(t, inst.guard).is_some() {
            return None;
        }
        let ns = self.num.as_ref()?;
        if ns.taint.position || !self.prd(t, inst.guard) {
            return None;
        }
        if !matches!(v, Val::Imm(_) | Val::Reg(_, 1)) {
            return None;
        }
        if ns.half.is_on()
            && self
                .half_val(t, v)
                .is_some_and(|b| b.iter().any(Option::is_some))
        {
            return None;
        }
        let guard_dependent = inst.guard.reg.is_some_and(|r| ns.taint.reg_tainted(t, r));
        let mut address_dependent = false;
        for value in [&addr.base, &addr.base2].into_iter().flatten() {
            match value {
                Val::Imm(_) => {}
                Val::Reg(r, slots) => {
                    for r in register_parts(*r, *slots) {
                        if self.reg_umask(t, r) != 0 {
                            return None;
                        }
                        address_dependent |= ns.taint.reg_tainted(t, r);
                    }
                }
                _ => return None,
            }
        }
        if guard_dependent || address_dependent {
            return None;
        }
        let (mem, destination) = memory(self, loc)?;
        let end = destination.offset.checked_add(4)?;
        if end > mem.bytes.len() as u64
            || (destination.offset..end).any(|o| mem.sh.get(o as usize).undef)
        {
            return None;
        }
        let bits = u32::from_le_bytes(data.try_into().expect("four-byte store payload"));
        let source = self.store_source_metadata(t, v, bits);
        assert!(
            source.defined,
            "eligible store source must be fully defined"
        );
        let (mem, _) = memory(self, loc).expect("eligible store has live memory");
        let ns = self
            .num
            .as_ref()
            .expect("eligible store has numerical state");
        let cells = (destination.offset.saturating_sub(3)..end)
            .map(|offset| {
                let shadow = mem.sh.get(offset as usize);
                let defined = !shadow.undef
                    && (destination.space == 1
                        || offset < SMEM_RESERVED as u64
                        || shadow.w.is_some());
                ns.effect_cell(
                    Key {
                        offset,
                        ..destination
                    },
                    *mem.bytes.get(offset as usize),
                    defined,
                    shadow.w.is_some(),
                )
            })
            .collect();
        let mut state = Engine {
            cells,
            numeric_written: Vec::new(),
            half_written: Vec::new(),
            track_numeric: true,
            track_half: ns.half.is_on(),
        };
        ns.effect_markers(&mut state);
        Some(WordEffect {
            pc: self.threads[t].pc,
            input: WordStoreInput {
                space: destination.space,
                allocation: destination.allocation,
                source,
                guard: Metadata {
                    captured: 1,
                    history: None,
                    tainted: false,
                    defined: true,
                },
                address_low: Metadata {
                    captured: destination.offset as u32,
                    history: None,
                    tainted: false,
                    defined: true,
                },
                address_high: Metadata {
                    captured: (destination.offset >> 32) as u32,
                    history: None,
                    tainted: false,
                    defined: true,
                },
            },
            state,
        })
    }

    pub(crate) fn commit_word_effect(&mut self, mut effect: WordEffect, loc: Loc) {
        #[cfg(test)]
        COMMIT_COUNT.with(|count| count.set(count.get() + 1));
        let (_, destination) = memory(self, loc).expect("prepared store has live memory");
        assert_eq!(destination.space, effect.input.space);
        assert_eq!(destination.allocation, effect.input.allocation);
        assert_eq!(
            destination.offset,
            effect.input.address_low.captured as u64
                | ((effect.input.address_high.captured as u64) << 32)
        );
        commit_word(&mut effect.state, effect.input).expect("eligible four-byte store commits");
        let mem = match loc {
            Loc::Shared { cta, .. } => &mut self.smem[cta as usize],
            Loc::Global { alloc, .. } => &mut self.gmem[alloc],
            Loc::Param { .. } => unreachable!("parameter stores are not eligible"),
        };
        for cell in &effect.state.cells {
            if destination.offset <= cell.key.offset && cell.key.offset < destination.offset + 4 {
                assert!(
                    !mem.sh.get(cell.key.offset as usize).undef,
                    "race bookkeeping changed eligible store definedness"
                );
                *mem.bytes.get_mut(cell.key.offset as usize) = cell.value;
                mem.sh.get_mut(cell.key.offset as usize).undef = !cell.defined;
            }
        }
        self.num
            .as_mut()
            .expect("prepared store has numerical state")
            .apply_effect_state(
                &effect.state,
                destination,
                effect.input.source.captured,
                effect.pc,
            );
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;
    use std::sync::Arc;

    use ptxv_case::Case;
    use ptxv_memory_write::Cell;
    use ptxv_memory_write::Origin;
    use ptxv_memory_write::WordHistory;

    use super::*;
    use crate::halfin::Src;
    use crate::num::Node;
    use crate::num::NodeKind;
    use crate::num::NumState;

    fn machine(code: &str, threads: u32) -> Machine {
        let mut case =
            Case::load(Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/e_resid"))
                .unwrap();
        case.m.params.clear();
        case.m.grid = [1, 1, 1];
        case.m.block = [threads, 1, 1];
        case.m.dynamic_smem = (SMEM_RESERVED + 16) as u32;
        case.m.buffers[0].shape = vec![16];
        let mut dis = String::from(".text.k:\n");
        for (index, operation) in code.lines().enumerate() {
            dis.push_str(&format!(
                "/*{:04x}*/ {operation} ;\n/* 0x000fc00000000000 */\n",
                index * 16
            ));
        }
        let kernel = ptxv_sass::lower(&dis, "k", &ptxv_sass::CubinMeta::default()).unwrap();
        let launch = crate::build_launch(&case, &kernel).unwrap();
        let mut machine = Machine::new(Arc::new(kernel), Arc::new(launch));
        let mut numeric = NumState::default();
        numeric.half.track = true;
        machine.num = Some(Box::new(numeric));
        machine
    }

    fn set_source(machine: &mut Machine, t: usize, bits: u32) {
        let base = machine.launch.allocs[0].base + 4;
        machine.wr(
            t,
            Reg {
                file: File::R,
                idx: 2,
            },
            base,
            2,
        );
        machine.wr(
            t,
            Reg {
                file: File::R,
                idx: 4,
            },
            bits as u64,
            1,
        );
    }

    fn defined(mem: &Mem, key: Key) -> bool {
        let shadow = mem.sh.get(key.offset as usize);
        !shadow.undef && (key.space == 1 || key.offset < SMEM_RESERVED as u64 || shadow.w.is_some())
    }

    #[test]
    fn ordinary_store_projects_one_checked_commit_to_every_memory_plane() {
        let mut machine = machine("STG.E [R2.64], R4\nEXIT", 1);
        let bits = 0x3f80_0000;
        set_source(&mut machine, 0, bits);
        let ns = machine.num.as_mut().unwrap();
        for hw in [bits, 0] {
            ns.nodes.push(Node {
                kind: NodeKind::Leaf,
                hw,
                exact: vec![],
                err: vec![],
                pc: 0,
            });
        }
        let destination = Key {
            space: 1,
            allocation: 0,
            offset: 4,
        };
        ns.taint.taint_alloc(0, 64);
        ns.f32_inputs.insert(0);
        ns.half.inputs.insert(0, crate::halfin::Half::Bf16);
        ns.apply_effect_state(
            &Engine {
                cells: (0..12)
                    .map(|offset| Cell {
                        key: Key {
                            offset,
                            ..destination
                        },
                        value: 0,
                        origin: Some(Origin::E {
                            alloc: 0,
                            elem: offset / 2,
                            b: (offset % 2) as u8,
                        }),
                        tainted: true,
                        defined: true,
                        written: false,
                        word: Some(WordHistory::Known(1)),
                    })
                    .collect(),
                numeric_written: vec![Key {
                    offset: 0,
                    ..destination
                }],
                half_written: vec![Key {
                    offset: 0,
                    ..destination
                }],
                track_numeric: true,
                track_half: true,
            },
            destination,
            0,
            0,
        );
        machine.num_put(
            0,
            Reg {
                file: File::R,
                idx: 4,
            },
            Some(0),
        );
        let instruction = machine.k.code[0].clone();
        let loc = Loc::Global { alloc: 0, off: 4 };
        let effect = machine
            .prepare_word_effect(0, &instruction, loc, &bits.to_le_bytes())
            .unwrap();
        assert_eq!(effect.state.cells.len(), 7);
        assert_eq!(effect.input.source.history, Some(0));
        let mut expected = effect.state;
        commit_word(&mut expected, effect.input).unwrap();
        let commits = COMMIT_COUNT.with(|count| count.get());
        machine.exec(&[0], 0).unwrap();
        assert_eq!(COMMIT_COUNT.with(|count| count.get()), commits + 1);
        let ns = machine.num.as_ref().unwrap();
        for expected in &expected.cells {
            let mem = &machine.gmem[0];
            let shadow = mem.sh.get(expected.key.offset as usize);
            let actual = ns.effect_cell(
                expected.key,
                *mem.bytes.get(expected.key.offset as usize),
                defined(mem, expected.key),
                shadow.w.is_some(),
            );
            assert_eq!(actual, *expected, "byte {}", expected.key.offset);
        }
        assert_eq!(ns.mem_node(Loc::Global { alloc: 0, off: 0 }), Some(1));
        assert_eq!(ns.mem_node(loc), Some(0));
        assert_eq!(machine.half_mem(loc, 4, None), vec![None; 4]);
        assert!(machine.findings.is_empty(), "{:?}", machine.findings);
    }

    #[test]
    fn checked_store_keeps_the_existing_access_and_race_bookkeeping() {
        let mut checked = machine("STG.E [R2.64], R4\nEXIT", 2);
        for t in 0..2 {
            set_source(&mut checked, t, (t + 1) as u32);
        }
        let mut legacy = checked.clone();
        legacy.num = None;
        for t in 0..2 {
            checked.exec(&[t], 0).unwrap();
            legacy.exec(&[t], 0).unwrap();
        }
        for offset in 0..64 {
            assert_eq!(
                checked.gmem[0].bytes.get(offset),
                legacy.gmem[0].bytes.get(offset)
            );
            assert_eq!(
                checked.gmem[0].sh.get(offset),
                legacy.gmem[0].sh.get(offset)
            );
        }
        let signatures = |m: &Machine| {
            m.findings
                .iter()
                .map(|f| (f.kind.clone(), f.sig.clone()))
                .collect::<Vec<_>>()
        };
        assert_eq!(signatures(&checked), signatures(&legacy));
        assert!(checked.findings.iter().any(|f| f.kind == "race"));
    }

    #[test]
    fn checked_shared_store_initializes_user_bytes_and_preserves_reserved_bytes() {
        for offset in [0, SMEM_RESERVED] {
            let mut checked = machine("STS [R2], R4\nEXIT", 1);
            checked.wr(
                0,
                Reg {
                    file: File::R,
                    idx: 2,
                },
                offset as u64,
                1,
            );
            checked.wr(
                0,
                Reg {
                    file: File::R,
                    idx: 4,
                },
                0x3f80_0000,
                1,
            );
            let loc = Loc::Shared {
                cta: 0,
                off: offset as u32,
            };
            let inst = checked.k.code[0].clone();
            let effect = checked
                .prepare_word_effect(0, &inst, loc, &0x3f80_0000u32.to_le_bytes())
                .unwrap();
            for cell in &effect.state.cells {
                assert_eq!(cell.defined, cell.key.offset < SMEM_RESERVED as u64);
                assert!(!cell.written);
            }
            let mut expected = effect.state;
            commit_word(&mut expected, effect.input).unwrap();
            let mut legacy = checked.clone();
            legacy.num = None;
            let commits = COMMIT_COUNT.with(|count| count.get());
            checked.exec(&[0], 0).unwrap();
            legacy.exec(&[0], 0).unwrap();
            assert_eq!(COMMIT_COUNT.with(|count| count.get()), commits + 1);
            for cell in &expected.cells {
                let mem = &checked.smem[0];
                let shadow = mem.sh.get(cell.key.offset as usize);
                assert_eq!(shadow, legacy.smem[0].sh.get(cell.key.offset as usize));
                assert_eq!(
                    checked.num.as_ref().unwrap().effect_cell(
                        cell.key,
                        *mem.bytes.get(cell.key.offset as usize),
                        defined(mem, cell.key),
                        shadow.w.is_some()
                    ),
                    *cell,
                );
            }
            assert!(checked.findings.is_empty(), "{:?}", checked.findings);
        }
    }

    #[test]
    fn same_value_store_preserves_unordered_undefined_data_on_legacy_dispatch() {
        let mut checked = machine("STG.E [R2.64], R4\nEXIT", 2);
        for t in 0..2 {
            set_source(&mut checked, t, 1);
        }
        checked.exec(&[0], 0).unwrap();
        for offset in 4..8 {
            checked.gmem[0].sh.get_mut(offset).undef = true;
        }
        let mut legacy = checked.clone();
        legacy.num = None;
        let commits = COMMIT_COUNT.with(|count| count.get());
        checked.exec(&[1], 0).unwrap();
        legacy.exec(&[1], 0).unwrap();
        assert_eq!(COMMIT_COUNT.with(|count| count.get()), commits);
        for offset in 4..8 {
            let shadow = checked.gmem[0].sh.get(offset);
            assert!(shadow.undef);
            assert_eq!(shadow, legacy.gmem[0].sh.get(offset));
        }
        assert!(
            checked
                .findings
                .iter()
                .any(|f| f.sig.starts_with("same-value:"))
        );
    }

    #[test]
    fn unsupported_store_planes_keep_their_existing_dispatch() {
        let mut machine = machine("@P0 STG.E [R2.64], R4\nEXIT", 1);
        set_source(&mut machine, 0, 0);
        let loc = Loc::Global { alloc: 0, off: 4 };
        let instruction = machine.k.code[0].clone();
        assert!(
            machine
                .prepare_word_effect(0, &instruction, loc, &[0; 4])
                .is_none()
        );
        machine.wp(
            0,
            Reg {
                file: File::P,
                idx: 0,
            },
            true,
        );
        assert!(
            machine
                .prepare_word_effect(0, &instruction, loc, &[0; 4])
                .is_some()
        );
        machine.wundef = 1;
        assert!(
            machine
                .prepare_word_effect(0, &instruction, loc, &[0; 4])
                .is_none()
        );
        machine.wundef = 0;
        machine.gmem[0].sh.get_mut(4).undef = true;
        assert!(
            machine
                .prepare_word_effect(0, &instruction, loc, &[0; 4])
                .is_none()
        );
        machine.gmem[0].sh.get_mut(4).undef = false;
        machine.half_set_reg(
            0,
            Reg {
                file: File::R,
                idx: 4,
            },
            Some([Some(Src::Zero); 4]),
        );
        assert!(
            machine
                .prepare_word_effect(0, &instruction, loc, &[0; 4])
                .is_none()
        );
    }
}
