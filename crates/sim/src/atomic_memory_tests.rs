/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::path::PathBuf;
use std::sync::Arc;

use ptxv_ir::AtomOp;
use ptxv_ir::File;
use ptxv_ir::Front;
use ptxv_ir::Op;
use ptxv_ir::Pred;
use ptxv_ir::Reg;
use ptxv_ir::Val;
use serde_json::json;

use super::NumState;
use crate::exec::Loc;
use crate::halfin::Half;
use crate::halfin::Src;
use crate::machine::Machine;

const OLD: u64 = (2.0f32.to_bits() as u64) << 32 | 1.0f32.to_bits() as u64;
const NEW: u64 = (3.0f32.to_bits() as u64) << 32 | 5.0f32.to_bits() as u64;

fn before_atomic(atomic: &str) -> Machine {
    before_atomic_bytes(atomic, &OLD.to_le_bytes(), true)
}

fn before_atomic_bytes(atomic: &str, initial: &[u8], tracked: bool) -> Machine {
    let ptx = format!(
        r#"
.version 8.8
.target sm_103a
.address_size 64
.visible .entry k(.param .u64 input) {{
  .reg .b32 %r<12>;
  .reg .b64 %p<3>;
  .reg .b16 %h<2>;
  .reg .pred %q;
  .shared .align 16 .b8 scratch[16];
  ld.param.u64 %p0, [input];
  ld.global.u32 %r0, [%p0];
  ld.global.u32 %r1, [%p0+4];
  ld.global.u32 %r2, [%p0+8];
  {atomic}
  ret;
}}
"#
    );
    let kernel = Arc::new(ptxv_ptx::parse_kernel(&ptx, Some("k")).unwrap());
    let manifest = serde_json::from_value(json!({
        "name":"atomic-memory", "description":"atomic memory effects",
        "entry":"k", "grid":[1,1,1], "block":[1,1,1], "outputs":[],
        "buffers":[{"name":"x","dtype":"f32","shape":[4],"init":"zeros"}],
        "params":[{"kind":"ptr","buffer":"x"}], "variants":[]
    }))
    .unwrap();
    let case = ptxv_case::Case {
        dir: PathBuf::new(),
        m: manifest,
    };
    let mut launch = crate::build_launch(&case, &kernel).unwrap();
    Arc::make_mut(&mut launch.allocs[0].init)[..initial.len()].copy_from_slice(initial);
    let mut machine = Machine::new(kernel, Arc::new(launch));
    if tracked {
        let mut ns = NumState::default();
        ns.f32_inputs.insert(0);
        ns.half.inputs.insert(0, Half::Bf16);
        ns.taint.taint_alloc(0, 16);
        machine.num = Some(Box::new(ns));
    }
    for _ in 0..32 {
        if matches!(
            machine.k.code[machine.threads[0].pc as usize].op,
            Op::Atom { .. }
        ) {
            if let Some(ns) = machine.num.as_ref() {
                for off in [0, 4, 8] {
                    assert!(
                        ns.mem.contains_key(&(1, 0, off)),
                        "missing initial history at {off}"
                    );
                }
            }
            return machine;
        }
        let action = machine.actions()[0].0.clone();
        machine.apply(&action);
    }
    panic!("did not reach the atomic instruction");
}

fn finish(machine: Machine) -> Machine {
    let (machine, findings, complete) = crate::explore::run_once(machine);
    assert!(complete && findings.is_empty(), "{findings:?}");
    machine
}

fn byte_state(machine: &Machine) -> Vec<(u8, bool, bool, Option<Src>, bool)> {
    (0..16)
        .map(|offset| {
            let byte = *machine.gmem[0].bytes.get(offset);
            let shadow = machine.gmem[0].sh.get(offset);
            let loc = Loc::Global {
                alloc: 0,
                off: offset as u64,
            };
            (
                byte,
                shadow.undef,
                shadow.w.is_some(),
                machine.half_mem(loc, 1, Some(&[byte]))[0],
                machine
                    .num
                    .as_ref()
                    .unwrap()
                    .taint
                    .mem_has((1, 0, offset as u64)),
            )
        })
        .collect()
}

#[test]
fn successful_wide_atomics_clear_upper_word_history() {
    for atomic in [
        format!("atom.global.cas.b64 %p1, [%p0], {OLD}, {NEW};"),
        format!("atom.global.exch.b64 %p1, [%p0], {NEW};"),
    ] {
        let machine = finish(before_atomic(&atomic));
        let ns = machine.num.as_ref().unwrap();
        assert!(!ns.mem.contains_key(&(1, 0, 0)), "{atomic}");
        assert!(
            !ns.mem.contains_key(&(1, 0, 4)),
            "stale upper history: {atomic}"
        );
        assert!(
            ns.mem.contains_key(&(1, 0, 8)),
            "adjacent history: {atomic}"
        );
        assert_eq!(ns.written, [(0, 0), (0, 4)].into_iter().collect());
        let actual: Vec<_> = (0..8)
            .map(|offset| *machine.gmem[0].bytes.get(offset))
            .collect();
        assert_eq!(actual, NEW.to_le_bytes(), "{atomic}");
    }
}

/// A 64-bit compare-and-swap of input data that fails here writes nothing, but another input
/// may swap (MISS 177, ASSUME N-EQUIV): both words it covers lose their nodes and count as
/// written. Its bytes, shadow, 16-bit provenance and taint and the thread's clock stay.
#[test]
fn failed_cas_forgets_both_words_and_keeps_byte_state() {
    let machine = before_atomic(&format!(
        "atom.global.cas.b64 %p1, [%p0], {}, {NEW};",
        OLD + 1
    ));
    let before = byte_state(&machine);
    let clock = machine.threads[0].vc.clone();
    let adjacent = machine.num.as_ref().unwrap().mem[&(1, 0, 8)];
    let machine = finish(machine);
    let ns = machine.num.as_ref().unwrap();
    assert!(!ns.mem.contains_key(&(1, 0, 0)));
    assert!(!ns.mem.contains_key(&(1, 0, 4)), "stale upper history");
    assert_eq!(ns.mem[&(1, 0, 8)], adjacent);
    assert_eq!(ns.written, [(0, 0), (0, 4)].into_iter().collect());
    assert_eq!(byte_state(&machine), before);
    assert_eq!(machine.threads[0].vc, clock);
}

#[test]
fn narrow_atomic_preserves_adjacent_word_history() {
    let machine = before_atomic("atom.global.exch.b32 %r0, [%p0], 0;");
    let upper = machine.num.as_ref().unwrap().mem[&(1, 0, 4)];
    let machine = finish(machine);
    let ns = machine.num.as_ref().unwrap();
    assert!(!ns.mem.contains_key(&(1, 0, 0)));
    assert_eq!(ns.mem[&(1, 0, 4)], upper);
    assert!(ns.mem.contains_key(&(1, 0, 8)));
    assert_eq!(ns.written, [(0, 0)].into_iter().collect());
}

fn destination(machine: &Machine) -> (Reg, u8) {
    match machine.k.code[machine.threads[0].pc as usize].op {
        Op::Atom {
            d: Some(d), bytes, ..
        } => (d, bytes),
        ref op => panic!("expected a returning atomic, got {op:?}"),
    }
}

fn global_bytes(machine: &Machine) -> Vec<u8> {
    (0..16)
        .map(|offset| *machine.gmem[0].bytes.get(offset))
        .collect()
}

fn check_result(atomic: &str, old: u64, replacement: u64, returned: u64) -> Machine {
    let machine = before_atomic_bytes(atomic, &old.to_le_bytes(), false);
    let (d, bytes) = destination(&machine);
    let mut expected = global_bytes(&machine);
    expected[..bytes as usize].copy_from_slice(&replacement.to_le_bytes()[..bytes as usize]);
    let machine = finish(machine);
    assert_eq!(global_bytes(&machine), expected, "{atomic}");
    assert_eq!(
        machine.rd(0, &Val::Reg(d, 1)).unwrap(),
        returned,
        "{atomic}"
    );
    machine
}

#[test]
fn integer_atomic_results_cover_signed_widths_and_bounds() {
    let cases: &[(&str, u64, u64, u64)] = &[
        ("or.b32", 0xf0, 0x0f, 0xff),
        ("and.b32", 0xf0, 0x0f, 0),
        ("xor.b32", 0xf3, 0x0f, 0xfc),
        ("add.u32", 0xffff_ffff, 1, 0),
        ("add.u32", 0x1_0000_0001, 0x2_0000_0002, 3),
        ("exch.b32", 0xabcd, 0x1_0000_0123, 0x123),
        ("min.u32", 0xffff_ffff, 1, 1),
        ("max.u32", 0xffff_ffff, 1, 0xffff_ffff),
        ("min.s32", 0x8000_0000, 1, 0x8000_0000),
        ("max.s32", 0x8000_0000, 1, 1),
        ("inc.u32", 4, 5, 5),
        ("inc.u32", 5, 5, 0),
        ("inc.u32", 6, 5, 0),
        ("inc.u32", 0xffff_ffff, 0xffff_ffff, 0),
        ("dec.u32", 0, 5, 5),
        ("dec.u32", 6, 5, 5),
        ("dec.u32", 5, 5, 4),
        ("dec.u32", 1, 0xffff_ffff, 0),
        (
            "or.b64",
            0xf000_0000_0000_00f0,
            0x0f00_0000_0000_000f,
            0xff00_0000_0000_00ff,
        ),
        ("and.b64", 0xf000_0000_0000_00f0, 0x0f00_0000_0000_000f, 0),
        (
            "xor.b64",
            0xf300_0000_0000_00f3,
            0x0f00_0000_0000_000f,
            0xfc00_0000_0000_00fc,
        ),
        ("add.u64", u64::MAX, 1, 0),
        (
            "exch.b64",
            0xabcd,
            0xfedc_ba98_7654_3210,
            0xfedc_ba98_7654_3210,
        ),
        ("min.u64", u64::MAX, 1, 1),
        ("max.u64", u64::MAX, 1, u64::MAX),
        ("min.s64", 0x8000_0000_0000_0000, 1, 0x8000_0000_0000_0000),
        ("max.s64", 0x8000_0000_0000_0000, 1, 1),
    ];
    for &(operation, old, value, replacement) in cases {
        let atomic = format!("atom.global.{operation} %p1, [%p0], {value};");
        let returned = if operation.ends_with("64") {
            old
        } else {
            old as u32 as u64
        };
        check_result(&atomic, old, replacement, returned);
    }
}

#[test]
fn cas_returns_old_and_truncates_operands() {
    for (width, old, compare, replacement, expected, wrote) in [
        (32, 0x1_0000_0001, 0x2_0000_0001, 0x3_0000_0005, 5, true),
        (32, 0x1_0000_0001, 2, 5, 1, false),
        (64, OLD, OLD, NEW, NEW, true),
        (64, OLD, OLD + 1, NEW, OLD, false),
    ] {
        let atomic = format!("atom.global.cas.b{width} %p1, [%p0], {compare}, {replacement};");
        let returned = if width == 64 { old } else { old as u32 as u64 };
        let machine = check_result(&atomic, old, expected, returned);
        for offset in 0..width / 8 {
            assert_eq!(machine.gmem[0].sh.get(offset).w.is_some(), wrote);
        }
    }
}

#[test]
fn spin_cas_returns_success_and_writes_failed_value() {
    for success in [false, true] {
        let atomic = format!(
            "atom.global.cas.b64 %p1, [%p0], {}, {NEW};",
            OLD + u64::from(!success)
        );
        let mut machine = before_atomic_bytes(&atomic, &OLD.to_le_bytes(), false);
        let (d, _) = destination(&machine);
        let predicate = Reg {
            file: File::P,
            idx: 0,
        };
        let pc = machine.threads[0].pc as usize;
        // CAST.SPIN is a SASS-only operation; this checks its Machine transition.
        match &mut Arc::make_mut(&mut machine.k).code[pc].op {
            Op::Atom { op, p, .. } => {
                *op = AtomOp::CasSpin;
                *p = Some(predicate);
            }
            _ => unreachable!(),
        }
        let clock = machine.threads[0].vc.clone();
        let machine = finish(machine);
        assert_eq!(machine.rd(0, &Val::Reg(d, 1)).unwrap(), u64::from(success));
        assert_eq!(
            machine.prd(
                0,
                Pred {
                    reg: Some(predicate),
                    neg: false
                }
            ),
            success
        );
        assert_ne!(machine.threads[0].vc, clock);
        let expected = if success { NEW } else { OLD };
        assert_eq!(&global_bytes(&machine)[..8], &expected.to_le_bytes());
        assert!((0..8).all(|offset| machine.gmem[0].sh.get(offset).w.is_some()));
    }
}

#[test]
fn sass_atomic_results_use_the_declared_register_word_count() {
    for spin in [false, true] {
        let atomic = format!("atom.global.cas.b64 %p1, [%p0], {OLD}, {NEW};");
        let mut machine = before_atomic_bytes(&atomic, &OLD.to_le_bytes(), false);
        let d = Reg {
            file: File::R,
            idx: 8,
        };
        let upper = Reg { idx: 9, ..d };
        machine.wr(0, upper, 0x1234_5678, 1);
        let pc = machine.threads[0].pc as usize;
        let kernel = Arc::make_mut(&mut machine.k);
        kernel.front = Front::Sass;
        kernel.nregs[File::R as usize] = kernel.nregs[File::V as usize];
        match &mut kernel.code[pc].op {
            Op::Atom { op, d: output, .. } => {
                if spin {
                    *op = AtomOp::CasSpin;
                }
                *output = Some(d);
            }
            _ => unreachable!(),
        }
        let machine = finish(machine);
        assert_eq!(
            machine.rd(0, &Val::Reg(d, 1)).unwrap(),
            if spin { 1 } else { OLD as u32 as u64 }
        );
        assert_eq!(
            machine.rd(0, &Val::Reg(upper, 1)).unwrap(),
            if spin { 0x1234_5678 } else { OLD >> 32 }
        );
    }
}

#[test]
fn floating_atomics_keep_lane_formats_and_old_results() {
    let cases: &[(&str, u64, u64, u64)] = &[
        ("add.f32", 0x3f80_0000, 0x4000_0000, 0x4040_0000),
        ("add.f32", 1, 1, 0),
        ("add.f32", 0x8000_0000, 0x8000_0000, 0x8000_0000),
        ("add.f32", 0x7f80_0000, 0xff80_0000, 0x7fff_ffff),
        (
            "add.f64",
            0x3ff0_0000_0000_0000,
            0x4000_0000_0000_0000,
            0x4008_0000_0000_0000,
        ),
        ("add.f64", 1, 1, 2),
        (
            "add.f64",
            0x8000_0000_0000_0000,
            0x8000_0000_0000_0000,
            0x8000_0000_0000_0000,
        ),
        (
            "add.f64",
            0x7ff0_0000_0000_0000,
            0xfff0_0000_0000_0000,
            0xfff8_0000_0000_0000,
        ),
        (
            "add.f64",
            0x7ff8_0000_0000_1234,
            0x7ff8_0000_0000_5678,
            0x7ff8_0000_0000_5678,
        ),
        ("add.noftz.bf16", 0x3f80, 0x4000, 0x4040),
        ("add.noftz.bf16", 1, 1, 2),
        ("add.noftz.bf16", 0x8000, 0x8000, 0x8000),
        ("add.noftz.bf16", 0x7f80, 0xff80, 0x7fff),
        ("add.noftz.bf16x2", 0xc000_3f80, 0x3f00_4000, 0xbfc0_4040),
        ("add.noftz.bf16x2", 0x0001_7f80, 0x0001_ff80, 0x0002_7fff),
    ];
    for &(operation, old, value, replacement) in cases {
        let output = if operation.ends_with(".bf16") {
            "%h0"
        } else {
            "%p1"
        };
        let atomic = format!("atom.global.{operation} {output}, [%p0], {value};");
        check_result(&atomic, old, replacement, old);
    }
}

#[test]
fn vector_f32_atomics_return_each_old_lane() {
    for lanes in [2, 4] {
        let old = [1.0f32, 2.0, 4.0, 8.0];
        let mut initial = Vec::new();
        let mut setup = String::new();
        for (lane, value) in old.into_iter().enumerate() {
            initial.extend_from_slice(&value.to_bits().to_le_bytes());
            setup.push_str(&format!(
                "mov.b32 %r{}, {};\n",
                4 + lane,
                (value * 2.0).to_bits()
            ));
        }
        let sources = (4..4 + lanes)
            .map(|i| format!("%r{i}"))
            .collect::<Vec<_>>()
            .join(", ");
        let outputs = (8..8 + lanes)
            .map(|i| format!("%r{i}"))
            .collect::<Vec<_>>()
            .join(", ");
        let atomic =
            format!("{setup} atom.global.add.v{lanes}.f32 {{{outputs}}}, [%p0], {{{sources}}};");
        let machine = before_atomic_bytes(&atomic, &initial, false);
        let (d, _) = destination(&machine);
        let machine = finish(machine);
        for (lane, value) in old.into_iter().enumerate() {
            let word = u32::from_le_bytes(std::array::from_fn(|byte| {
                *machine.gmem[0].bytes.get(lane * 4 + byte)
            }));
            assert_eq!(
                word,
                (if lane < lanes { value * 3.0 } else { value }).to_bits()
            );
            if lane < lanes {
                assert_eq!(
                    machine
                        .rd(
                            0,
                            &Val::Reg(
                                Reg {
                                    idx: d.idx + lane as u16,
                                    ..d
                                },
                                1
                            )
                        )
                        .unwrap(),
                    value.to_bits() as u64
                );
            }
        }
    }
}

#[test]
fn shared_atomics_update_the_resolved_shared_location() {
    let atomic = format!(
        "st.shared.u64 [scratch], {OLD}; atom.shared.cas.b64 %p1, [scratch], {OLD}, {NEW};"
    );
    let machine = before_atomic_bytes(&atomic, &OLD.to_le_bytes(), false);
    let (d, _) = destination(&machine);
    let address = match &machine.k.code[machine.threads[0].pc as usize].op {
        Op::Atom { addr, .. } => machine.addr(0, addr).unwrap() as usize,
        _ => unreachable!(),
    };
    let global = global_bytes(&machine);
    let machine = finish(machine);
    assert_eq!(machine.rd(0, &Val::Reg(d, 1)).unwrap(), OLD);
    assert_eq!(global_bytes(&machine), global);
    let actual: Vec<_> = (0..8)
        .map(|byte| *machine.smem[0].bytes.get(address + byte))
        .collect();
    assert_eq!(actual, NEW.to_le_bytes());
}

#[test]
fn malformed_atomic_widths_fail_before_memory_effects() {
    for (operation, bytes) in [
        (AtomOp::Add, 2),
        (AtomOp::Inc, 8),
        (AtomOp::Dec, 8),
        (AtomOp::AddBf16, 8),
        (AtomOp::AddBf16x2, 8),
    ] {
        let mut machine = before_atomic("atom.global.exch.b64 %p1, [%p0], 0;");
        let pc = machine.threads[0].pc;
        match &mut Arc::make_mut(&mut machine.k).code[pc as usize].op {
            Op::Atom {
                op, bytes: width, ..
            } => {
                *op = operation;
                *width = bytes;
            }
            _ => unreachable!(),
        }
        let before = byte_state(&machine);
        let histories = machine.num.as_ref().unwrap().mem.clone();
        let markers = machine.num.as_ref().unwrap().written.clone();
        let clock = machine.threads[0].vc.clone();
        let error = machine
            .exec(&[0], pc)
            .expect_err("unsupported atomic width");
        assert!(error.contains("width"), "{operation:?}/{bytes}: {error}");
        assert_eq!(byte_state(&machine), before);
        assert_eq!(machine.num.as_ref().unwrap().mem, histories);
        assert_eq!(machine.num.as_ref().unwrap().written, markers);
        assert_eq!(machine.threads[0].vc, clock);
    }
}
