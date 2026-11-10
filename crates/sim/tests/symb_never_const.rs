// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Values the concolic expressions (symb.rs) must never fold into a constant of the captured
//! value, one test each: a write under an input-dependent guard, a load through an
//! input-dependent address, a register read before any write, bytes landed by cp.async, the
//! elements of a vector load, a float atomic's result; and a shuffle (the source lane's
//! expression) and the bf16-to-f32 widening (one canonical node). Checked by evaluating the
//! stored values' expressions at other inputs, on both fronts where the form survives ptxas.

mod cc_common;

use std::collections::HashMap;

use cc_common::*;
use ptxv_sim::machine::Machine;
use ptxv_sim::symb::E;
use ptxv_sim::symb::Evaluator;
use ptxv_sim::symb::Node;
use ptxv_sim::symb::SymState;
use ptxv_sim::symb::Un;

const PTX: &str = r#"
.version 9.0
.target sm_103a
.address_size 64

.visible .entry nc(.param .u64 pa, .param .u64 pb, .param .u64 ptbl, .param .u64 pbf, .param .u64 pacc, .param .u64 pout)
.reqntid 64
{
	.reg .pred %p<8>;
	.reg .b32 %r<80>;
	.reg .b16 %h<4>;
	.reg .f32 %f<4>;
	.reg .b64 %rd<40>;
	.shared .align 16 .b8 sh[256];

	ld.param.u64 %rd1, [pa];
	ld.param.u64 %rd2, [pb];
	ld.param.u64 %rd3, [ptbl];
	ld.param.u64 %rd4, [pbf];
	ld.param.u64 %rd5, [pacc];
	ld.param.u64 %rd6, [pout];
	cvta.to.global.u64 %rd1, %rd1;
	cvta.to.global.u64 %rd2, %rd2;
	cvta.to.global.u64 %rd3, %rd3;
	cvta.to.global.u64 %rd4, %rd4;
	cvta.to.global.u64 %rd5, %rd5;
	cvta.to.global.u64 %rd6, %rd6;
	mov.u32 %r1, %tid.x;
	mul.wide.u32 %rd10, %r1, 4;
	add.s64 %rd11, %rd1, %rd10;
	add.s64 %rd12, %rd2, %rd10;
	ld.global.u32 %r2, [%rd11];
	ld.global.u32 %r3, [%rd12];
	shl.b32 %r4, %r1, 6;
	mul.wide.u32 %rd13, %r4, 1;
	add.s64 %rd14, %rd6, %rd13;
	// 0: a write under an input-dependent guard
	mov.u32 %r10, 1000;
	setp.lt.u32 %p1, %r2, 5;
	@%p1 add.u32 %r10, %r3, 7;
	st.global.u32 [%rd14], %r10;
	// 1: a load through an input-dependent address
	and.b32 %r11, %r2, 7;
	mul.wide.u32 %rd15, %r11, 4;
	add.s64 %rd16, %rd3, %rd15;
	ld.global.u32 %r12, [%rd16];
	st.global.u32 [%rd14+4], %r12;
	// 2: a register read before any write
	add.u32 %r20, %r19, 1;
	st.global.u32 [%rd14+8], %r20;
	// 3: bytes landed by cp.async, read by the next thread
	mov.u32 %r30, sh;
	shl.b32 %r31, %r1, 2;
	add.u32 %r32, %r30, %r31;
	cp.async.ca.shared.global [%r32], [%rd12], 4;
	cp.async.wait_all;
	bar.sync 0;
	add.u32 %r33, %r1, 1;
	and.b32 %r33, %r33, 63;
	shl.b32 %r33, %r33, 2;
	add.u32 %r34, %r30, %r33;
	ld.shared.u32 %r35, [%r34];
	st.global.u32 [%rd14+12], %r35;
	// 4: elements of a vector load
	and.b32 %r40, %r1, 15;
	mul.wide.u32 %rd17, %r40, 16;
	add.s64 %rd18, %rd2, %rd17;
	ld.global.v4.u32 {%r41, %r42, %r43, %r44}, [%rd18];
	st.global.u32 [%rd14+16], %r43;
	st.global.u32 [%rd14+20], %r41;
	// 5: a float atomic's result
	atom.global.add.f32 %f1, [%rd5], 0f3F800000;
	mov.b32 %r50, %f1;
	st.global.u32 [%rd14+24], %r50;
	// 6: a shuffle from lane 3
	shfl.sync.idx.b32 %r60, %r3, 3, 0x1f, 0xffffffff;
	st.global.u32 [%rd14+28], %r60;
	// 7: a bf16 element widened to f32 by a shift
	mul.wide.u32 %rd19, %r1, 2;
	add.s64 %rd20, %rd4, %rd19;
	ld.global.u16 %h1, [%rd20];
	cvt.u32.u16 %r70, %h1;
	shl.b32 %r71, %r70, 16;
	st.global.u32 [%rd14+32], %r71;
	ret;
}
"#;

const TOML: &str = r#"
name = "nc"
description = "never-constant expressions"
ptx = "kernel.ptx"
entry = "nc"
grid = [1, 1, 1]
block = [64, 1, 1]
outputs = ["out"]

[[buffers]]
name = "a"
dtype = "u32"
shape = [64]
init = "iota"

[[buffers]]
name = "b"
dtype = "u32"
shape = [64]
init = "iota"

[[buffers]]
name = "tbl"
dtype = "u32"
shape = [8]
init = "iota"

[[buffers]]
name = "bf"
dtype = "u16"
shape = [64]
init = "iota"

[[buffers]]
name = "acc"
dtype = "f32"
shape = [1]
init = "zeros"

[[buffers]]
name = "out"
dtype = "u32"
shape = [1024]
init = "zeros"

[[params]]
kind = "ptr"
buffer = "a"

[[params]]
kind = "ptr"
buffer = "b"

[[params]]
kind = "ptr"
buffer = "tbl"

[[params]]
kind = "ptr"
buffer = "bf"

[[params]]
kind = "ptr"
buffer = "acc"

[[params]]
kind = "ptr"
buffer = "out"

[[variants]]
id = "good"
expect = ["clean"]
"#;

struct Run {
    m0: Machine,
    sym: Box<SymState>,
    mfin: Machine,
}

impl Run {
    fn new(front: &str) -> Option<Run> {
        let dir = scratch_dir("nc", front);
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("kernel.ptx"), PTX).unwrap();
        std::fs::write(dir.join("case.toml"), TOML).unwrap();
        let m0 = match machine(&dir, "good", front) {
            Ok(m) => m,
            Err(e) if front == "sass" => {
                println!("sass skipped: {e:#}");
                return None;
            }
            Err(e) => panic!("{e:#}"),
        };
        let (_, s) = ptxv_sim::coverage::analyze_with(&m0, Some(Default::default()));
        let (sym, mfin) = s.unwrap();
        assert_eq!(sym.n_mismatch, 0, "{front}: {:#?}", sym.mismatches);
        Some(Run { m0, sym, mfin })
    }

    fn alloc(&self, name: &str) -> u32 {
        self.m0
            .launch
            .allocs
            .iter()
            .position(|a| a.name == name)
            .unwrap() as u32
    }

    /// The expression of word `slot` thread `t` stored.
    fn out(&mut self, t: u64, slot: u64) -> E {
        let oa = self.alloc("out");
        let off = 64 * t + 4 * slot;
        let o = self.mfin.outputs();
        let v = u32::from_le_bytes(o[off as usize..off as usize + 4].try_into().unwrap()) as u64;
        let e = self.sym.global_expr(&self.m0, oa, off, 4, v);
        assert_eq!(self.sym.ar.conc[e as usize], v);
        e
    }

    /// The input leaf of word `i` of allocation `name`.
    fn leaf(&self, name: &str, i: u64) -> E {
        let a = self.alloc(name);
        (0..self.sym.ar.len())
            .find(|&x| matches!(self.sym.ar.nodes[x], Node::Input { alloc, off, bytes: 4 } if alloc == a && off == 4 * i))
            .unwrap_or_else(|| panic!("no leaf {name}[{i}]")) as E
    }

    fn eval(&self, e: E, env: &[(E, u64)]) -> Option<u64> {
        let env: HashMap<E, u64> = env.iter().copied().collect();
        Evaluator::new(&self.sym.ar, &self.mfin).eval(e, &env)
    }

    fn show(&self, e: E) -> String {
        self.sym.ar.show(e, &|a| format!("a{a}"), 8)
    }
}

fn fronts() -> impl Iterator<Item = (&'static str, Run)> {
    ["ptx", "sass"]
        .into_iter()
        .filter_map(|f| Run::new(f).map(|r| (f, r)))
}

/// `@p add r, b, 7` with `p = a < 5`: `ite(p, b + 7, 1000)`, both for threads the guard holds
/// for (t = 2) and for those it fails for (t = 9).
#[test]
fn guarded_write_is_ite() {
    for (front, mut r) in fronts() {
        let e = r.out(2, 0);
        let (a2, b2) = (r.leaf("a", 2), r.leaf("b", 2));
        assert_eq!(r.eval(e, &[(a2, 9)]), Some(1000), "{front}: {}", r.show(e));
        assert_eq!(r.eval(e, &[(b2, 50)]), Some(57), "{front}: {}", r.show(e));
        let e = r.out(9, 0);
        assert!(
            r.sym.ar.is_const(e).is_none(),
            "{front}: guard-false write folded"
        );
        let a9 = r.leaf("a", 9);
        assert_eq!(r.eval(e, &[(a9, 1)]), Some(16), "{front}: {}", r.show(e));
    }
}

/// `tbl[a & 7]`: defined at the captured address only.
#[test]
fn load_through_input_address_is_at_addr() {
    for (front, mut r) in fronts() {
        let e = r.out(3, 1);
        assert!(
            r.sym
                .ar
                .cone(e)
                .iter()
                .any(|&x| matches!(r.sym.ar.nodes[x as usize], Node::AtAddr { .. })),
            "{front}: {}",
            r.show(e)
        );
        let (a3, t3) = (r.leaf("a", 3), r.leaf("tbl", 3));
        assert_eq!(r.eval(e, &[(a3, 4)]), None, "{front}: another address");
        assert_eq!(
            r.eval(e, &[(a3, 11), (t3, 77)]),
            Some(77),
            "{front}: same address"
        );
    }
}

/// `r19 + 1`, r19 never written: an opaque leaf (not input), never a constant (PTX only:
/// ptxas gives the register some value).
#[test]
fn register_read_before_write_is_opaque() {
    let mut r = Run::new("ptx").unwrap();
    let e = r.out(0, 2);
    assert!(r.sym.ar.is_const(e).is_none(), "{}", r.show(e));
    assert!(r.sym.ar.undef[e as usize], "{}", r.show(e));
    assert!(!r.sym.ar.dep[e as usize], "an undefined value is not input");
}

/// cp.async moves `b[t]` to shared; the next thread's load reads the input leaf `b[t + 1]`.
#[test]
fn cp_async_bytes_keep_input_leaves() {
    for (front, mut r) in fronts() {
        let e = r.out(0, 3);
        assert_eq!(e, r.leaf("b", 1), "{front}: {}", r.show(e));
    }
}

/// `ld.v4 {r41..r44}, [b + 16 (t & 15)]`: element 2 is `b[4 (t & 15) + 2]`, element 0
/// `b[4 (t & 15)]`.
#[test]
fn vector_load_elements_are_their_own_leaves() {
    for (front, mut r) in fronts() {
        let e = r.out(1, 4);
        assert_eq!(e, r.leaf("b", 6), "{front}: {}", r.show(e));
        let e = r.out(1, 5);
        assert_eq!(e, r.leaf("b", 4), "{front}: {}", r.show(e));
    }
}

/// `atom.add.f32` of a constant: the old value depends on other threads' order: opaque.
#[test]
fn float_atomic_result_is_opaque() {
    for (front, mut r) in fronts() {
        let e = r.out(5, 6);
        assert!(r.sym.ar.is_const(e).is_none(), "{front}: {}", r.show(e));
        assert!(
            r.sym.ar.leaves(e).iter().any(
                |&x| matches!(r.sym.ar.nodes[x as usize], Node::Opaque { id, .. }
                if r.sym.ar.opaque[id as usize].why == "float atomic result")
            ),
            "{front}: {}",
            r.show(e)
        );
    }
}

/// `shfl.idx` from lane 3: the source lane's leaf, `b[32 (t / 32) + 3]`.
#[test]
fn shuffle_is_source_lane_expression() {
    for (front, mut r) in fronts() {
        for t in [0u64, 17, 40] {
            let e = r.out(t, 7);
            assert_eq!(
                e,
                r.leaf("b", 32 * (t / 32) + 3),
                "{front} t{t}: {}",
                r.show(e)
            );
        }
    }
}

/// `zext(bf[t]) << 16`: the canonical `B2F` node of the 16-bit input element.
#[test]
fn bf16_widening_is_one_node() {
    for (front, mut r) in fronts() {
        let e = r.out(5, 8);
        match r.sym.ar.nodes[e as usize] {
            Node::Un { op: Un::B2F, a, .. } => {
                assert!(
                    matches!(
                        r.sym.ar.nodes[a as usize],
                        Node::Input {
                            bytes: 2,
                            off: 10,
                            ..
                        }
                    ),
                    "{front}: {}",
                    r.show(e)
                );
            }
            _ => panic!("{front}: {}", r.show(e)),
        }
    }
}
