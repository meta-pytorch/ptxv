/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use ptxv_ir::Addr;
use ptxv_ir::Cmp;
use ptxv_ir::File;
use ptxv_ir::Front;
use ptxv_ir::Inst;
use ptxv_ir::MemSync;
use ptxv_ir::Op;
use ptxv_ir::Pred;
use ptxv_ir::Reg;
use ptxv_ir::SReg;
use ptxv_ir::Scope;
use ptxv_ir::Sem;
use ptxv_ir::Space;
use ptxv_ir::Val;

use crate::machine::Footprint;
use crate::machine::Machine;
fn inst(op: Op) -> Inst {
    Inst {
        guard: Pred::TRUE,
        op,
        uniform: false,
        loc: 0,
        text: String::new(),
    }
}
fn base(bssy: bool, upguard: bool) -> Machine {
    let case = ptxv_case::Case {
        dir: std::path::PathBuf::new(),
        m: serde_json::from_value(serde_json::json!({
            "name": "bsync-frame", "description": "enabled convergence mask actions",
            "entry": "k", "grid": [1, 1, 1], "block": [2, 1, 1],
            "outputs": [], "buffers": [], "params": [], "variants": []
        }))
        .unwrap(),
    };
    let mut k = ptxv_ptx::parse_kernel(
        ".version 8.8\n.target sm_100a\n.address_size 64\n.visible .entry k(){ret;}\n",
        Some("k"),
    )
    .unwrap();
    k.front = Front::Sass;
    let writer = if upguard {
        Op::Setp {
            d: Reg {
                file: File::UP,
                idx: 0,
            },
            cmp: Cmp::Ne,
            bits: 32,
            a: Val::Imm(0),
            b: Val::Imm(0),
            and: Pred::TRUE,
            or: false,
            xor: false,
            ex: None,
        }
    } else {
        Op::Mov {
            d: Reg {
                file: File::UR,
                idx: 4,
            },
            a: Val::Imm(1),
        }
    };
    k.code = vec![
        inst(Op::Setp {
            d: Reg {
                file: File::UP,
                idx: 0,
            },
            cmp: Cmp::Eq,
            bits: 32,
            a: Val::Imm(0),
            b: Val::Imm(0),
            and: Pred::TRUE,
            or: false,
            xor: false,
            ex: None,
        }),
        inst(Op::Setp {
            d: Reg {
                file: File::P,
                idx: 0,
            },
            cmp: Cmp::Ne,
            bits: 32,
            a: Val::SReg(SReg::Tid(0)),
            b: Val::Imm(0),
            and: Pred::TRUE,
            or: false,
            xor: false,
            ex: None,
        }),
        inst(Op::Bra {
            target: 6,
            any: false,
            tid0_falls: false,
            all: false,
            cond: None,
            div: None,
        }),
        inst(Op::Bssy { b: 0 }),
        inst(Op::Bsync { b: 0 }),
        inst(Op::Bsync { b: 1 }),
        inst(writer),
        inst(if bssy { Op::Bssy { b: 0 } } else { Op::Nop }),
        inst(Op::Ld {
            d: Reg {
                file: File::R,
                idx: 0,
            },
            n: 1,
            space: Space::Shared,
            bytes: 4,
            addr: Addr {
                base: None,
                base2: None,
                off: 0,
            },
            sync: Some(MemSync {
                sem: Sem::Relaxed,
                scope: Scope::Cta,
            }),
            nc: false,
            sext: false,
        }),
        inst(Op::Exit),
    ];
    k.code[2].guard = Pred {
        reg: Some(Reg {
            file: File::P,
            idx: 0,
        }),
        neg: false,
    };
    if upguard {
        k.code[4].guard = Pred {
            reg: Some(Reg {
                file: File::UP,
                idx: 0,
            }),
            neg: false,
        };
    }
    let launch = crate::build_launch(&case, &k).unwrap();
    let mut m = Machine::new(Arc::new(k), Arc::new(launch));
    assert_eq!(m.threads.len(), 2);
    step(&mut m, 0);
    step(&mut m, 3);
    assert_eq!(m.threads[0].pc, 4);
    assert_eq!(m.threads[1].pc, 6);
    assert_eq!(m.warps[0].bmask[0], 1);
    m
}
fn step_mode(m: &mut Machine, pc: u32, sync_only: bool) -> Footprint {
    let action = m
        .actions()
        .into_iter()
        .map(|x| x.0)
        .find(|a| matches!(a,crate::explore::Action::Split{pc:p,..}if*p==pc))
        .expect("actual action must be enabled");
    m.set_footprint(Some(Box::new(Footprint {
        sync_only,
        ..Default::default()
    })));
    m.apply(&action);
    let mut fp = *m.take_footprint().unwrap();
    fp.normalize();
    fp
}
fn step(m: &mut Machine, pc: u32) -> Footprint {
    step_mode(m, pc, false)
}
fn run(bssy: bool, upguard: bool, sync_only: bool) {
    let mut m = base(bssy, upguard);
    m.sfoot = Some(Arc::new(crate::machine::SFoot::new(&[])));
    let mut l = m.clone();
    let fa = step_mode(&mut l, 4, sync_only);
    let fb = step_mode(&mut l, 6, sync_only);
    let mut r = m.clone();
    let _ = step_mode(&mut r, 6, sync_only);
    let _ = step_mode(&mut r, 4, sync_only);
    let same = l.fingerprint() == r.fingerprint();
    let conflict = fa.conflicts(&fb);
    println!(
        "FRAME sync_only={sync_only} bssy={bssy} upguard={upguard} same={same} conflict={conflict} left0pc={} right0pc={} ar={:?} aw={:?} br={:?} bw={:?}",
        l.threads[0].pc, r.threads[0].pc, fa.r, fa.w, fb.r, fb.w
    );
    assert!(l
        .sfoot
        .as_ref()
        .unwrap()
        .snapshot()
        .1
        .iter()
        .any(|x| x.0 == 19));
    if bssy && !upguard {
        assert!(!same);
        assert!(conflict, "mask19 must retain BSSY dependency");
    } else if upguard {
        assert!(conflict, "UPguard must retain uniform dependency");
    } else {
        assert!(same);
        assert!(!conflict);
    }
}
#[test]
fn convergence_mask_mutation_conflicts_with_a_peer_bsync() {
    for sync_only in [false, true] {
        run(true, false, sync_only);
    }
}

#[test]
fn independent_uniform_write_stays_independent_of_unconditional_bsync() {
    for sync_only in [false, true] {
        run(false, false, sync_only);
    }
}

#[test]
fn uniform_bsync_guard_keeps_the_shared_register_dependency() {
    for sync_only in [false, true] {
        run(true, true, sync_only);
    }
}
