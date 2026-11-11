/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! ASSUME(S-SMEM-STRONG): SASS LDS / STS of the CTA's shared memory are relaxed strong
//! accesses at CTA scope (an LDS whose scoreboard is waited on before the next memory
//! instruction acquires: `ptxv_sass::smem_acquires`). The hardware-faithful model.
//!
//! Strong accesses are scheduling points, which costs every kernel that uses shared memory
//! (r2_cute_small: 1.5x the states). So a verdict is computed in two passes: first with
//! LDS / STS plain (`weak`); if that pass finds no race, it is also the strong model's verdict
//! (strong accesses only add happens-before, and without races the order of plain accesses
//! changes no outcome, E-DRF). Otherwise, if a race may be an artifact of plain LDS/STS
//! (`needs_strong`), the strong kernel decides; a race that is one in both models stands.

use std::sync::Arc;

use ptxv_ir::Front;
use ptxv_ir::Kernel;
use ptxv_ir::Op;
use ptxv_ir::Space;

use crate::explore::Verdict;
use crate::machine::Machine;

/// The kernel with SASS LDS / STS plain, if it has any.
pub fn weak(k: &Kernel) -> Option<Kernel> {
    if k.front != Front::Sass {
        return None;
    }
    let mut w = k.clone();
    let mut any = false;
    for i in &mut w.code {
        if let Op::Ld {
            space: Space::Shared,
            sync,
            ..
        }
        | Op::St {
            space: Space::Shared,
            sync,
            ..
        } = &mut i.op
        {
            if sync.is_some() {
                *sync = None;
                any = true;
            }
        }
    }
    any.then_some(w)
}

fn with_kernel(mut m: Machine, k: &Arc<Kernel>) -> Machine {
    m.k = k.clone();
    m
}

/// The weak pass's machine: the weak kernel, and `weak_stop` set (the search may stop at the
/// first race that `needs_strong` would send to the strong pass, whose verdict replaces the
/// weak one). `PTXV_NO_WEAK_STOP` explores the weak pass in full.
fn weak_machine(m: Machine, w: &Arc<Kernel>, strong: &Kernel) -> Machine {
    let mut m = with_kernel(m, w);
    if std::env::var("PTXV_NO_WEAK_STOP").is_err() {
        m.weak_stop = if acquiring_lds(strong) { 2 } else { 1 };
    }
    m
}

/// Does the kernel have an acquiring LDS (S-SMEM-STRONG)?
pub fn acquiring_lds(k: &Kernel) -> bool {
    k.code.iter().any(|i| {
        matches!(
            &i.op,
            Op::Ld {
                space: Space::Shared,
                sync: Some(s),
                ..
            } if s.acquires()
        )
    })
}

/// Is `sig` a race that sends the weak verdict to the strong pass (`needs_strong`), given
/// whether the strong kernel has acquiring LDS?
pub fn strong_race(sig: &str, acq: bool) -> bool {
    sig.starts_with("race-ldsts:") || (acq && sig.starts_with("race:gmem:"))
}

fn finish(mut v: Verdict, strong: bool) -> Verdict {
    if strong {
        v.notes.push(
            "a race with LDS/STS as plain accesses: re-checked with them as relaxed .cta strong (S-SMEM-STRONG)"
                .into(),
        );
    }
    v
}

/// Does the weak verdict depend on LDS/STS being plain? A race between two LDS/STS (or an
/// LDS/STS and an ATOMS, `race-ldsts` in exec.rs) is none in
/// the strong model; a global-memory race may be ordered by an acquiring LDS. Any other race
/// (an async-proxy write against an LDS, say) is one in both, and the verdict reports it.
fn needs_strong(v: &Verdict, k: &Kernel) -> bool {
    let acq = acquiring_lds(k);
    v.findings
        .iter()
        .any(|f| f.kind == "race" && strong_race(&f.sig, acq))
}

/// `run` on the weak kernel, then, if a race there may be an artifact of plain LDS/STS
/// (`needs_strong`), on `m0`'s own (strong) kernel.
pub fn two_pass(m0: Machine, run: impl Fn(Machine) -> Verdict) -> Verdict {
    let Some(w) = weak(&m0.k) else {
        return run(m0);
    };
    let w = Arc::new(w);
    let v = run(weak_machine(m0.clone(), &w, &m0.k));
    if !needs_strong(&v, &m0.k) {
        return finish(v, false);
    }
    finish(run(m0), true)
}

/// `two_pass` for the entry points that build machines with `make`.
pub fn two_pass_make(
    make: &dyn Fn() -> Machine,
    run: &mut dyn FnMut(&dyn Fn() -> Machine) -> Verdict,
) -> Verdict {
    let Some(w) = weak(&make().k) else {
        return run(make);
    };
    let w = Arc::new(w);
    let k = make().k;
    let v = run(&|| weak_machine(make(), &w, &k));
    if !needs_strong(&v, &k) {
        return finish(v, false);
    }
    finish(run(make), true)
}
