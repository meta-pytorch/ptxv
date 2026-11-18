/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A separate certificate verdict for the admitted PTX constant global-memory subset.
//! Candidate generation is untrusted; acceptance repeats admission on the original launch.

use ptxv_ir::Kernel;
use ptxv_memory_write::memory_safety;
use ptxv_memory_write::memory_safety::Program;
use ptxv_memory_write::memory_safety::State;
use serde::Serialize;

use crate::machine::Launch;
use crate::strict_memory_admission;
use crate::strict_memory_admission::Refusal;
use crate::strict_memory_status::Status;
use crate::strict_memory_status::status_for_certificate;

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum Reason {
    Admission { detail: Refusal },
    InvalidLayout,
    StateCap,
    UnsupportedTransition,
    BadState,
    CertificateRejected,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct Report {
    scope: &'static str,
    status: Status,
    states: usize,
    state_cap: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    reason: Option<Reason>,
}

impl Report {
    pub fn is_safe(&self) -> bool {
        self.status.is_safe()
    }

    pub fn status_name(&self) -> &'static str {
        match self.status {
            Status::Safe => "safe",
            Status::Unsafe => "unsafe",
            Status::Inconclusive => "inconclusive",
        }
    }

    fn inconclusive(states: usize, state_cap: usize, reason: Reason) -> Self {
        Self {
            scope: "typed_ir_ptx_constant_global_memory",
            status: Status::Inconclusive,
            states,
            state_cap,
            reason: Some(reason),
        }
    }
}

pub fn verify_certificate(kernel: &Kernel, launch: &Launch, certificate: &[State]) -> bool {
    strict_memory_admission::verify_certificate(kernel, launch, certificate)
}

enum Candidate {
    Complete(Vec<State>),
    Unsafe(usize),
    Inconclusive(usize, Reason),
}

fn state_present(states: &[State], state: &State) -> bool {
    let mut index = 0;
    while index < states.len() {
        if states[index] == *state {
            return true;
        }
        index += 1;
    }
    false
}

fn append_thread(
    program: &Program,
    state: &State,
    thread: usize,
    states: &mut Vec<State>,
    state_cap: usize,
) -> Option<Candidate> {
    let Some(movements) = memory_safety::thread_moves(program, state, thread) else {
        return Some(Candidate::Inconclusive(
            states.len(),
            Reason::UnsupportedTransition,
        ));
    };
    let mut movement = 0;
    while movement < movements.len() {
        let next = memory_safety::apply_move(program, state, &movements[movement]);
        if next.bad {
            return Some(Candidate::Unsafe(states.len()));
        }
        if !state_present(states, &next) {
            if states.len() >= state_cap {
                return Some(Candidate::Inconclusive(states.len(), Reason::StateCap));
            }
            states.push(next);
        }
        movement += 1;
    }
    None
}

fn append_successors(
    program: &Program,
    state: &State,
    states: &mut Vec<State>,
    state_cap: usize,
) -> Option<Candidate> {
    let mut thread = 0;
    while thread < program.threads.len() {
        if let Some(stop) = append_thread(program, state, thread, states, state_cap) {
            return Some(stop);
        }
        thread += 1;
    }
    None
}

fn candidate(program: &Program, state_cap: usize) -> Candidate {
    if state_cap == 0 {
        return Candidate::Inconclusive(0, Reason::StateCap);
    }
    let Some(initial) = memory_safety::initial(program) else {
        return Candidate::Inconclusive(0, Reason::InvalidLayout);
    };
    let mut states = Vec::new();
    states.push(initial);
    let mut cursor = 0;
    while cursor < states.len() {
        let state = states[cursor].clone();
        if let Some(stop) = append_successors(program, &state, &mut states, state_cap) {
            return stop;
        }
        cursor += 1;
    }
    Candidate::Complete(states)
}

pub fn check(kernel: &Kernel, launch: &Launch, state_cap: usize) -> Report {
    let program = match strict_memory_admission::project(kernel, launch) {
        Ok(program) => program,
        Err(reason) => {
            return Report::inconclusive(0, state_cap, Reason::Admission { detail: reason });
        }
    };
    match candidate(&program, state_cap) {
        Candidate::Complete(certificate) => {
            let status = status_for_certificate(kernel, launch, &certificate);
            Report {
                scope: "typed_ir_ptx_constant_global_memory",
                status,
                states: certificate.len(),
                state_cap,
                reason: if status.is_safe() {
                    None
                } else {
                    Some(Reason::CertificateRejected)
                },
            }
        }
        Candidate::Unsafe(states) => Report {
            scope: "typed_ir_ptx_constant_global_memory",
            status: Status::Unsafe,
            states,
            state_cap,
            reason: Some(Reason::BadState),
        },
        Candidate::Inconclusive(states, reason) => Report::inconclusive(states, state_cap, reason),
    }
}
