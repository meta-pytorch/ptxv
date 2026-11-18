/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ptxv_case::Manifest;
use ptxv_memory_write::checked_execution;
use ptxv_memory_write::execution_core;
use ptxv_memory_write::execution_core::Fault;
use ptxv_memory_write::input_independent_execution;
use ptxv_memory_write::memory_safety;
use ptxv_memory_write::value_flow_dependencies;
use serde::Serialize;

use crate::machine::ExecutionMachine;
use crate::strict_memory_frontend_launch::build_program;
use crate::strict_memory_status::Status;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Reason {
    SourceRefused,
    UnsupportedFront,
    StateCap,
    ExecutionFault,
    BadState,
    CertificateRejected,
    InputDependent,
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
            scope: "raw_ptx_fixed_launch_all_payloads_legal_placements",
            status: Status::Inconclusive,
            states,
            state_cap,
            reason: Some(reason),
        }
    }

    pub fn unsupported_front(state_cap: usize) -> Self {
        Self::inconclusive(0, state_cap, Reason::UnsupportedFront)
    }
}

pub enum Candidate {
    Complete(Vec<input_independent_execution::State>),
    Unsafe(usize),
    Inconclusive(usize, Reason),
}

pub fn check<F>(
    source: &[u8],
    entry: &[u8],
    manifest: &Manifest,
    state_cap: usize,
    producer: F,
) -> Report
where
    F: FnOnce(&execution_core::Program, usize) -> Candidate,
{
    let program = match build_program(source, entry, manifest) {
        Some(program) => program,
        None => return Report::inconclusive(0, state_cap, Reason::SourceRefused),
    };
    match producer(&program, state_cap) {
        Candidate::Complete(certificate) => {
            if input_independent_execution::verify(&program, &certificate) {
                Report {
                    scope: "raw_ptx_fixed_launch_all_payloads_legal_placements",
                    status: Status::Safe,
                    states: certificate.len(),
                    state_cap,
                    reason: None,
                }
            } else {
                Report::inconclusive(certificate.len(), state_cap, Reason::CertificateRejected)
            }
        }
        Candidate::Unsafe(states) => Report {
            scope: "raw_ptx_fixed_launch_all_payloads_legal_placements",
            status: Status::Unsafe,
            states,
            state_cap,
            reason: Some(Reason::BadState),
        },
        Candidate::Inconclusive(states, reason) => Report::inconclusive(states, state_cap, reason),
    }
}

#[derive(Clone)]
struct Node {
    machine: ExecutionMachine<()>,
    monitor: memory_safety::State,
    dependencies: value_flow_dependencies::State,
}

impl Node {
    fn capture(&self) -> Option<input_independent_execution::State> {
        Some(input_independent_execution::State {
            concrete: checked_execution::State {
                execution: execution_core::snapshot_state(self.machine.state()?),
                monitor: self.monitor.clone(),
            },
            dependencies: self.dependencies.clone(),
        })
    }
}

pub fn native_candidate(program: &execution_core::Program, state_cap: usize) -> Candidate {
    if state_cap == 0 {
        return Candidate::Inconclusive(0, Reason::StateCap);
    }
    let Some(initial) = input_independent_execution::initial(program) else {
        return Candidate::Inconclusive(0, Reason::SourceRefused);
    };
    let Some(frame) = checked_execution::monitor_program(program) else {
        return Candidate::Inconclusive(0, Reason::SourceRefused);
    };
    let Ok(machine) = ExecutionMachine::<()>::strict(program.clone()) else {
        return Candidate::Inconclusive(0, Reason::ExecutionFault);
    };
    let mut nodes = vec![Node {
        machine,
        monitor: initial.concrete.monitor.clone(),
        dependencies: initial.dependencies.clone(),
    }];
    let mut states = vec![initial];
    let mut cursor = 0;
    while cursor < nodes.len() {
        if !checked_execution::good(program, &states[cursor].concrete) {
            return Candidate::Unsafe(states.len());
        }
        let mut actor = 0;
        while actor < program.initial.threads.len() {
            if nodes[cursor].machine.enabled(actor, |_, _| false) {
                let mut next = nodes[cursor].clone();
                let event = match next
                    .machine
                    .step(actor, |state, _| (Err(Fault::WrongMode), state))
                {
                    Ok(event) => event,
                    Err(
                        Fault::OutOfBounds
                        | Fault::Misaligned
                        | Fault::Uninitialized
                        | Fault::DeadAllocation,
                    ) => {
                        return Candidate::Unsafe(states.len());
                    }
                    Err(_) => return Candidate::Inconclusive(states.len(), Reason::ExecutionFault),
                };
                let dependencies = match input_independent_execution::advance_dependencies(
                    program,
                    &states[cursor].concrete.execution,
                    &states[cursor].dependencies,
                    actor,
                ) {
                    Some(dependencies) => dependencies,
                    None => return Candidate::Inconclusive(states.len(), Reason::InputDependent),
                };
                next.dependencies = dependencies;
                next.monitor = memory_safety::update_memory(&frame, &next.monitor, event);
                let Some(captured) = next.capture() else {
                    return Candidate::Inconclusive(states.len(), Reason::ExecutionFault);
                };
                if !input_independent_execution::state_present(&states, &captured) {
                    if states.len() >= state_cap {
                        return Candidate::Inconclusive(states.len(), Reason::StateCap);
                    }
                    states.push(captured);
                    nodes.push(next);
                }
            }
            actor += 1;
        }
        cursor += 1;
    }
    Candidate::Complete(states)
}
