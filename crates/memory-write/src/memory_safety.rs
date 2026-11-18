/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Conservative memory-event semantics and a closed-state certificate checker.
//! This module does not establish that native PTX lowering covers every behavior.

/// One nonaliasing allocation present at program entry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Allocation {
    pub base: u64,
    pub bytes: u64,
    pub initially_defined: bool,
}

/// Stores carry the definedness of their payload, independently of its value.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessKind {
    Load,
    Store { defined: bool },
}

/// An instruction's complete physical byte range and required alignment.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Request {
    pub address: u64,
    pub width: u64,
    pub alignment: u64,
    pub kind: AccessKind,
}

/// Unsupported instructions have no accepting interpretation.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Instruction {
    Access(Request),
    Free(usize),
    Choice(Vec<usize>),
    Halt,
    Skip,
    Unsupported,
}

/// Each thread executes its own instruction sequence; every interleaving is allowed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    pub allocations: Vec<Allocation>,
    pub threads: Vec<Vec<Instruction>>,
}

/// Latest overlapping stores determine current byte definedness.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WrittenSpan {
    pub address: u64,
    pub width: u64,
    pub defined: bool,
}

/// Every access is retained, including unused loads and same-value stores.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AccessRecord {
    pub thread: usize,
    pub address: u64,
    pub width: u64,
    pub is_write: bool,
}

/// A certificate contains actual states, compared structurally without hashes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct State {
    pub pcs: Vec<usize>,
    pub live: Vec<bool>,
    pub writes: Vec<WrittenSpan>,
    pub history: Vec<AccessRecord>,
    pub bad: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Event {
    Access { thread: usize, request: Request },
    Free { thread: usize, allocation: usize },
    Control,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Move {
    pub event: Event,
    pub pcs: Vec<usize>,
}

fn overlaps(a: u64, aw: u64, b: u64, bw: u64) -> bool {
    aw != 0
        && bw != 0
        && (a as u128) < b as u128 + bw as u128
        && (b as u128) < a as u128 + aw as u128
}

/// Reject overlapping physical allocations and ranges beyond the address domain.
pub fn layout_valid(program: &Program) -> bool {
    let mut valid = true;
    let mut i = 0;
    while i < program.allocations.len() && valid {
        let a = &program.allocations[i];
        if a.bytes == 0 || a.base as u128 + a.bytes as u128 > u64::MAX as u128 + 1 {
            valid = false;
        }
        let mut j = 0;
        while j < i && valid {
            let b = &program.allocations[j];
            if overlaps(a.base, a.bytes, b.base, b.bytes) {
                valid = false;
            }
            j += 1;
        }
        i += 1;
    }
    valid
}

/// Initial state is derived from the program, never supplied by a certificate.
pub fn initial(program: &Program) -> Option<State> {
    if !layout_valid(program) {
        return None;
    }
    Some(State {
        pcs: vec![0; program.threads.len()],
        live: vec![true; program.allocations.len()],
        writes: Vec::new(),
        history: Vec::new(),
        bad: false,
    })
}

fn contains(base: u64, width: u64, address: u64) -> bool {
    base <= address && (address as u128) < base as u128 + width as u128
}

fn byte_defined(program: &Program, state: &State, address: u64) -> bool {
    let mut found = false;
    let mut defined = false;
    let mut i = state.writes.len();
    while i > 0 && !found {
        i -= 1;
        let w = state.writes[i];
        if contains(w.address, w.width, address) {
            defined = w.defined;
            found = true;
        }
    }
    let mut j = 0;
    while j < program.allocations.len() && !found {
        let a = &program.allocations[j];
        if contains(a.base, a.bytes, address) {
            defined = a.initially_defined;
            found = true;
        }
        j += 1;
    }
    defined
}

/// Check a memory event before advancing the PC or changing memory metadata.
pub fn access_valid(program: &Program, state: &State, thread: usize, request: Request) -> bool {
    if state.live.len() != program.allocations.len()
        || request.width == 0
        || request.alignment == 0
        || request.address % request.alignment != 0
        || request.address as u128 + request.width as u128 > u64::MAX as u128 + 1
    {
        return false;
    }
    let mut found = false;
    let mut i = 0;
    while i < program.allocations.len() {
        let a = &program.allocations[i];
        if a.base <= request.address {
            let offset = request.address - a.base;
            if state.live[i] && offset <= a.bytes && request.width <= a.bytes - offset {
                found = true;
            }
        }
        i += 1;
    }
    if !found {
        return false;
    }
    let is_write = matches!(request.kind, AccessKind::Store { .. });
    let mut valid = true;
    let mut j = 0;
    while j < state.history.len() && valid {
        let h = state.history[j];
        if h.thread != thread
            && (h.is_write || is_write)
            && overlaps(h.address, h.width, request.address, request.width)
        {
            valid = false;
        }
        j += 1;
    }
    if !is_write {
        let mut offset = 0;
        while offset < request.width && valid {
            match request.address.checked_add(offset) {
                None => valid = false,
                Some(address) => valid = byte_defined(program, state, address),
            }
            offset += 1;
        }
    }
    valid
}

pub fn thread_moves(program: &Program, state: &State, thread: usize) -> Option<Vec<Move>> {
    let mut result = Vec::new();
    let code = &program.threads[thread];
    let pc = state.pcs[thread];
    if pc > code.len() {
        return None;
    }
    if pc < code.len() {
        match &code[pc] {
            Instruction::Unsupported => return None,
            Instruction::Choice(targets) => {
                let count = targets.len();
                if count == 0 {
                    return None;
                }
                let mut i = 0;
                let mut known = true;
                while i < count && known {
                    let target = targets[i];
                    if target > code.len() {
                        known = false;
                    } else {
                        let mut pcs = state.pcs.clone();
                        pcs[thread] = target;
                        result.push(Move {
                            event: Event::Control,
                            pcs,
                        });
                    }
                    i += 1;
                }
                if !known {
                    return None;
                }
            }
            Instruction::Halt => {
                let mut pcs = state.pcs.clone();
                pcs[thread] = code.len();
                result.push(Move {
                    event: Event::Control,
                    pcs,
                });
            }
            Instruction::Skip => {
                let next_pc = pc + 1;
                let mut pcs = state.pcs.clone();
                pcs[thread] = next_pc;
                result.push(Move {
                    event: Event::Control,
                    pcs,
                });
            }
            Instruction::Free(allocation) => {
                let next_pc = pc + 1;
                let mut pcs = state.pcs.clone();
                pcs[thread] = next_pc;
                result.push(Move {
                    event: Event::Free {
                        thread,
                        allocation: *allocation,
                    },
                    pcs,
                });
            }
            Instruction::Access(request) => {
                let next_pc = pc + 1;
                let mut pcs = state.pcs.clone();
                pcs[thread] = next_pc;
                result.push(Move {
                    event: Event::Access {
                        thread,
                        request: *request,
                    },
                    pcs,
                });
            }
        }
    }
    Some(result)
}

pub fn moves(program: &Program, state: &State) -> Option<Vec<Move>> {
    if state.pcs.len() != program.threads.len() || state.live.len() != program.allocations.len() {
        return None;
    }
    if state.bad {
        return Some(Vec::new());
    }
    let mut result = Vec::new();
    let mut known = true;
    let mut thread = 0;
    while thread < program.threads.len() && known {
        match thread_moves(program, state, thread) {
            None => known = false,
            Some(next) => {
                let mut i = 0;
                while i < next.len() {
                    result.push(next[i].clone());
                    i += 1;
                }
            }
        }
        thread += 1;
    }
    if known { Some(result) } else { None }
}

pub fn update_memory(program: &Program, state: &State, event: Event) -> State {
    let mut next = state.clone();
    match event {
        Event::Control => {}
        Event::Access { thread, request } => {
            if !access_valid(program, state, thread, request) {
                next.bad = true;
            } else {
                if let AccessKind::Store { defined } = request.kind {
                    next.writes.push(WrittenSpan {
                        address: request.address,
                        width: request.width,
                        defined,
                    });
                }
                next.history.push(AccessRecord {
                    thread,
                    address: request.address,
                    width: request.width,
                    is_write: matches!(request.kind, AccessKind::Store { .. }),
                });
            }
        }
        Event::Free { thread, allocation } => {
            if allocation >= next.live.len()
                || allocation >= program.allocations.len()
                || !next.live[allocation]
            {
                next.bad = true;
            } else {
                let a = &program.allocations[allocation];
                let request = Request {
                    address: a.base,
                    width: a.bytes,
                    alignment: 1,
                    kind: AccessKind::Store { defined: false },
                };
                if !access_valid(program, state, thread, request) {
                    next.bad = true;
                } else {
                    next.history.push(AccessRecord {
                        thread,
                        address: a.base,
                        width: a.bytes,
                        is_write: true,
                    });
                    next.live[allocation] = false;
                }
            }
        }
    }
    next
}

pub fn apply_move(program: &Program, state: &State, movement: &Move) -> State {
    let mut next = update_memory(program, state, movement.event);
    if !next.bad {
        next.pcs = movement.pcs.clone();
    }
    next
}

/// Recompute every enabled thread and every branch. None means unknown semantics.
pub fn successors(program: &Program, state: &State) -> Option<Vec<State>> {
    let Some(choices) = moves(program, state) else {
        return None;
    };
    let mut result = Vec::new();
    let mut i = 0;
    while i < choices.len() {
        result.push(apply_move(program, state, &choices[i]));
        i += 1;
    }
    Some(result)
}

fn state_present(states: &[State], state: &State) -> bool {
    let mut found = false;
    let mut i = 0;
    while i < states.len() && !found {
        if states[i] == *state {
            found = true;
        }
        i += 1;
    }
    found
}

fn state_closed(program: &Program, certificate: &[State], state: &State) -> bool {
    if state.bad {
        return false;
    }
    let Some(next) = successors(program, state) else {
        return false;
    };
    let mut closed = true;
    let mut i = 0;
    while i < next.len() && closed {
        closed = state_present(certificate, &next[i]);
        i += 1;
    }
    closed
}

/// Accept only a safe, exact, successor-closed set containing the derived entry state.
/// Truncation, unsupported semantics and omitted branches cannot establish acceptance.
pub fn verify(program: &Program, certificate: &[State]) -> bool {
    let Some(start) = initial(program) else {
        return false;
    };
    if !state_present(certificate, &start) {
        return false;
    }
    let mut accepted = true;
    let mut i = 0;
    while i < certificate.len() && accepted {
        accepted = state_closed(program, certificate, &certificate[i]);
        i += 1;
    }
    accepted
}

#[cfg(test)]
mod tests {
    use super::*;

    fn access(address: u64, kind: AccessKind) -> Instruction {
        Instruction::Access(Request {
            address,
            width: 4,
            alignment: 4,
            kind,
        })
    }

    fn program(threads: Vec<Vec<Instruction>>) -> Program {
        Program {
            allocations: vec![Allocation {
                base: 4096,
                bytes: 16,
                initially_defined: false,
            }],
            threads,
        }
    }

    fn candidate(program: &Program, cap: usize) -> Vec<State> {
        let Some(start) = initial(program) else {
            return Vec::new();
        };
        let mut states = vec![start];
        let mut index = 0;
        while index < states.len() && states.len() < cap {
            if let Some(next) = successors(program, &states[index]) {
                for state in next {
                    if !states.contains(&state) {
                        states.push(state);
                    }
                }
            }
            index += 1;
        }
        states
    }

    #[test]
    fn private_stores_then_loads_are_safe_in_every_interleaving() {
        let p = program(vec![
            vec![
                access(4096, AccessKind::Store { defined: true }),
                access(4096, AccessKind::Load),
            ],
            vec![
                access(4100, AccessKind::Store { defined: true }),
                access(4100, AccessKind::Load),
            ],
        ]);
        let c = candidate(&p, 1000);
        assert!(c.len() > 4);
        assert!(verify(&p, &c));
        assert!(!verify(&p, &c[..c.len() - 1]));
        assert!(!verify(&p, &candidate(&p, 2)));
    }

    #[test]
    fn dead_uninitialized_reads_and_same_value_writes_are_not_safe() {
        let dead = program(vec![vec![
            access(4096, AccessKind::Load),
            Instruction::Halt,
        ]]);
        assert!(!verify(&dead, &candidate(&dead, 100)));
        let races = program(vec![
            vec![access(4096, AccessKind::Store { defined: true })],
            vec![access(4096, AccessKind::Store { defined: true })],
        ]);
        assert!(!verify(&races, &candidate(&races, 100)));
    }

    #[test]
    fn rejects_each_spatial_initialization_and_lifetime_fault() {
        let cases = [
            vec![access(4112, AccessKind::Load)],
            vec![access(4097, AccessKind::Store { defined: true })],
            vec![
                access(4096, AccessKind::Store { defined: false }),
                access(4096, AccessKind::Load),
            ],
            vec![
                Instruction::Free(0),
                access(4096, AccessKind::Store { defined: true }),
            ],
            vec![Instruction::Free(0), Instruction::Free(0)],
        ];
        for code in cases {
            let p = program(vec![code]);
            assert!(!verify(&p, &candidate(&p, 100)));
        }
    }

    #[test]
    fn a_bad_prefix_cannot_be_hidden_by_a_safe_terminal_state() {
        let p = program(vec![vec![
            Instruction::Choice(vec![1, 2]),
            Instruction::Halt,
            access(4096, AccessKind::Load),
        ]]);
        let c = candidate(&p, 100);
        let pruned: Vec<_> = c.into_iter().filter(|s| !s.bad).collect();
        assert!(!verify(&p, &pruned));
    }

    #[test]
    fn unsupported_semantics_and_forged_layouts_never_pass() {
        for op in [
            Instruction::Unsupported,
            Instruction::Choice(vec![]),
            Instruction::Choice(vec![99]),
        ] {
            let p = program(vec![vec![op]]);
            assert!(!verify(&p, &candidate(&p, 100)));
        }
        let mut p = program(vec![]);
        p.allocations.push(p.allocations[0].clone());
        assert!(!verify(&p, &[]));
        p.allocations = vec![Allocation {
            base: u64::MAX,
            bytes: 2,
            initially_defined: true,
        }];
        assert!(!verify(&p, &[]));
    }

    #[test]
    fn top_address_and_latest_undefined_store_are_checked_exactly() {
        let p = Program {
            allocations: vec![Allocation {
                base: u64::MAX,
                bytes: 1,
                initially_defined: true,
            }],
            threads: vec![vec![Instruction::Access(Request {
                address: u64::MAX,
                width: 1,
                alignment: 1,
                kind: AccessKind::Load,
            })]],
        };
        assert!(verify(&p, &candidate(&p, 100)));
        let p = program(vec![vec![
            access(4096, AccessKind::Store { defined: true }),
            Instruction::Access(Request {
                address: 4097,
                width: 1,
                alignment: 1,
                kind: AccessKind::Store { defined: false },
            }),
            access(4096, AccessKind::Load),
        ]]);
        assert!(!verify(&p, &candidate(&p, 100)));
    }

    #[test]
    fn lifetime_end_conflicts_with_prior_peer_access() {
        let mut p = program(vec![
            vec![access(4096, AccessKind::Load)],
            vec![Instruction::Free(0)],
        ]);
        p.allocations[0].initially_defined = true;
        let start = initial(&p).unwrap();
        let read_first = successors(&p, &start).unwrap().remove(0);
        let after_free = successors(&p, &read_first).unwrap();
        assert!(after_free.iter().any(|state| state.bad));
        assert!(!verify(&p, &candidate(&p, 100)));
        let own = program(vec![vec![
            access(4096, AccessKind::Store { defined: true }),
            Instruction::Free(0),
        ]]);
        assert!(verify(&own, &candidate(&own, 100)));
    }

    #[test]
    fn move_enumeration_leaves_memory_untouched_and_bad_application_keeps_pc() {
        let p = program(vec![vec![access(4096, AccessKind::Load)]]);
        let start = initial(&p).unwrap();
        let choices = moves(&p, &start).unwrap();
        assert_eq!(choices.len(), 1);
        assert_eq!(choices[0].pcs, vec![1]);
        assert_eq!(start, initial(&p).unwrap());
        let next = apply_move(&p, &start, &choices[0]);
        assert!(next.bad);
        assert_eq!(next.pcs, start.pcs);
        assert_eq!(next.history, start.history);
        assert_eq!(next.writes, start.writes);
        let control = program(vec![vec![Instruction::Choice(vec![0, 1])]]);
        let state = initial(&control).unwrap();
        let choices = moves(&control, &state).unwrap();
        assert_eq!(
            choices
                .iter()
                .map(|movement| movement.pcs[0])
                .collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert!(
            choices
                .iter()
                .all(|movement| movement.event == Event::Control)
        );
        let finished = apply_move(&control, &state, &choices[1]);
        assert!(!finished.bad);
        assert_eq!(finished.pcs, vec![1]);
        assert_eq!(finished.live, state.live);
        assert_eq!(finished.history, state.history);
        assert_eq!(finished.writes, state.writes);
    }

    #[test]
    fn skip_preserves_native_pc_and_memory() {
        let p = program(vec![vec![Instruction::Skip, Instruction::Halt]]);
        let start = initial(&p).unwrap();
        let choices = moves(&p, &start).unwrap();
        assert_eq!(
            choices,
            vec![Move {
                event: Event::Control,
                pcs: vec![1]
            }]
        );
        let next = apply_move(&p, &start, &choices[0]);
        assert_eq!(next.pcs, vec![1]);
        assert_eq!(next.live, start.live);
        assert_eq!(next.writes, start.writes);
        assert_eq!(next.history, start.history);
        assert!(!next.bad);
        assert!(verify(&p, &candidate(&p, 100)));
    }
}
