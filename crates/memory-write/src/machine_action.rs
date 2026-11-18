/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::barrier::ArrivalStep;
use crate::effects;
use crate::effects::AccessKind;
use crate::effects::PhaseKey;
use crate::machine_footprint;
use crate::machine_footprint::Footprint;
use crate::machine_recorder;
use crate::machine_recorder::Access;
use crate::machine_recorder::Direct;
use crate::machine_recorder::Recorder;
use crate::machine_recorder::Route;
use crate::machine_registers;
use crate::machine_registers::PredicateWrite;
use crate::machine_registers::WarpRegisters;
use crate::machine_registers::WordWrite;
use crate::machine_warp;
use crate::machine_warp::MaskWrite;
use crate::machine_warp::WarpCore;

#[derive(Debug, Default)]
pub struct ActionState {
    pub fp: Option<Box<Footprint>>,
    recorder: Recorder,
}

pub fn new() -> ActionState {
    ActionState {
        fp: None,
        recorder: machine_recorder::new(),
    }
}

pub fn is_recording(state: &ActionState) -> bool {
    state.recorder.trace.is_some()
}

pub fn snapshot(state: &ActionState) -> ActionState {
    let fp = match &state.fp {
        Some(footprint) => Some(Box::new((**footprint).clone())),
        None => None,
    };
    ActionState {
        fp,
        recorder: machine_recorder::fresh_snapshot(&state.recorder),
    }
}

impl Clone for ActionState {
    fn clone(&self) -> Self {
        snapshot(self)
    }
}

pub fn set_footprint(state: &mut ActionState, footprint: Option<Box<Footprint>>) {
    state.fp = footprint;
}

pub fn take_footprint(state: &mut ActionState) -> Option<Box<Footprint>> {
    std::mem::replace(&mut state.fp, None)
}

fn footprint(state: &ActionState) -> Option<&Footprint> {
    match &state.fp {
        Some(footprint) => Some(footprint),
        None => None,
    }
}

pub fn begin(state: &mut ActionState) -> bool {
    let context = machine_footprint::begin_context(footprint(state));
    machine_recorder::begin(&mut state.recorder, context)
}

fn apply_route(state: &mut ActionState, route: Route) {
    let Route::Direct(effect) = route else {
        return;
    };
    let Some(footprint) = &mut state.fp else {
        panic!("recorder routed a direct effect without a footprint");
    };
    match effect {
        Direct::Access { resource, kind } => {
            let (space, owner, start, end) = effects::resource_range(resource);
            footprint.add(space, owner, start, end, kind == AccessKind::Write);
        }
        Direct::Release(key) => footprint.rel.push(key),
        Direct::Acquire(key) => footprint.acq.push(key),
        Direct::Fault => footprint.all = true,
    }
}

pub fn record_access(state: &mut ActionState, slice: bool, access: Access) -> Option<Access> {
    let context = machine_footprint::record_context(footprint(state), slice);
    let delivery = machine_recorder::record_access(&mut state.recorder, context, access);
    apply_route(state, delivery.footprint);
    delivery.slice
}

pub fn record_release(state: &mut ActionState, key: u64) {
    let context = machine_footprint::record_context(footprint(state), false);
    let route = machine_recorder::record_release(&mut state.recorder, context, key);
    apply_route(state, route);
}

pub fn record_acquire(state: &mut ActionState, key: u64) {
    let context = machine_footprint::record_context(footprint(state), false);
    let route = machine_recorder::record_acquire(&mut state.recorder, context, key);
    apply_route(state, route);
}

pub fn record_fault(state: &mut ActionState) {
    let context = machine_footprint::record_context(footprint(state), false);
    let route = machine_recorder::record_fault(&mut state.recorder, context);
    apply_route(state, route);
}

pub fn record_finding(state: &mut ActionState) {
    let context = machine_footprint::record_context(footprint(state), false);
    let route = machine_recorder::record_finding(&mut state.recorder, context);
    apply_route(state, route);
}

pub fn record_arrival(state: &mut ActionState, key: PhaseKey, step: ArrivalStep) -> bool {
    let context = machine_footprint::record_context(footprint(state), false);
    machine_recorder::record_arrival(&mut state.recorder, context, key, step)
}

pub fn finish(state: &mut ActionState, force_write: bool) {
    let context = machine_footprint::finish_context(footprint(state));
    if let Some(summary) = machine_recorder::finish(&mut state.recorder, context, force_write) {
        let Some(footprint) = &mut state.fp else {
            panic!("recorder produced a summary without a footprint");
        };
        machine_footprint::merge_summary(footprint, &summary);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UniformOp {
    Word(WordWrite),
    Predicate(PredicateWrite),
    PoisonWord(u16),
    PoisonPredicate(u16),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct UniformRequest {
    pub warp: u32,
    pub op: UniformOp,
}

pub fn write_uniform(
    registers: &mut WarpRegisters,
    state: &mut ActionState,
    slice: bool,
    request: UniformRequest,
) -> Option<Access> {
    let access = match request.op {
        UniformOp::Word(write) => machine_registers::write_uniform(registers, request.warp, write),
        UniformOp::Predicate(write) => {
            machine_registers::write_uniform_predicate(registers, request.warp, write)
        }
        UniformOp::PoisonWord(index) => {
            machine_registers::poison_uniform(registers, request.warp, index)
        }
        UniformOp::PoisonPredicate(index) => {
            machine_registers::poison_uniform_predicate(registers, request.warp, index)
        }
    };
    match access {
        Some(access) => record_access(state, slice, access),
        None => None,
    }
}

pub fn write_mask(
    warp: &mut WarpCore,
    state: &mut ActionState,
    slice: bool,
    write: MaskWrite,
) -> Option<Access> {
    match machine_warp::write_mask(warp, write) {
        Some(access) => record_access(state, slice, access),
        None => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::effects::Visibility;

    fn access() -> Access {
        Access {
            resource: effects::uniform_register(7),
            kind: AccessKind::Write,
            visibility: Visibility::SyncOnly,
        }
    }

    #[test]
    fn slice_observation_survives_disabled_or_filtered_footprints() {
        let mut state = new();
        assert_eq!(record_access(&mut state, true, access()), Some(access()));
        assert!(state.fp.is_none());
        set_footprint(&mut state, Some(Box::default()));
        assert_eq!(record_access(&mut state, true, access()), Some(access()));
        assert!(state.fp.as_deref().unwrap().is_empty());
        assert_eq!(state.recorder.trace, None);
    }

    #[test]
    fn buffered_fault_keeps_observations_and_snapshot_starts_another_action() {
        let mut state = new();
        set_footprint(&mut state, Some(Box::default()));
        assert!(begin(&mut state));
        let mut event = access();
        event.visibility = Visibility::All;
        assert_eq!(record_access(&mut state, true, event), Some(event));
        record_release(&mut state, 9);
        record_fault(&mut state);
        let mut fork = snapshot(&state);
        assert!(begin(&mut fork));
        finish(&mut fork, false);
        assert!(fork.fp.as_ref().unwrap().is_empty());
        assert!(fork.fp.as_ref().unwrap().rel.is_empty());
        finish(&mut state, false);
        let footprint = state.fp.as_deref().unwrap();
        assert!(footprint.all);
        assert_eq!(footprint.w, [(17, 7, 0, 1)]);
        assert_eq!(footprint.rel, [9]);
        assert_eq!(state.recorder.trace, None);
    }

    #[test]
    fn disappearing_footprint_still_consumes_buffered_trace() {
        let mut state = new();
        set_footprint(&mut state, Some(Box::default()));
        assert!(begin(&mut state));
        record_release(&mut state, 19);
        assert!(take_footprint(&mut state).is_some());
        finish(&mut state, false);
        assert_eq!(state.recorder.trace, None);
        set_footprint(&mut state, Some(Box::default()));
        assert!(begin(&mut state));
        finish(&mut state, false);
        assert!(state.fp.as_deref().unwrap().rel.is_empty());
    }
}
