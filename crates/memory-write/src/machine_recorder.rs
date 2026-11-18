/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::barrier::ArrivalStep;
use crate::effects;
use crate::effects::AccessKind;
use crate::effects::Effect;
use crate::effects::FoldMode;
use crate::effects::PhaseKey;
use crate::effects::Resource;
use crate::effects::Summary;
use crate::effects::Visibility;

#[derive(Debug, Default)]
pub struct Recorder {
    pub trace: Option<Vec<Effect>>,
}

impl Clone for Recorder {
    fn clone(&self) -> Self {
        fresh_snapshot(self)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BeginContext {
    pub footprint_empty: Option<bool>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecordContext {
    pub mode: Option<FoldMode>,
    pub slice: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FinishFootprint {
    pub mode: FoldMode,
    pub has_membership: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FinishContext {
    pub footprint: Option<FinishFootprint>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Access {
    pub resource: Resource,
    pub kind: AccessKind,
    pub visibility: Visibility,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Direct {
    Access {
        resource: Resource,
        kind: AccessKind,
    },
    Release(u64),
    Acquire(u64),
    Fault,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Route {
    Off,
    Buffered,
    Direct(Direct),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Delivery {
    pub slice: Option<Access>,
    pub footprint: Route,
}

pub fn new() -> Recorder {
    Recorder { trace: None }
}

pub fn fresh_snapshot(_state: &Recorder) -> Recorder {
    new()
}

pub fn begin(state: &mut Recorder, context: BeginContext) -> bool {
    if let Some(trace) = &mut state.trace {
        trace.push(Effect::Finding);
        return false;
    }
    if let Some(true) = context.footprint_empty {
        state.trace = Some(Vec::new());
        return true;
    }
    false
}

fn route(
    state: &mut Recorder,
    context: RecordContext,
    event: Effect,
    direct: Option<Direct>,
) -> Route {
    if context.mode.is_none() {
        return Route::Off;
    }
    if let Some(trace) = &mut state.trace {
        trace.push(event);
        return Route::Buffered;
    }
    match direct {
        Some(effect) => Route::Direct(effect),
        None => Route::Off,
    }
}

pub fn record_access(state: &mut Recorder, context: RecordContext, event: Access) -> Delivery {
    let direct = match context.mode {
        Some(mode) => {
            if effects::included(mode, event.visibility, event.resource.space) {
                Some(Direct::Access {
                    resource: event.resource,
                    kind: event.kind,
                })
            } else {
                None
            }
        }
        None => None,
    };
    let footprint = route(
        state,
        context,
        Effect::Access {
            resource: event.resource,
            kind: event.kind,
            visibility: event.visibility,
        },
        direct,
    );
    Delivery {
        slice: if context.slice { Some(event) } else { None },
        footprint,
    }
}

pub fn record_release(state: &mut Recorder, context: RecordContext, key: u64) -> Route {
    route(
        state,
        context,
        Effect::Release(key),
        Some(Direct::Release(key)),
    )
}

pub fn record_acquire(state: &mut Recorder, context: RecordContext, key: u64) -> Route {
    route(
        state,
        context,
        Effect::Acquire(key),
        Some(Direct::Acquire(key)),
    )
}

pub fn record_fault(state: &mut Recorder, context: RecordContext) -> Route {
    route(state, context, Effect::Fault, Some(Direct::Fault))
}

pub fn record_finding(state: &mut Recorder, context: RecordContext) -> Route {
    route(state, context, Effect::Finding, None)
}

pub fn record_arrival(
    state: &mut Recorder,
    context: RecordContext,
    key: PhaseKey,
    step: ArrivalStep,
) -> bool {
    if context.mode.is_none() {
        return false;
    }
    if let Some(trace) = &mut state.trace {
        trace.push(Effect::Arrival { key, step });
        return true;
    }
    false
}

pub fn finish(state: &mut Recorder, context: FinishContext, force_write: bool) -> Option<Summary> {
    let trace = std::mem::replace(&mut state.trace, None);
    let Some(mut trace) = trace else {
        return None;
    };
    let Some(footprint) = context.footprint else {
        return None;
    };
    if force_write || footprint.has_membership {
        trace.push(Effect::Finding);
    }
    Some(effects::fold(footprint.mode, &trace))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::barrier::TransactionDelta;
    use crate::effects::ResourceSpace;

    fn access() -> Access {
        Access {
            resource: effects::uniform_register(7),
            kind: AccessKind::Write,
            visibility: Visibility::All,
        }
    }

    fn recording() -> RecordContext {
        RecordContext {
            mode: Some(FoldMode::Full),
            slice: true,
        }
    }

    fn finish_context() -> FinishContext {
        FinishContext {
            footprint: Some(FinishFootprint {
                mode: FoldMode::Full,
                has_membership: false,
            }),
        }
    }

    #[test]
    fn slice_delivery_does_not_require_footprint_and_keeps_sync_only() {
        let mut state = new();
        let mut event = access();
        event.visibility = Visibility::SyncOnly;
        let delivery = record_access(
            &mut state,
            RecordContext {
                mode: None,
                slice: true,
            },
            event,
        );
        assert_eq!(
            delivery,
            Delivery {
                slice: Some(event),
                footprint: Route::Off
            }
        );
        assert_eq!(state.trace, None);
    }

    #[test]
    fn nested_begin_keeps_trace_and_invalidates_arrival_classification() {
        let mut state = new();
        assert!(begin(
            &mut state,
            BeginContext {
                footprint_empty: Some(true)
            }
        ));
        record_release(&mut state, recording(), 9);
        assert!(!begin(
            &mut state,
            BeginContext {
                footprint_empty: None
            }
        ));
        assert_eq!(state.trace, Some(vec![Effect::Release(9), Effect::Finding]));
    }

    #[test]
    fn dirty_begin_uses_direct_route_without_resetting_observations() {
        let mut state = new();
        assert!(!begin(
            &mut state,
            BeginContext {
                footprint_empty: Some(false)
            }
        ));
        assert_eq!(
            record_access(&mut state, recording(), access()).footprint,
            Route::Direct(Direct::Access {
                resource: access().resource,
                kind: AccessKind::Write
            })
        );
        assert_eq!(state.trace, None);
    }

    #[test]
    fn fault_keeps_prefix_and_suffix_and_finish_consumes_trace() {
        let mut state = new();
        begin(
            &mut state,
            BeginContext {
                footprint_empty: Some(true),
            },
        );
        record_access(&mut state, recording(), access());
        record_release(&mut state, recording(), 5);
        record_fault(&mut state, recording());
        record_acquire(&mut state, recording(), 8);
        let summary = finish(&mut state, finish_context(), false).unwrap();
        assert!(summary.all);
        assert_eq!(summary.writes, [(17, 7, 0, 1)]);
        assert_eq!(summary.releases, [5]);
        assert_eq!(summary.acquires, [8]);
        assert_eq!(state.trace, None);
    }

    #[test]
    fn missing_footprint_still_consumes_trace_and_clone_is_fresh() {
        let mut state = new();
        begin(
            &mut state,
            BeginContext {
                footprint_empty: Some(true),
            },
        );
        record_release(&mut state, recording(), 1);
        assert_eq!(state.clone().trace, None);
        assert_eq!(
            finish(&mut state, FinishContext { footprint: None }, true),
            None
        );
        assert_eq!(state.trace, None);
    }

    #[test]
    fn arrivals_require_active_buffer_and_later_findings_make_membership_write() {
        let key = PhaseKey {
            cta: 1,
            offset: 32,
            phase: 0,
        };
        let step = ArrivalStep {
            delta: TransactionDelta::Add(0),
            pending_before: 4,
            arrivals: 1,
            completed: false,
            faulted: false,
        };
        let mut state = new();
        assert!(!record_arrival(&mut state, recording(), key, step));
        begin(
            &mut state,
            BeginContext {
                footprint_empty: Some(true),
            },
        );
        assert!(record_arrival(&mut state, recording(), key, step));
        record_finding(&mut state, recording());
        let summary = finish(&mut state, finish_context(), false).unwrap();
        assert!(summary.reads.is_empty());
        assert_eq!(
            summary.writes,
            [(
                effects::resource_tag(ResourceSpace::MbarMembership),
                1,
                crate::barrier::phase_key(32, 0),
                crate::barrier::phase_key(32, 0) + 1
            )]
        );
    }
}
