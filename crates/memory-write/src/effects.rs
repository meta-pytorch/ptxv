/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Action-local events and their projection into transition footprints. Barrier membership
//! is classified after the complete trace, so a later finding cannot leave an earlier
//! arrival incorrectly marked as independent.

use crate::barrier;
use crate::footprint::Range;

/// Disjoint resource classes with a stable encoding for existing footprint consumers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResourceSpace {
    Shared,
    Global,
    Tmem,
    TmemAllocator,
    CtaExit,
    BulkGroup,
    MbarPhase,
    MbarArrivalPhase,
    WarpPc,
    GridGate,
    StrongShared,
    StrongGlobal,
    MbarPhaseKey,
    ClusterBarrier,
    NamedBarrier,
    MbarInitialization,
    MbarPendingCount,
    UniformRegisters,
    MbarMembership,
    ConvergenceBarrier,
    L1Line,
    L1Sm,
    TensorCoreACollector,
    MbarOverArrival,
    NicQueue,
    /// Remote-operation pipes (`SPACE_REMOTE_PIPE`, MISS 222 / 232).
    RemotePipe,
    CtaAll,
}

pub fn resource_tag(space: ResourceSpace) -> u8 {
    match space {
        ResourceSpace::Shared => 0,
        ResourceSpace::Global => 1,
        ResourceSpace::Tmem => 2,
        ResourceSpace::TmemAllocator => 3,
        ResourceSpace::CtaExit => 4,
        ResourceSpace::BulkGroup => 5,
        ResourceSpace::MbarPhase => 6,
        ResourceSpace::MbarArrivalPhase => 7,
        ResourceSpace::WarpPc => 8,
        ResourceSpace::GridGate => 9,
        ResourceSpace::StrongShared => 10,
        ResourceSpace::StrongGlobal => 11,
        ResourceSpace::MbarPhaseKey => 12,
        ResourceSpace::ClusterBarrier => 13,
        ResourceSpace::NamedBarrier => 14,
        ResourceSpace::MbarInitialization => 15,
        ResourceSpace::MbarPendingCount => 16,
        ResourceSpace::UniformRegisters => 17,
        ResourceSpace::MbarMembership => 18,
        ResourceSpace::ConvergenceBarrier => 19,
        ResourceSpace::L1Line => 20,
        ResourceSpace::L1Sm => 21,
        ResourceSpace::TensorCoreACollector => 24,
        ResourceSpace::MbarOverArrival => 25,
        ResourceSpace::NicQueue => 26,
        ResourceSpace::RemotePipe => 27,
        ResourceSpace::CtaAll => 98,
    }
}

/// A half-open address or key range, qualified by resource class and owner.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Resource {
    pub space: ResourceSpace,
    pub owner: u32,
    pub start: u64,
    pub end: u64,
}

pub fn resource_range(resource: Resource) -> Range {
    (
        resource_tag(resource.space),
        resource.owner,
        resource.start,
        resource.end,
    )
}

pub fn uniform_register(warp: u32) -> Resource {
    Resource {
        space: ResourceSpace::UniformRegisters,
        owner: warp,
        start: 0,
        end: 1,
    }
}

pub fn convergence_barrier(warp: u32, barrier: u8) -> Resource {
    let start = u64::from(barrier);
    Resource {
        space: ResourceSpace::ConvergenceBarrier,
        owner: warp,
        start,
        end: start + 1,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AccessKind {
    Read,
    Write,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Visibility {
    All,
    SyncOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FoldMode {
    Full,
    SyncOnly,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PhaseKey {
    pub cta: u16,
    pub offset: u32,
    pub phase: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Effect {
    Access {
        resource: Resource,
        kind: AccessKind,
        visibility: Visibility,
    },
    Release(u64),
    Acquire(u64),
    Arrival {
        key: PhaseKey,
        step: barrier::ArrivalStep,
    },
    Finding,
    /// Conflicts with every transition without discarding earlier or later effects.
    Fault,
}

/// Unnormalized ranges and synchronization keys. Consumers retain their existing sorting,
/// coalescing and deduplication before conflict queries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Summary {
    pub reads: Vec<Range>,
    pub writes: Vec<Range>,
    pub releases: Vec<u64>,
    pub acquires: Vec<u64>,
    pub all: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArrivalEntry {
    pub key: PhaseKey,
    pub batch: barrier::ArrivalBatch,
}

/// Incremental state for one scheduler action. Membership entries remain deferred until
/// `finish` sees every event, including later faults and raw resource accesses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FoldState {
    pub mode: FoldMode,
    pub summary: Summary,
    pub arrivals: Vec<ArrivalEntry>,
    pub invalidated: bool,
}

pub fn begin(mode: FoldMode) -> FoldState {
    FoldState {
        mode,
        summary: Summary {
            reads: Vec::new(),
            writes: Vec::new(),
            releases: Vec::new(),
            acquires: Vec::new(),
            all: false,
        },
        arrivals: Vec::new(),
        invalidated: false,
    }
}

pub fn included(mode: FoldMode, visibility: Visibility, space: ResourceSpace) -> bool {
    if mode == FoldMode::Full && visibility == Visibility::SyncOnly {
        return false;
    }
    !(mode == FoldMode::SyncOnly
        && matches!(
            space,
            ResourceSpace::Shared
                | ResourceSpace::Global
                | ResourceSpace::Tmem
                | ResourceSpace::MbarPhase
                | ResourceSpace::WarpPc
        ))
}

pub fn poisons_arrivals(space: ResourceSpace, kind: AccessKind) -> bool {
    space == ResourceSpace::MbarMembership
        || space == ResourceSpace::MbarInitialization && kind == AccessKind::Write
}

fn find_arrival(entries: &[ArrivalEntry], key: PhaseKey) -> Option<usize> {
    let mut index = 0;
    while index < entries.len() && entries[index].key != key {
        index += 1;
    }
    if index < entries.len() {
        Some(index)
    } else {
        None
    }
}

pub fn record_arrival(
    mut state: FoldState,
    key: PhaseKey,
    step: barrier::ArrivalStep,
) -> FoldState {
    let index = match find_arrival(&state.arrivals, key) {
        Some(index) => index,
        None => {
            state.invalidated |= state.arrivals.len() != 0;
            state.arrivals.push(ArrivalEntry {
                key,
                batch: barrier::arrival_batch(step.pending_before),
            });
            state.arrivals.len() - 1
        }
    };
    let entry = &mut state.arrivals[index];
    entry.batch = barrier::record_arrival_batch(entry.batch, step);
    state
}

pub fn apply(mut state: FoldState, effect: Effect) -> FoldState {
    match effect {
        Effect::Access {
            resource,
            kind,
            visibility,
        } => {
            state.invalidated |= poisons_arrivals(resource.space, kind);
            if included(state.mode, visibility, resource.space) {
                let range = resource_range(resource);
                match kind {
                    AccessKind::Read => state.summary.reads.push(range),
                    AccessKind::Write => state.summary.writes.push(range),
                }
            }
            state
        }
        Effect::Release(key) => {
            state.summary.releases.push(key);
            state
        }
        Effect::Acquire(key) => {
            state.summary.acquires.push(key);
            state
        }
        Effect::Arrival { key, step } => record_arrival(state, key, step),
        Effect::Finding => {
            state.invalidated = true;
            state
        }
        Effect::Fault => {
            state.invalidated = true;
            state.summary.all = true;
            state
        }
    }
}

/// Resume collection without publishing a partial action's membership classification.
pub fn fold_from(mut state: FoldState, trace: &[Effect]) -> FoldState {
    let mut index = 0;
    while index < trace.len() {
        state = apply(state, trace[index]);
        index += 1;
    }
    state
}

/// Publish the summary after the whole action, applying its final membership policy.
pub fn finish(mut state: FoldState) -> Summary {
    let mut index = 0;
    while index < state.arrivals.len() {
        let entry = state.arrivals[index];
        let batch = if state.invalidated {
            barrier::invalidate_arrival_batch(entry.batch)
        } else {
            entry.batch
        };
        let start = barrier::phase_key(entry.key.offset, entry.key.phase);
        let range = resource_range(Resource {
            space: ResourceSpace::MbarMembership,
            owner: u32::from(entry.key.cta),
            start,
            end: start + 1,
        });
        if barrier::arrival_batch_writes(batch) {
            state.summary.writes.push(range);
        } else {
            state.summary.reads.push(range);
        }
        index += 1;
    }
    state.summary
}

/// Summarize one complete action; faults add dependence rather than roll effects back.
pub fn fold(mode: FoldMode, trace: &[Effect]) -> Summary {
    finish(fold_from(begin(mode), trace))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resource(space: ResourceSpace) -> Resource {
        Resource {
            space,
            owner: 3,
            start: 4,
            end: 8,
        }
    }

    fn access(space: ResourceSpace, kind: AccessKind) -> Effect {
        Effect::Access {
            resource: resource(space),
            kind,
            visibility: Visibility::All,
        }
    }

    fn arrival(offset: u32, pending: u32, count: u32) -> Effect {
        Effect::Arrival {
            key: PhaseKey {
                cta: 3,
                offset,
                phase: 0,
            },
            step: barrier::ArrivalStep {
                delta: barrier::TransactionDelta::Add(0),
                pending_before: pending,
                arrivals: count,
                completed: false,
                faulted: false,
            },
        }
    }

    #[test]
    fn faults_retain_prefix_and_later_effects() {
        let effects = [
            access(ResourceSpace::Global, AccessKind::Write),
            Effect::Release(7),
            Effect::Fault,
            access(ResourceSpace::ConvergenceBarrier, AccessKind::Read),
            Effect::Acquire(9),
        ];
        let out = fold(FoldMode::Full, &effects);
        assert!(out.all);
        assert_eq!(
            out.writes,
            [resource_range(resource(ResourceSpace::Global))]
        );
        assert_eq!(
            out.reads,
            [resource_range(resource(ResourceSpace::ConvergenceBarrier))]
        );
        assert_eq!(out.releases, [7]);
        assert_eq!(out.acquires, [9]);
        for split in 0..=effects.len() {
            assert_eq!(
                fold_from(begin(FoldMode::Full), &effects),
                fold_from(
                    fold_from(begin(FoldMode::Full), &effects[..split]),
                    &effects[split..],
                )
            );
        }
    }

    #[test]
    fn membership_budget_covers_the_entire_trace() {
        let read = fold(FoldMode::Full, &[arrival(32, 4, 1), arrival(32, 3, 1)]);
        assert_eq!(read.reads.len(), 1);
        assert!(read.writes.is_empty());
        let oversubscribed = fold(FoldMode::Full, &[arrival(32, 4, 1), arrival(32, 3, 2)]);
        assert!(oversubscribed.reads.is_empty());
        assert_eq!(oversubscribed.writes.len(), 1);
        for poison in [
            Effect::Finding,
            Effect::Fault,
            access(ResourceSpace::MbarInitialization, AccessKind::Write),
            access(ResourceSpace::MbarMembership, AccessKind::Read),
        ] {
            for effects in [[poison, arrival(32, 4, 1)], [arrival(32, 4, 1), poison]] {
                assert!(!fold(FoldMode::Full, &effects).writes.is_empty());
            }
        }
        let mixed = fold(FoldMode::Full, &[arrival(32, 4, 1), arrival(40, 4, 1)]);
        assert!(mixed.reads.is_empty());
        assert_eq!(mixed.writes.len(), 2);
    }

    #[test]
    fn sync_only_visibility_and_filtering_match_native_policy() {
        let sync = Effect::Access {
            resource: resource(ResourceSpace::ClusterBarrier),
            kind: AccessKind::Read,
            visibility: Visibility::SyncOnly,
        };
        assert!(fold(FoldMode::Full, &[sync]).reads.is_empty());
        assert_eq!(fold(FoldMode::SyncOnly, &[sync]).reads.len(), 1);
        let ordinary = access(ResourceSpace::Global, AccessKind::Read);
        assert_eq!(fold(FoldMode::Full, &[ordinary]).reads.len(), 1);
        assert!(fold(FoldMode::SyncOnly, &[ordinary]).reads.is_empty());
        let hidden_poison = Effect::Access {
            resource: resource(ResourceSpace::MbarMembership),
            kind: AccessKind::Read,
            visibility: Visibility::SyncOnly,
        };
        let out = fold(FoldMode::Full, &[arrival(32, 4, 1), hidden_poison]);
        assert!(out.reads.is_empty());
        assert_eq!(out.writes.len(), 1);
    }

    #[test]
    fn tensor_core_collector_retains_native_sync_visibility() {
        let resource = Resource {
            space: ResourceSpace::TensorCoreACollector,
            owner: 3,
            start: 0,
            end: 1,
        };
        assert_eq!(resource_range(resource), (24, 3, 0, 1));
        for mode in [FoldMode::Full, FoldMode::SyncOnly] {
            for visibility in [Visibility::All, Visibility::SyncOnly] {
                let out = fold(
                    mode,
                    &[Effect::Access {
                        resource,
                        kind: AccessKind::Write,
                        visibility,
                    }],
                );
                let expected = if mode == FoldMode::Full && visibility == Visibility::SyncOnly {
                    Vec::new()
                } else {
                    vec![(24, 3, 0, 1)]
                };
                assert_eq!(out.writes, expected);
                assert!(out.reads.is_empty());
                assert!(!out.all);
            }
        }
    }

    #[test]
    fn resource_namespaces_are_distinct() {
        let spaces = [
            (ResourceSpace::Shared, 0),
            (ResourceSpace::Global, 1),
            (ResourceSpace::Tmem, 2),
            (ResourceSpace::TmemAllocator, 3),
            (ResourceSpace::CtaExit, 4),
            (ResourceSpace::BulkGroup, 5),
            (ResourceSpace::MbarPhase, 6),
            (ResourceSpace::MbarArrivalPhase, 7),
            (ResourceSpace::WarpPc, 8),
            (ResourceSpace::GridGate, 9),
            (ResourceSpace::StrongShared, 10),
            (ResourceSpace::StrongGlobal, 11),
            (ResourceSpace::MbarPhaseKey, 12),
            (ResourceSpace::ClusterBarrier, 13),
            (ResourceSpace::NamedBarrier, 14),
            (ResourceSpace::MbarInitialization, 15),
            (ResourceSpace::MbarPendingCount, 16),
            (ResourceSpace::UniformRegisters, 17),
            (ResourceSpace::MbarMembership, 18),
            (ResourceSpace::ConvergenceBarrier, 19),
            (ResourceSpace::L1Line, 20),
            (ResourceSpace::L1Sm, 21),
            (ResourceSpace::TensorCoreACollector, 24),
            (ResourceSpace::MbarOverArrival, 25),
            (ResourceSpace::NicQueue, 26),
            (ResourceSpace::RemotePipe, 27),
            (ResourceSpace::CtaAll, 98),
        ];
        for &(left, expected) in &spaces {
            assert_eq!(resource_tag(left), expected);
            for &(right, _) in &spaces {
                assert_eq!(resource_tag(left) == resource_tag(right), left == right);
            }
        }
    }
}
