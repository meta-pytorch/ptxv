/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::effects::FoldMode;
use crate::effects::Summary;
use crate::footprint::Range;
use crate::machine_recorder::BeginContext;
use crate::machine_recorder::FinishContext;
use crate::machine_recorder::FinishFootprint;
use crate::machine_recorder::RecordContext;

/// What one transition touched, for the DPOR dependency relation. Spaces: 0 shared memory
/// (id = CTA; mbarrier objects are their 8 bytes: poll = read, arrive/init/complete = write),
/// 1 global memory (id = allocation), 2 TMEM words (id = CTA, index = tmem_idx), 3 TMEM
/// allocator (id = CTA), 4 CTA exit state (id = CTA), 5 bulk-group state (id = issuer),
/// 6 mbarrier phase state (polls read, phase completions and init write), 7 mbarrier
/// arrival-phase keys (see `mbar_publish_transition`), 8 merge-sensitive pcs (id = warp), 9 grid
/// launch gates, 10/11 strong shared/global accesses, 12 mbarrier phase keys (`rec_phase`),
/// 13 barrier.cluster state, 14 named-barrier generations (id = CTA, index = barrier id << 32
/// | generation: arrivals read theirs, `bar.arrive` writes all; E-BAR-OVER writes, see
/// `Op::BarSync` and `bar_try_release` in exec.rs), 15 mbarrier (re)initialization: init and
/// inval write it (id = CTA, index = offset; sync-only, for the explorer's phase-aware
/// footprints, `sync_ample`), 16 reserved legacy mbarrier pending-count dependencies
/// (`SPACE_MBAR_COUNT`; superseded by audited phase membership in space 18);
/// 17 warp-shared registers and predicates: every split reads its warp; uniform value or
/// definedness writes conflict with peer splits of the same warp; unconditional BSYNC
/// uses its separate mask footprint when no additional tracking is active.
/// 18 audited mbarrier phase membership: a fully audited plain arrival batch reads its phase
/// when its total contribution is at most half the initial pending count; other mutations write;
/// 19 convergence barrier masks (id = warp, index = barrier register);
/// 20 / 21 L1 lines and SMs (l1_cores.rs `SPACE_LINE` / `SPACE_SM`),
/// 24 (`SPACE_COLL_A`) the tensor core's A collector buffer (id = CTA; every MMA writes it in a
/// kernel that uses `.collector::a::*`, S-TC-COLLECTOR), 25 mbarrier over-arrivals
/// (`SPACE_MBAR_OVER`, `rec_late`: updates read the key of the phase they land in; an arrival
/// of more than the pending arrivals at once writes it, and an update that lands after a
/// completion its actor does not know of writes the key of the completed phase), 26 NIC
/// queues (`SPACE_NIC_QUEUE`, id = queue). Named spaces must differ from each other and from
/// the literal ones (test `footprint_spaces_are_disjoint`).
#[derive(Clone, Debug, Default)]
pub struct Footprint {
    /// Record synchronization only (the stateful explorer's reduction): drop plain memory and
    /// TMEM (spaces 0-2, whose conflicts are races, E-DRF), the warp-internal space 8, and
    /// space 6, which space 12 refines by phase.
    pub sync_only: bool,
    pub r: Vec<(u8, u32, u64, u64)>,
    pub w: Vec<(u8, u32, u64, u64)>,
    /// Conflicts with everything (trap).
    pub all: bool,
    /// Synchronization that commutes but orders: release / acquire keys (`sync_key`). A
    /// transition that acquires a key happens after every earlier transition that released
    /// it (DPOR happens-before), without making the releasers mutually dependent.
    pub rel: Vec<u64>,
    pub acq: Vec<u64>,
}

impl Footprint {
    pub fn add(&mut self, space: u8, id: u32, lo: u64, hi: u64, write: bool) {
        if self.sync_only && matches!(space, 0 | 1 | 2 | 6 | 8) {
            return;
        }
        let v = if write { &mut self.w } else { &mut self.r };
        append_range(v, (space, id, lo, hi));
    }
    /// Sort and merge ranges.
    pub fn normalize(&mut self) {
        for v in [&mut self.rel, &mut self.acq] {
            v.sort_unstable();
            v.dedup();
        }
        for v in [&mut self.r, &mut self.w] {
            v.sort();
            let mut out: Vec<(u8, u32, u64, u64)> = Vec::with_capacity(v.len());
            for &x in v.iter() {
                if let Some(l) = out.last_mut() {
                    if l.0 == x.0 && l.1 == x.1 && x.2 <= l.3 {
                        l.3 = l.3.max(x.3);
                        continue;
                    }
                }
                out.push(x);
            }
            *v = out;
        }
    }
    pub fn union(&mut self, o: &Footprint) {
        self.r.extend_from_slice(&o.r);
        self.w.extend_from_slice(&o.w);
        self.rel.extend_from_slice(&o.rel);
        self.acq.extend_from_slice(&o.acq);
        self.all |= o.all;
        self.normalize();
    }
    /// Dependent: some write of one overlaps a read or write of the other.
    pub fn conflicts(&self, o: &Footprint) -> bool {
        crate::footprint::conflicts(self.all, &self.r, &self.w, o.all, &o.r, &o.w)
    }
    pub fn is_empty(&self) -> bool {
        !self.all && self.r.is_empty() && self.w.is_empty()
    }
    /// Does `self` acquire something `o` released?
    pub fn acquires_from(&self, o: &Footprint) -> bool {
        !self.acq.is_empty() && !o.rel.is_empty() && self.acq.iter().any(|k| o.rel.contains(k))
    }
}

/// Append one range, coalescing only with the last overlapping or adjacent range.
/// The input is left in its existing order; normalization is a separate boundary.
pub fn append_range(ranges: &mut Vec<Range>, range: Range) {
    let len = ranges.len();
    if len != 0 {
        let index = len - 1;
        let last = ranges[index];
        if last.0 == range.0 && last.1 == range.1 && range.2 <= last.3 && range.3 >= last.2 {
            ranges[index] = (last.0, last.1, last.2.min(range.2), last.3.max(range.3));
            return;
        }
    }
    ranges.push(range);
}

/// Merge a complete action summary through the same native streaming range policy.
pub fn merge_summary(footprint: &mut Footprint, summary: &Summary) {
    let mut index = 0;
    while index < summary.reads.len() {
        let range = summary.reads[index];
        footprint.add(range.0, range.1, range.2, range.3, false);
        index += 1;
    }
    index = 0;
    while index < summary.writes.len() {
        let range = summary.writes[index];
        footprint.add(range.0, range.1, range.2, range.3, true);
        index += 1;
    }
    index = 0;
    while index < summary.releases.len() {
        footprint.rel.push(summary.releases[index]);
        index += 1;
    }
    index = 0;
    while index < summary.acquires.len() {
        footprint.acq.push(summary.acquires[index]);
        index += 1;
    }
    footprint.all |= summary.all;
}

pub fn begin_context(footprint: Option<&Footprint>) -> BeginContext {
    BeginContext {
        footprint_empty: footprint.map(|fp| {
            !fp.all && fp.r.len() == 0 && fp.w.len() == 0 && fp.rel.len() == 0 && fp.acq.len() == 0
        }),
    }
}

pub fn record_context(footprint: Option<&Footprint>, slice: bool) -> RecordContext {
    RecordContext {
        mode: footprint.map(|fp| mode(fp.sync_only)),
        slice,
    }
}

pub fn mode(sync_only: bool) -> FoldMode {
    if sync_only {
        FoldMode::SyncOnly
    } else {
        FoldMode::Full
    }
}

fn has_membership(ranges: &[Range]) -> bool {
    let mut index = 0;
    while index < ranges.len() {
        if ranges[index].0 == 18 {
            return true;
        }
        index += 1;
    }
    false
}

/// Snapshot the phase-membership presence once, when a complete action is finalized.
pub fn finish_context(footprint: Option<&Footprint>) -> FinishContext {
    FinishContext {
        footprint: footprint.map(|fp| FinishFootprint {
            mode: mode(fp.sync_only),
            has_membership: has_membership(&fp.r) || has_membership(&fp.w),
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn completed_summary_preserves_keys_all_and_mode_filtering() {
        let summary = Summary {
            reads: vec![(1, 0, 3, 4), (18, 1, 4, 5), (18, 1, 5, 6)],
            writes: vec![(24, 2, 0, 1)],
            releases: vec![4, 4],
            acquires: vec![7],
            all: true,
        };
        let mut fp = Footprint {
            sync_only: true,
            rel: vec![1],
            acq: vec![2],
            ..Footprint::default()
        };
        merge_summary(&mut fp, &summary);
        assert_eq!(fp.r, [(18, 1, 4, 6)]);
        assert_eq!(fp.w, [(24, 2, 0, 1)]);
        assert_eq!(fp.rel, [1, 4, 4]);
        assert_eq!(fp.acq, [2, 7]);
        assert!(fp.all && fp.sync_only);
        assert_eq!(begin_context(Some(&fp)).footprint_empty, Some(false));
        assert!(finish_context(Some(&fp)).footprint.unwrap().has_membership);
        let key_only = Footprint {
            rel: vec![9],
            ..Footprint::default()
        };
        assert!(key_only.is_empty());
        assert_eq!(begin_context(Some(&key_only)).footprint_empty, Some(false));
    }
}
