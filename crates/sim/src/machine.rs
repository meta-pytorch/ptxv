/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Machine state for one launch: threads, warps, memories, synchronization objects, pending
//! async operations, and the happens-before (vector clock) shadow used for race detection.
//!
//! Everything that determines future behavior is hashed into the state fingerprint,
//! *including* the race shadow, so two paths are merged only if they are indistinguishable
//! for every future check (sound state caching).

use std::collections::BTreeMap;
use std::collections::BTreeSet;
use std::hash::Hash;
use std::hash::Hasher;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;

use ptxv_ir::*;
use serde::Serialize;

use crate::num::NumState;

pub const PARAM_BASE: u64 = 0x7e00_0000_0000;
pub const GLOBAL_BASE: u64 = 0x7000_0000_0000;

/// Strong reads made acquiring by a later acquire of the same thread (`Machine::fence_acq`).
#[derive(Default, Debug)]
pub struct FenceAcqState {
    /// (thread, writer) -> pcs of the thread's relaxed reads that observed a release of that
    /// writer (an entry of `Thread::acq_pending`).
    pub origins: std::collections::HashMap<(u32, u32), BTreeSet<u32>>,
    /// (thread, pc) of every such read whose pending release a later acquire (fence, exit
    /// fence, warp poll) joined.
    pub acquired: BTreeSet<(u32, u32)>,
}
/// Shared by every copy of a machine (`Machine::fence_acq`).
pub type FenceAcq = Arc<std::sync::Mutex<FenceAcqState>>;
/// Racing reads per race signature (`Machine::race_log`).
pub type RaceLog = Arc<std::sync::Mutex<BTreeMap<String, Option<BTreeSet<(u32, u32)>>>>>;
/// The generic shared window (4 GB): its low word is a shared::cluster address.
/// ASSUME(S-SWIN): GB300 puts it elsewhere (high word 0xfffa in fixtures/s_routing_forms);
/// only its being one 4 GB-aligned window matters.
pub const SHARED_GENERIC_BASE: u64 = 0x7c00_0000_0000;

/// Sparse vector clock. Representation: a shared immutable base (sorted (actor, clock)
/// pairs, with its hash precomputed) plus at most one "own" entry that overrides it. Threads
/// tick only their own entry, so after a barrier every released thread shares one base and
/// ticking is O(1) with no copy. Missing actors are 0.
///
/// The hash is the XOR of a per-entry mix over the *logical* clock (max of base and own), so
/// equal clocks hash equally whatever their representation (canonical states merge).
#[derive(Debug, Default)]
pub struct VC {
    base: Arc<VcBase>,
    /// (actor, clock) overriding the base for that actor; clock 0 = none.
    own: (u32, u32),
}

thread_local! {
    /// Set while `Machine::unshared` deep-copies a state.
    static UNSHARING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
    /// The VC bases copied so far by `Machine::unshared`, by address (sharing is kept).
    static VC_COPIES: std::cell::RefCell<std::collections::HashMap<usize, Arc<VcBase>>> =
        Default::default();
    /// `set_scratch_copies_shadow`.
    static PROV_CLONE_FULL: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

/// Differential tests: copies of a machine copy the provenance shadow too (as before copies
/// got a placeholder, coverage.rs `ProvState::clone`), on this thread.
pub fn set_scratch_copies_shadow(on: bool) {
    PROV_CLONE_FULL.set(on);
}

/// `set_scratch_copies_shadow` is on.
pub(crate) fn prov_clone_full() -> bool {
    PROV_CLONE_FULL.get()
}

impl Clone for VC {
    fn clone(&self) -> Self {
        let base = if UNSHARING.get() {
            VC_COPIES.with_borrow_mut(|c| {
                c.entry(Arc::as_ptr(&self.base) as usize)
                    .or_insert_with(|| {
                        Arc::new(VcBase {
                            e: self.base.e.clone(),
                            h: self.base.h,
                        })
                    })
                    .clone()
            })
        } else {
            self.base.clone()
        };
        VC {
            base,
            own: self.own,
        }
    }
}

#[derive(Debug, Default)]
struct VcBase {
    e: Vec<(u32, u32)>,
    h: u64,
}

#[inline]
fn vc_mix(a: u32, c: u32) -> u64 {
    if c == 0 {
        return 0;
    }
    let mut f = FastHasher {
        s: 0x0bad_c0de_dead_beef,
    };
    f.mix(((a as u64) << 32) | c as u64);
    f.finish()
}

impl VcBase {
    fn from(e: Vec<(u32, u32)>) -> Arc<VcBase> {
        let h = e.iter().fold(0, |h, &(a, c)| h ^ vc_mix(a, c));
        Arc::new(VcBase { e, h })
    }
    #[inline]
    fn get(&self, i: u32) -> u32 {
        match self.e.binary_search_by_key(&i, |e| e.0) {
            Ok(k) => self.e[k].1,
            Err(_) => 0,
        }
    }
}

impl VC {
    pub fn new() -> VC {
        VC::default()
    }
    /// Entries (actor, clock) in actor order, own entry folded in.
    fn entries(&self) -> Vec<(u32, u32)> {
        let mut e = self.base.e.clone();
        if self.own.1 != 0 {
            match e.binary_search_by_key(&self.own.0, |x| x.0) {
                Ok(k) => e[k].1 = e[k].1.max(self.own.1),
                Err(k) => e.insert(k, self.own),
            }
        }
        e
    }
    fn logical_hash(&self) -> u64 {
        if self.own.1 == 0 {
            return self.base.h;
        }
        let b = self.base.get(self.own.0);
        self.base.h ^ vc_mix(self.own.0, b) ^ vc_mix(self.own.0, b.max(self.own.1))
    }
}

impl PartialEq for VC {
    fn eq(&self, o: &VC) -> bool {
        self.logical_hash() == o.logical_hash() && self.entries() == o.entries()
    }
}
impl Eq for VC {}

impl Hash for VC {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.logical_hash().hash(h);
    }
}

/// Launch groups (groups.rs): per global byte (allocation, offset), the accesses of earlier
/// groups (epoch, is a write) that a later access must happen after (every later access for
/// a write, a later write for a read).
pub type XgMap = std::collections::HashMap<(u32, u64), Vec<(Epoch, bool)>>;

/// Pointwise minimum (a lower bound of both clocks).
pub fn vc_meet(a: &VC, b: &VC) -> VC {
    if a == b {
        return a.clone();
    }
    let (ea, eb) = (a.entries(), b.entries());
    let mut e = Vec::new();
    let (mut i, mut j) = (0, 0);
    while i < ea.len() && j < eb.len() {
        match ea[i].0.cmp(&eb[j].0) {
            std::cmp::Ordering::Less => i += 1,
            std::cmp::Ordering::Greater => j += 1,
            std::cmp::Ordering::Equal => {
                let c = ea[i].1.min(eb[j].1);
                if c > 0 {
                    e.push((ea[i].0, c));
                }
                i += 1;
                j += 1;
            }
        }
    }
    VC {
        base: VcBase::from(e),
        own: (0, 0),
    }
}

/// The join of `vs`. When they all share one base (a converged warp after its previous
/// lockstep join), this is that base plus their own entries: one allocation instead of one
/// merge per clock.
pub fn vc_join_all<'a>(vs: impl IntoIterator<Item = &'a VC>) -> VC {
    let vs: Vec<&VC> = vs.into_iter().collect();
    let Some(first) = vs.first() else {
        return VC::new();
    };
    if vs.iter().all(|v| Arc::ptr_eq(&v.base, &first.base)) {
        let mut e = first.base.e.clone();
        for v in &vs {
            if v.own.1 != 0 {
                match e.binary_search_by_key(&v.own.0, |x| x.0) {
                    Ok(k) => e[k].1 = e[k].1.max(v.own.1),
                    Err(k) => e.insert(k, v.own),
                }
            }
        }
        return VC {
            base: VcBase::from(e),
            own: (0, 0),
        };
    }
    let mut j = VC::new();
    for v in vs {
        vc_join(&mut j, v);
    }
    j
}

/// `a` joined with every clock of `vs`: each distinct base merged once, then the own
/// entries the result does not already cover inserted together. Equal to joining them one by
/// one (the join is associative and commutative); a barrier's arrivals after an earlier
/// barrier share one base, so this is one merge instead of one per arrival.
pub fn vc_join_many(a: &mut VC, vs: &[&VC]) {
    let mut bases: Vec<&Arc<VcBase>> = vs.iter().map(|v| &v.base).collect();
    bases.sort_unstable_by_key(|b| Arc::as_ptr(b) as usize);
    bases.dedup_by(|x, y| Arc::ptr_eq(x, y));
    for b in bases {
        vc_join(
            a,
            &VC {
                base: b.clone(),
                own: (0, 0),
            },
        );
    }
    let mut own: Vec<(u32, u32)> = vs
        .iter()
        .map(|v| v.own)
        .filter(|&(i, c)| c != 0 && vc_get(a, i) < c)
        .collect();
    if own.is_empty() {
        return;
    }
    own.sort_unstable();
    let mut e = a.entries();
    let mut out = Vec::with_capacity(e.len() + own.len());
    let mut k = 0;
    for (i, c) in own {
        while k < e.len() && e[k].0 < i {
            out.push(e[k]);
            k += 1;
        }
        match out.last_mut() {
            Some(l) if l.0 == i => l.1 = l.1.max(c),
            _ if k < e.len() && e[k].0 == i => {
                out.push((i, e[k].1.max(c)));
                k += 1;
            }
            _ => out.push((i, c)),
        }
    }
    out.extend(e.drain(k..));
    a.base = VcBase::from(out);
    a.own = (0, 0);
}

pub fn vc_join(a: &mut VC, b: &VC) {
    if b.base.e.is_empty() && b.own.1 == 0 {
        return;
    }
    // b adds nothing: same base and b's own entry already covered.
    if Arc::ptr_eq(&a.base, &b.base) && (b.own.1 == 0 || vc_get(a, b.own.0) >= b.own.1) {
        return;
    }
    // a is empty apart from its own entry: share b's base.
    if a.base.e.is_empty() && b.own.1 == 0 {
        a.base = b.base.clone();
        return;
    }
    let y: std::borrow::Cow<[(u32, u32)]> = if b.own.1 == 0 {
        std::borrow::Cow::Borrowed(&b.base.e)
    } else {
        std::borrow::Cow::Owned(b.entries())
    };
    // When one side covers the other, the merge below would rebuild that side's base entry for
    // entry: keep a's (a covers b), or share b's (b covers a's base; a's own entry stays). No
    // allocation, no rehash, sharing kept.
    match cover(&a.base.e, &y) {
        Cover2::Left => return,
        Cover2::Right if b.own.1 == 0 => {
            a.base = b.base.clone();
            return;
        }
        _ => {}
    }
    let x = &a.base.e;
    let mut out = Vec::with_capacity(x.len().max(y.len()));
    let (mut i, mut j) = (0, 0);
    while i < x.len() || j < y.len() {
        match (x.get(i), y.get(j)) {
            (Some(&(p, cp)), Some(&(q, cq))) if p == q => {
                out.push((p, cp.max(cq)));
                i += 1;
                j += 1;
            }
            (Some(&(p, cp)), Some(&(q, _))) if p < q => {
                out.push((p, cp));
                i += 1;
            }
            (Some(_), Some(&e)) => {
                out.push(e);
                j += 1;
            }
            (Some(&e), None) => {
                out.push(e);
                i += 1;
            }
            (None, Some(&e)) => {
                out.push(e);
                j += 1;
            }
            (None, None) => unreachable!(),
        }
    }
    a.base = VcBase::from(out);
}

/// Which of two entry lists (sorted by actor) covers the other: `Left` when every entry of
/// `y` is in `x` with at least its clock (merging them gives `x`), `Right` when every entry of
/// `x` is in `y` likewise (merging gives `y`), else `Neither`. Only exact answers: an actor
/// one side lacks counts against covering whatever its clock. Binary searches when `y` is much
/// shorter than `x` (then only `Left` is possible), else one merge scan that stops as soon as
/// neither can cover.
enum Cover2 {
    Left,
    Right,
    Neither,
}

fn cover(x: &[(u32, u32)], y: &[(u32, u32)]) -> Cover2 {
    if y.len() * 16 < x.len() {
        let left = y
            .iter()
            .all(|&(q, c)| matches!(x.binary_search_by_key(&q, |e| e.0), Ok(k) if x[k].1 >= c));
        return if left { Cover2::Left } else { Cover2::Neither };
    }
    let (mut i, mut j) = (0, 0);
    let (mut x_more, mut y_more) = (false, false);
    while i < x.len() && j < y.len() && !(x_more && y_more) {
        match x[i].0.cmp(&y[j].0) {
            std::cmp::Ordering::Less => {
                x_more = true;
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                y_more = true;
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                x_more |= x[i].1 > y[j].1;
                y_more |= y[j].1 > x[i].1;
                i += 1;
                j += 1;
            }
        }
    }
    x_more |= i < x.len();
    y_more |= j < y.len();
    match (x_more, y_more) {
        (_, false) => Cover2::Left,
        (false, true) => Cover2::Right,
        (true, true) => Cover2::Neither,
    }
}

/// `a` joined with each of `xs` in turn (`vc_join`). The last few results are kept per thread,
/// keyed by the representations of the inputs (the memo holds them, so a key cannot be
/// reused by other clocks): the shared-memory and tensor-memory clocks of an access
/// (`Machine::local_vc`) are asked for again and again with the same inputs.
pub fn vc_join_memo(a: &VC, xs: &[&VC]) -> VC {
    type Entry = (VC, Vec<VC>, VC);
    thread_local! {
        static MEMO: std::cell::RefCell<Vec<Entry>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    let hit = MEMO.with_borrow(|m| {
        m.iter()
            .find(|(k, ks, _)| {
                vc_unified(k, a)
                    && ks.len() == xs.len()
                    && ks.iter().zip(xs).all(|(k, x)| vc_unified(k, x))
            })
            .map(|e| e.2.clone())
    });
    if let Some(v) = hit {
        return v;
    }
    let mut v = a.clone();
    for x in xs {
        vc_join(&mut v, x);
    }
    MEMO.with_borrow_mut(|m| {
        m.insert(
            0,
            (
                a.clone(),
                xs.iter().map(|x| (*x).clone()).collect(),
                v.clone(),
            ),
        );
        m.truncate(8);
    });
    v
}

/// Do `a` and `b` share one representation (so they are equal)?
pub fn vc_unified(a: &VC, b: &VC) -> bool {
    Arc::ptr_eq(&a.base, &b.base) && a.own == b.own
}

/// Both clocks become the join of the two (`vc_join` each way), sharing one representation, so
/// that the next unify of the pair is a pointer comparison.
pub fn vc_unify(a: &mut VC, b: &mut VC) {
    if vc_unified(a, b) {
        return;
    }
    vc_join(a, b);
    *b = a.clone();
}

/// Entries (actor, clock) of `v` in actor order.
pub fn vc_entries(v: &VC) -> Vec<(u32, u32)> {
    v.entries()
}

/// Does `a` know everything `b` knows? Cheap when `b` has few entries.
pub fn vc_covers(a: &VC, b: &VC) -> bool {
    b.entries().iter().all(|&(x, c)| vc_get(a, x) >= c)
}

/// Entries (actor, clock) of `v` with actor >= `from`, in actor order.
pub fn vc_tail(v: &VC, from: u32) -> Vec<(u32, u32)> {
    let k = v.base.e.partition_point(|x| x.0 < from);
    let mut e: Vec<(u32, u32)> = v.base.e[k..].to_vec();
    if v.own.1 != 0 && v.own.0 >= from {
        match e.binary_search_by_key(&v.own.0, |x| x.0) {
            Ok(i) => e[i].1 = e[i].1.max(v.own.1),
            Err(i) => e.insert(i, v.own),
        }
    }
    e
}

/// The clock with entries `e` (sorted by actor, distinct actors, no zero clocks).
pub fn vc_from_sorted(e: Vec<(u32, u32)>) -> VC {
    VC {
        base: VcBase::from(e),
        own: (0, 0),
    }
}

/// E-READ-DONE: the clock entry of actor `a`'s completed reads (`Machine::read_release`).
pub const RR_BIT: u32 = 1 << 31;

/// Actors every thread has, with ids `k n + t` (`n` threads, `k` below this): the thread
/// itself, its tcgen05.ld/st actor (`Machine::tc_actor_id`), its tcgen05 pipe
/// (`Machine::pipe_actor_id`), its second tcgen05 pipe, for the `.cta_group` other than its
/// first MMA-pipe operation's (`Machine::pipe2_actor_id`; MISS 104), and its SASS `STTM`
/// writes (`Machine::sttm_actor_id`; MISS 107). Async actor ids start after them
/// (`async_base`).
pub const FIXED_ACTORS_PER_THREAD: u32 = 5;

/// The first async actor id with `n` threads (`Machine::async_base`).
pub fn async_base(n: u32) -> u32 {
    FIXED_ACTORS_PER_THREAD * n
}

/// log2 of a thread's async actor ids with `n` threads (`Machine::async_ids`): the largest
/// power of two that keeps every id, `async_base(n) + t * ids + serial`, below `RR_BIT`.
/// MISS 68: a fixed 4096 let a thread's 4097th async operation take the next thread's first id.
pub fn async_shift(n: u32) -> u32 {
    let n = n.max(1) as u64;
    let room = (RR_BIT as u64).saturating_sub(FIXED_ACTORS_PER_THREAD as u64 * n) / n;
    room.max(1).ilog2()
}

/// The (issuing thread, serial) of async actor `a` (at least `async_base(n)`, not a NIC's own)
/// with `n` threads and async shift `shift` (`async_shift`): the one decoder of async actor
/// ids, behind `Machine::async_split` and `actor_thread_of`.
pub fn async_split_of(a: u32, n: u32, shift: u32) -> (usize, u32) {
    let r = a - async_base(n);
    ((r >> shift) as usize, r & ((1 << shift) - 1))
}

/// The thread behind actor `a` with `n` threads and async shift `shift` (`async_shift`): a
/// fixed actor's thread, or an async actor's issuer (`async_split_of`; `Machine::actor_thread`,
/// for code that has the threads but not the `Machine`).
pub fn actor_thread_of(a: u32, n: u32, shift: u32) -> usize {
    if a < async_base(n) {
        (a % n) as usize
    } else {
        async_split_of(a, n, shift).0
    }
}

/// E-READ-DONE: the entries of `rr` that `v` does not know yet. (Not pruned against
/// happens-before: `Machine::read_release` forwards read entries, not thread entries, so a
/// read entry dropped because happens-before covers it would be lost at the next relaxed
/// store; `r4c_ring_reuse_is_load_buffering_under_the_strict_model`.)
pub fn rr_new(v: &VC, rr: &VC) -> Option<VC> {
    let e: Vec<(u32, u32)> = rr
        .entries()
        .into_iter()
        .filter(|&(a, c)| vc_get(v, a) < c)
        .collect();
    (!e.is_empty()).then(|| vc_from_sorted(e))
}

/// `vc_get(vc, ·)` with the last answers kept (direct-mapped by actor): a shadow check asks
/// the same clock once per byte of an access.
pub struct VcMemo<'a> {
    vc: &'a VC,
    slots: [std::cell::Cell<(u64, u32)>; 32],
}

impl<'a> VcMemo<'a> {
    pub fn new(vc: &'a VC) -> Self {
        VcMemo {
            vc,
            slots: Default::default(),
        }
    }
    #[inline]
    pub fn get(&self, i: u32) -> u32 {
        let s = &self.slots[i as usize % 32];
        let (k, c) = s.get();
        if k == i as u64 + 1 {
            return c;
        }
        let c = vc_get(self.vc, i);
        s.set((i as u64 + 1, c));
        c
    }
    /// `e.hb(vc)`.
    #[inline]
    pub fn hb(&self, e: &Epoch) -> bool {
        e.clock <= self.get(e.actor)
    }
}

#[inline]
pub fn vc_get(v: &VC, i: u32) -> u32 {
    let b = v.base.get(i);
    if v.own.1 != 0 && v.own.0 == i {
        b.max(v.own.1)
    } else {
        b
    }
}

/// The clock that knows only `actor` up to `clock`.
pub fn vc_single(actor: u32, clock: u32) -> VC {
    VC {
        base: VcBase::from(vec![(actor, clock)]),
        own: (0, 0),
    }
}

/// `v` with actor `i`'s entry raised to at least `c` (an async actor's epoch).
pub fn vc_raise(v: &mut VC, i: u32, c: u32) {
    if v.own.1 != 0 && v.own.0 != i {
        v.base = VcBase::from(v.entries());
        v.own = (0, 0);
    }
    let c = vc_get(v, i).max(c);
    v.own = (i, c);
}

/// `vc_raise` of each clock of `vs` by actor `i` to `c`, sharing the work: a clock whose own
/// entry is another actor's keeps it over a base with `i` raised, built once per distinct
/// base (copies one thread issued between two of its synchronizations share one).
pub fn vc_raise_many<'a>(vs: impl IntoIterator<Item = &'a VC>, i: u32, c: u32) -> Vec<VC> {
    let mut bases: Vec<(*const VcBase, Arc<VcBase>)> = Vec::new();
    vs.into_iter()
        .map(|v| {
            if v.own.1 == 0 || v.own.0 == i {
                let mut r = v.clone();
                vc_raise(&mut r, i, c);
                return r;
            }
            let key = Arc::as_ptr(&v.base);
            let base = match bases.iter().find(|b| b.0 == key) {
                Some(b) => b.1.clone(),
                None => {
                    let mut e = v.base.e.clone();
                    match e.binary_search_by_key(&i, |x| x.0) {
                        Ok(k) => e[k].1 = e[k].1.max(c),
                        Err(k) => e.insert(k, (i, c)),
                    }
                    let b = VcBase::from(e);
                    bases.push((key, b.clone()));
                    b
                }
            };
            VC { base, own: v.own }
        })
        .collect()
}

pub fn vc_tick(v: &mut VC, i: u32) {
    if v.own.1 != 0 && v.own.0 != i {
        // Fold the other actor's own entry into a new base first.
        v.base = VcBase::from(v.entries());
        v.own = (0, 0);
    }
    let c = vc_get(v, i) + 1;
    v.own = (i, c);
}

/// Insert or replace `e` in a list kept sorted by actor (canonical state: commuting
/// accesses must produce identical shadows).
pub fn epoch_insert(list: &mut Vec<Epoch>, e: Epoch) {
    match list.binary_search_by_key(&e.actor, |x| x.actor) {
        Ok(k) => list[k] = e,
        Err(k) => list.insert(k, e),
    }
}

/// One access: who, at which logical time, by which instruction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub struct Epoch {
    pub actor: u32,
    pub clock: u32,
    pub pc: u32,
    /// 0: a plain access; otherwise a strong access at scope `strong - 1` (`strong_code`).
    /// Strong accesses that are morally strong do not race with each other (S-SCOPE).
    pub strong: u8,
    /// On a write: a plain write of the value already there (S-SAME-VALUE): a read sees that
    /// value in either order, so reads are not checked against it. On a read (`Sh::r`,
    /// hardware model only, E-HW-CORW): a strong read that returned the bytes of the shadow's
    /// last write, with no write to the byte since.
    pub sv: bool,
    /// The mapping the access went through (S-VA-ALIAS): 0 the buffer's primary mapping,
    /// `MC_VIEW` a multicast (NVLS) mapping, k a further mapping (`Launch::views[k-1]`).
    /// Nonzero only for allocations reached through several mappings.
    pub view: u16,
}

/// `Epoch::strong` of a TMEM allocation / deallocation (`tmem_mark`): every word is written.
pub const TMEM_MARK: u8 = u8::MAX;

/// `Epoch::strong` of a strong access at `s`.
pub fn strong_code(s: Scope) -> u8 {
    s as u8 + 1
}

/// Morally strong with everything (mbarrier objects, legacy atomics).
pub const STRONG_SYS: u8 = Scope::Sys as u8 + 1;

/// ASSUME(S-FENCE-SC): one SC fence (`fence.sc` / `membar` at gpu / sys scope, SASS
/// `MEMBAR.SC.GPU` / `.SYS`) as `exec.rs` `sc_fence` recorded it.
#[derive(Clone, Debug, Hash)]
pub struct ScFence {
    pub actor: u32,
    /// The fence's own clock entry: an operation knows the fence (the fence happens before
    /// it) exactly when its clock's entry for `actor` is at least this.
    pub own: u32,
    pub device: u32,
    pub sys: bool,
    /// The clock at the fence (what a later fence it synchronizes with acquires).
    pub vc: VC,
}

/// ASSUME(S-FENCE-SC): one strong global read-modify-write as `exec.rs` `sc_note_write`
/// recorded it.
#[derive(Clone, Debug, Hash)]
pub struct ScWrite {
    pub actor: u32,
    /// As `ScFence::own`.
    pub own: u32,
    pub strong: u8,
    pub pc: u32,
    /// The clock at the write.
    pub vc: VC,
    /// The join of the clocks of the morally strong read-modify-writes to the same location
    /// before it in coherence order (the schedule's order of writes to the location).
    pub pre: VC,
}

/// ASSUME(S-FENCE-SC): the run's SC fences (gpu / sys scope) and strong global
/// read-modify-writes. Fences are keyed by (actor, own clock), so the log does not depend on
/// the order unrelated fences ran in (states still merge).
#[derive(Clone, Debug, Default, Hash)]
pub struct ScLog {
    pub fences: BTreeMap<(u32, u32), ScFence>,
    /// Per location (`Machine::syncs` key), its read-modify-writes in coherence order.
    pub writes: BTreeMap<(u8, u32, u64), Vec<ScWrite>>,
    pub nwrites: usize,
    /// The caps were reached: later fences and writes are not recorded (fewer edges).
    pub full: bool,
}

/// The last strong write of a location, as a synchronization source (S-SCOPE).
#[derive(Clone, Debug)]
pub struct SyncCell {
    pub writer: u32,
    /// Scope of the write itself (moral strength with a reader).
    pub scope: Scope,
    /// The write's instruction (moral strength with a reader: complete overlap, S-SCOPE);
    /// `u32::MAX` for a write that is not an instruction (the NIC's).
    pub pc: u32,
    /// What a reader that observes it acquires from the write's own release pattern: the
    /// writer's clock at the release (release write) or at its last release fence (relaxed
    /// write after a fence).
    pub rel: VC,
    /// Scope the release was made at (the fence's, for a fence-then-relaxed-write).
    pub rel_scope: Scope,
    /// The release patterns of earlier writes that this read-modify-write continues (sorted
    /// by writer and scope, merged): a reader synchronizes with each only when the pattern's
    /// first operation and the reader's acquire are morally strong (PTX ISA 9.5 section
    /// 8.9.4, S-SCOPE). Empty under the hardware ordering, which joins them into `rel`.
    pub cont: Vec<SyncRel>,
    /// The join of `cont`'s releases: what a read acquires from them when every one passes
    /// (one clock join in place of one per pattern).
    pub cont_join: VC,
    /// E-READ-DONE: the writer's completed reads (`Machine::read_release`), which a strong
    /// read observing this write orders before itself for write-after-read purposes.
    pub rr: VC,
    /// The writer's clock at the write (strict PTX level only, else 0): a lane of the same
    /// warp at the same instruction that does not know it read it in lane order only, which
    /// is no synchronization (`Machine::sync_observe_cell`, MISS 140).
    pub own: u32,
}

/// A release pattern a synchronization source continues (`SyncCell::cont`): the thread of its
/// first operation, what it releases, and the scope of the release.
#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct SyncRel {
    pub writer: u32,
    pub rel: VC,
    pub rel_scope: Scope,
}

/// Every field but `cont_join`, a function of `cont`.
impl Hash for SyncCell {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.writer.hash(h);
        self.scope.hash(h);
        self.pc.hash(h);
        self.rel.hash(h);
        self.rel_scope.hash(h);
        self.cont.hash(h);
        self.rr.hash(h);
        self.own.hash(h);
    }
}

impl SyncCell {
    /// Add a continued release, joined into an entry of the same writer and scope (and into
    /// `join`, the entries' join, `SyncCell::cont_join`).
    pub fn add_cont(
        cont: &mut Vec<SyncRel>,
        join: &mut VC,
        writer: u32,
        rel: &VC,
        rel_scope: Scope,
    ) {
        if *rel == VC::new() {
            return;
        }
        vc_join(join, rel);
        match cont.binary_search_by_key(&(writer, rel_scope), |c| (c.writer, c.rel_scope)) {
            Ok(i) => vc_join(&mut cont[i].rel, rel),
            Err(i) => cont.insert(
                i,
                SyncRel {
                    writer,
                    rel: rel.clone(),
                    rel_scope,
                },
            ),
        }
    }

    /// Launch groups: `cont` entrywise met with `o`'s, when both continue the same releases,
    /// and the met entries' join (`cont_join`).
    pub fn cont_meet(&self, o: &SyncCell) -> Option<(Vec<SyncRel>, VC)> {
        if self.cont.len() != o.cont.len()
            || self
                .cont
                .iter()
                .zip(&o.cont)
                .any(|(a, b)| (a.writer, a.rel_scope) != (b.writer, b.rel_scope))
        {
            return None;
        }
        let cont: Vec<SyncRel> = self
            .cont
            .iter()
            .zip(&o.cont)
            .map(|(a, b)| SyncRel {
                writer: a.writer,
                rel: vc_meet(&a.rel, &b.rel),
                rel_scope: a.rel_scope,
            })
            .collect();
        let join = vc_join_all(cont.iter().map(|c| &c.rel));
        Some((cont, join))
    }
}

impl Epoch {
    /// Does this access happen-before an actor whose clock is `v`?
    #[inline]
    pub fn hb(&self, v: &VC) -> bool {
        self.clock <= vc_get(v, self.actor)
    }
}

/// Per-byte shadow: last write and reads since it (FastTrack-style), plus the earlier
/// accesses that a strong write does not cover (`Machine::access_x`).
#[derive(Clone, Default, Debug, PartialEq, Eq)]
pub struct Sh {
    pub w: Option<Epoch>,
    /// Reads not covered by a later write: since the last write, and before it when that
    /// write is strong (a strong write hides neither a plain read nor an unordered strong one).
    pub r: Readers,
    /// Earlier writes the last write `w` does not cover, sorted by (actor, strong): a plain
    /// write before a strong one, or a strong write unordered with it. Empty unless strong
    /// writes followed other writes. A later access must be checked against them as against
    /// `w`: a plain write followed by its own thread's atomic still races with another
    /// thread's unordered atomic (FastTrack's single last-write epoch is precise only when
    /// every write covers the earlier ones).
    pub hw: Option<Box<[Epoch]>>,
    /// S-INIT: the last write stored undefined bits (an `STS` of an undefined register); a
    /// load reads it as never written.
    pub undef: bool,
    /// The writes may have stored different values: strong writes demoted at a launch-group
    /// boundary (`demote`). Until the next plain write of another value, the same-value
    /// invariant (every write of the shadow stored the current value) does not hold.
    pub mixed: bool,
    /// S-NC: the byte was read by a non-coherent load (`ld.global.nc`, `LDG .CONSTANT`): the
    /// grids that did (bit g, grids from 63 on share bit 63) and the lowest such load's pc.
    /// Order-free (a union and a minimum), so commuting reads leave the same shadow.
    pub nc: Option<(u64, u32)>,
}

/// The reads since the last write, one per actor, sorted by actor (canonical state: commuting
/// accesses must produce identical shadows). Up to two are stored inline: shadow pages are
/// cloned on every transition that touches them, and a heap list per read byte made
/// malloc/free a quarter of exploration time.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Readers {
    Inline(u8, [Epoch; 2]),
    Heap(Vec<Epoch>),
}

impl Default for Readers {
    fn default() -> Self {
        Readers::Inline(0, [Epoch::default(); 2])
    }
}

impl Sh {
    /// Every write of the shadow: the last one and the hidden earlier ones.
    #[inline]
    pub fn writes(&self) -> impl Iterator<Item = &Epoch> {
        self.w.iter().chain(self.hw.iter().flat_map(|h| h.iter()))
    }

    /// Every recorded access as plain (launch groups, `groups.rs`): a later access that is
    /// not ordered after it races, strong or not (TMEM marks are kept).
    pub fn demote(&mut self) {
        // S-NC: earlier groups' grids are stream-ordered before every later write.
        self.nc = None;
        let d = |e: &mut Epoch| {
            if e.strong != TMEM_MARK {
                e.strong = 0;
            }
        };
        if self
            .writes()
            .any(|e| e.strong != 0 && e.strong != TMEM_MARK)
        {
            self.mixed = true;
        }
        if let Some(w) = self.w.as_mut() {
            d(w);
        }
        if let Some(h) = self.hw.as_mut() {
            h.iter_mut().for_each(d);
            // ASSUME(S-SAME-VALUE): a plain last write with uncovered earlier writes means
            // those wrote the same value (the same-value rule in `access_x`); demoted strong
            // writes need not have, so the last write joins them.
            if let Some(w) = self.w.take() {
                let mut v = h.to_vec();
                v.push(w);
                v.sort_unstable_by_key(|e| (e.actor, e.strong, e.clock));
                *h = v.into_boxed_slice();
            }
        }
        match &mut self.r {
            Readers::Inline(n, a) => a[..*n as usize].iter_mut().for_each(d),
            Readers::Heap(v) => v.iter_mut().for_each(d),
        }
    }
}

impl Readers {
    #[inline]
    pub fn as_slice(&self) -> &[Epoch] {
        match self {
            Readers::Inline(n, a) => &a[..*n as usize],
            Readers::Heap(v) => v,
        }
    }
    #[inline]
    pub fn iter(&self) -> std::slice::Iter<'_, Epoch> {
        self.as_slice().iter()
    }
    #[inline]
    pub fn len(&self) -> usize {
        self.as_slice().len()
    }
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    #[inline]
    pub fn clear(&mut self) {
        *self = Readers::default();
    }
    /// A write to the byte: no kept read returned the bytes of the (new) last write
    /// (E-HW-CORW, `Epoch::sv` on reads).
    pub fn forget_rf(&mut self) {
        match self {
            Readers::Inline(n, a) => a[..*n as usize].iter_mut().for_each(|e| e.sv = false),
            Readers::Heap(v) => v.iter_mut().for_each(|e| e.sv = false),
        }
    }
    /// Keep the reads for which `f` holds (order kept).
    pub fn retain(&mut self, mut f: impl FnMut(&Epoch) -> bool) {
        if self.iter().all(&mut f) {
            return;
        }
        let v: Vec<Epoch> = self.iter().copied().filter(|e| f(e)).collect();
        *self = Readers::from_sorted(v);
    }
    fn from_sorted(v: Vec<Epoch>) -> Self {
        if v.len() <= 2 {
            let mut a = [Epoch::default(); 2];
            a[..v.len()].copy_from_slice(&v);
            Readers::Inline(v.len() as u8, a)
        } else {
            Readers::Heap(v)
        }
    }
    /// Add read `e` of `access_x`, sorted by (actor, strong, view): it replaces the same
    /// actor's reads it covers (`strong` at least `e.strong`: a later read of the same thread
    /// with a scope no wider is racy with whatever the earlier one was racy with; through the
    /// same mapping, S-VA-ALIAS: a read through another mapping is checked against a later
    /// write through this one) and keeps that actor's plainer ones (a strong read does not
    /// cover an earlier plain read).
    pub fn insert_read(&mut self, e: Epoch) {
        let s = self.as_slice();
        // Fast path: at most one read of `e.actor`, not plainer than `e`, through the same
        // mapping (the common case): exactly the per-actor `insert`.
        // The list is sorted by actor: `e.actor`'s reads are one run, found by bisection.
        let lo = s.partition_point(|x| x.actor < e.actor);
        let mut mine = s[lo..].iter().take_while(|x| x.actor == e.actor);
        let one = mine
            .next()
            .is_none_or(|x| x.strong >= e.strong && x.view == e.view)
            && mine.next().is_none();
        if one {
            self.insert(e);
            return;
        }
        let mut v: Vec<Epoch> = s
            .iter()
            .copied()
            .filter(|x| x.actor != e.actor || x.strong < e.strong || x.view != e.view)
            .collect();
        let k = v.partition_point(|x| (x.actor, x.strong, x.view) < (e.actor, e.strong, e.view));
        v.insert(k, e);
        *self = Readers::from_sorted(v);
    }
    /// Insert or replace `e` (by actor).
    pub fn insert(&mut self, e: Epoch) {
        match self {
            Readers::Inline(n, a) => {
                let k = *n as usize;
                match a[..k].binary_search_by_key(&e.actor, |x| x.actor) {
                    Ok(j) => a[j] = e,
                    Err(j) if k < 2 => {
                        a.copy_within(j..k, j + 1);
                        a[j] = e;
                        *n += 1;
                    }
                    Err(_) => {
                        let mut v = a.to_vec();
                        epoch_insert(&mut v, e);
                        *self = Readers::Heap(v);
                    }
                }
            }
            Readers::Heap(v) => epoch_insert(v, e),
        }
    }
}

impl<'a> IntoIterator for &'a Readers {
    type Item = &'a Epoch;
    type IntoIter = std::slice::Iter<'a, Epoch>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Epoch {
    /// The epoch in two words, for hashing.
    #[inline]
    fn words(&self) -> (u64, u64) {
        (
            (self.actor as u64) << 32 | self.clock as u64,
            (self.pc as u64) << 8
                | self.strong as u64
                | (self.view as u64) << 40
                | (self.sv as u64) << 62
                | 1 << 63,
        )
    }
}

// Hashed per page, so it is on the fingerprint's hot path: two words per epoch.
impl Hash for Sh {
    fn hash<H: Hasher>(&self, h: &mut H) {
        match &self.w {
            Some(e) => {
                let (a, b) = e.words();
                h.write_u64(a);
                h.write_u64(b);
            }
            None => h.write_u64(0),
        }
        h.write_u64(self.r.len() as u64);
        for e in &self.r {
            let (a, b) = e.words();
            h.write_u64(a);
            h.write_u64(b);
        }
        if self.undef {
            h.write_u64(7 << 60);
        }
        if self.mixed {
            h.write_u64(5 << 60);
        }
        if let Some((g, pc)) = self.nc {
            h.write_u64(g);
            h.write_u64(pc as u64 | 3 << 60);
        }
        if let Some(hw) = &self.hw {
            h.write_u64(hw.len() as u64 | 1 << 62);
            for e in hw.iter() {
                let (a, b) = e.words();
                h.write_u64(a);
                h.write_u64(b);
            }
        }
    }
}

/// Shared memory below this offset is reserved by the driver/ptxas (initialized at launch);
/// user data starts here on both front-ends.
pub const SMEM_RESERVED: usize = 0x400;

const PAGE: usize = 512;

/// Fast 64-bit hasher for cached sub-object hashes (folded 128-bit multiply per word, as in
/// wyhash/ahash). SipHash cost 60% of exploration time on rung 1b; the outer 128-bit
/// fingerprint still combines these with two independently seeded SipHashes.
#[derive(Default)]
pub struct FastHasher {
    pub(crate) s: u64,
}

impl FastHasher {
    #[inline]
    fn mix(&mut self, w: u64) {
        let m = ((self.s ^ w) as u128).wrapping_mul(0x9e37_79b9_7f4a_7c15_f39c_c060_5ced_c835);
        self.s = (m as u64) ^ ((m >> 64) as u64);
    }
}

impl Hasher for FastHasher {
    fn write(&mut self, bytes: &[u8]) {
        let mut c = bytes.chunks_exact(8);
        for w in &mut c {
            self.mix(u64::from_le_bytes(w.try_into().unwrap()));
        }
        let r = c.remainder();
        if !r.is_empty() {
            let mut b = [0u8; 8];
            b[..r.len()].copy_from_slice(r);
            self.mix(u64::from_le_bytes(b) ^ ((r.len() as u64) << 56));
        }
    }
    fn write_u8(&mut self, i: u8) {
        self.mix(i as u64 | 0x100);
    }
    fn write_u16(&mut self, i: u16) {
        self.mix(i as u64 | 0x2_0000);
    }
    fn write_u32(&mut self, i: u32) {
        self.mix(i as u64 | 0x4_0000_0000);
    }
    fn write_u64(&mut self, i: u64) {
        self.mix(i);
    }
    fn write_usize(&mut self, i: usize) {
        self.mix(i as u64);
    }
    fn finish(&self) -> u64 {
        // murmur3 fmix64 avalanche
        let mut h = self.s;
        h ^= h >> 33;
        h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
        h ^= h >> 33;
        h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
        h ^ (h >> 33)
    }
}

/// Shared, copy-on-write value with a cached hash (reset on mutation). Fingerprinting a
/// state then costs O(touched objects) instead of O(all memory).
#[derive(Debug)]
pub struct Cached<T> {
    v: T,
    h: std::sync::atomic::AtomicU64,
}

impl<T: Clone> Clone for Cached<T> {
    fn clone(&self) -> Self {
        Cached {
            v: self.v.clone(),
            h: std::sync::atomic::AtomicU64::new(self.h.load(std::sync::atomic::Ordering::Relaxed)),
        }
    }
}

#[derive(Debug)]
pub struct Cow<T>(Arc<Cached<T>>);

impl<T: Clone> Clone for Cow<T> {
    fn clone(&self) -> Self {
        if UNSHARING.get() {
            Cow(Arc::new(self.0.as_ref().clone()))
        } else {
            Cow(self.0.clone())
        }
    }
}

impl<T: Clone + Hash> Cow<T> {
    pub fn new(v: T) -> Self {
        Cow(Arc::new(Cached {
            v,
            h: std::sync::atomic::AtomicU64::new(0),
        }))
    }
    pub fn make_mut(&mut self) -> &mut T {
        // Copy a shared value without `Arc::make_mut`'s compare-and-swap, which needs the
        // count's cache line exclusively: contended when explorer threads share pages.
        if Arc::strong_count(&self.0) > 1 {
            self.0 = Arc::new(self.0.as_ref().clone());
        }
        let c = Arc::make_mut(&mut self.0);
        c.h.store(0, std::sync::atomic::Ordering::Relaxed);
        &mut c.v
    }
    fn cached_hash(&self) -> u64 {
        use std::sync::atomic::Ordering::Relaxed;
        let h = self.0.h.load(Relaxed);
        if h != 0 {
            return h;
        }
        let mut d = FastHasher {
            s: 0x243f_6a88_85a3_08d3,
        };
        self.0.v.hash(&mut d);
        let h = d.finish() | 1;
        self.0.h.store(h, Relaxed);
        h
    }
}

impl<T> std::ops::Deref for Cow<T> {
    type Target = T;
    fn deref(&self) -> &T {
        &self.0.v
    }
}

impl<T: Clone + Hash> Hash for Cow<T> {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.cached_hash().hash(h);
    }
}

/// Copy-on-write paged array: cloning a machine is cheap (one reference count per array, not
/// one per page), and a transition copies only the page table and the pages it touches. Both
/// levels cache their hash, so an untouched array costs O(1) to fingerprint.
#[derive(Debug)]
pub struct Pages<T: Clone> {
    pages: Cow<Vec<Cow<Vec<T>>>>,
    len: usize,
}

impl<T: Clone> Clone for Pages<T> {
    fn clone(&self) -> Self {
        // `Machine::unshared` copies the page table but keeps sharing the pages: copying all
        // of memory costs more than a page's count is touched (only when a page is written).
        let pages = if UNSHARING.get() {
            let t = &self.pages.0;
            Cow(Arc::new(Cached {
                v: t.v.iter().map(|p| Cow(p.0.clone())).collect(),
                h: std::sync::atomic::AtomicU64::new(
                    t.h.load(std::sync::atomic::Ordering::Relaxed),
                ),
            }))
        } else {
            self.pages.clone()
        };
        Pages {
            pages,
            len: self.len,
        }
    }
}

impl<T: Clone + Default + Hash> Pages<T> {
    pub fn new(len: usize) -> Self {
        let n = len.div_ceil(PAGE).max(1);
        Pages {
            pages: Cow::new((0..n).map(|_| Cow::new(vec![T::default(); PAGE])).collect()),
            len,
        }
    }
}

impl<T: Clone> Pages<T> {
    /// Number of pages.
    pub fn npages(&self) -> usize {
        self.pages.len()
    }
    /// Is page `pi` the same stored page in both (cheap: no comparison of contents)?
    pub fn same_page(&self, o: &Self, pi: usize) -> bool {
        Arc::ptr_eq(&self.pages[pi].0, &o.pages[pi].0)
    }
    /// Elements per page.
    pub const PAGE: usize = PAGE;
}

impl<T: Clone + Hash> Pages<T> {
    pub fn from_vec(v: Vec<T>, fill: T) -> Self {
        let len = v.len();
        let mut pages = Vec::new();
        for chunk in v.chunks(PAGE) {
            let mut p = chunk.to_vec();
            p.resize(PAGE, fill.clone());
            pages.push(Cow::new(p));
        }
        if pages.is_empty() {
            pages.push(Cow::new(vec![fill; PAGE]));
        }
        Pages {
            pages: Cow::new(pages),
            len,
        }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// All elements equal (`fill`), each page stored as one element until it is written
    /// unevenly (`for_run`).
    pub fn uniform(len: usize, fill: T) -> Self {
        let n = len.div_ceil(PAGE).max(1);
        Pages {
            pages: Cow::new((0..n).map(|_| Cow::new(vec![fill.clone()])).collect()),
            len,
        }
    }
    #[inline]
    pub fn get(&self, i: usize) -> &T {
        let p = &self.pages[i / PAGE];
        if p.len() == 1 { &p[0] } else { &p[i % PAGE] }
    }
    /// Page `pi`, writable and stored element by element.
    #[inline]
    fn dense_mut(&mut self, pi: usize) -> &mut Vec<T> {
        let p = self.pages.make_mut()[pi].make_mut();
        if p.len() == 1 {
            let v = p[0].clone();
            *p = vec![v; PAGE];
        }
        p
    }
    #[inline]
    pub fn get_mut(&mut self, i: usize) -> &mut T {
        &mut self.dense_mut(i / PAGE)[i % PAGE]
    }
    /// Elements `i..` up to the end of `i`'s page, writable (one copy-on-write check for the
    /// run instead of one per element).
    #[inline]
    pub fn run_mut(&mut self, i: usize) -> &mut [T] {
        &mut self.dense_mut(i / PAGE)[i % PAGE..]
    }
    /// Elements `i..` up to the end of `i`'s page (not for `uniform` arrays: use `get`).
    #[inline]
    pub fn run(&self, i: usize) -> &[T] {
        let p = &self.pages[i / PAGE];
        debug_assert!(p.len() == PAGE);
        &p[i % PAGE..]
    }
    /// Is `i`'s page stored as one element (all its elements equal)?
    #[inline]
    pub fn uniform_page(&self, i: usize) -> bool {
        self.pages[i / PAGE].len() == 1
    }
    /// Stores `i`'s page element by element, so that the next `for_run` over it calls its
    /// closure once per element.
    #[inline]
    pub fn split_page(&mut self, i: usize) {
        self.dense_mut(i / PAGE);
    }
}

impl<T: Clone + Hash + PartialEq> Pages<T> {
    /// `f(index, element)` for the elements `i..i + n` up to the end of `i`'s page; returns
    /// how many that is. A whole page whose elements are all equal is stored (and updated:
    /// `f` runs once, with the page's first index) as one element. Bulk copies and MMA
    /// operand reads give every byte of a tile the same shadow, and a dense shadow page cost
    /// 32 KB per state that touched it. `f` must depend on the index only through what it
    /// reports; a caller whose `f` depends on it otherwise splits the page first
    /// (`split_page`; `access_x`: a write that keeps the value of only some bytes).
    #[inline]
    pub fn for_run(&mut self, i: usize, n: usize, mut f: impl FnMut(usize, &mut T)) -> usize {
        let (pi, o) = (i / PAGE, i % PAGE);
        let k = (PAGE - o).min(n);
        let full = o == 0 && k == PAGE;
        let p = self.pages.make_mut()[pi].make_mut();
        if p.len() == 1 {
            if full {
                f(i, &mut p[0]);
                return k;
            }
            let v = p[0].clone();
            *p = vec![v; PAGE];
        }
        for (j, x) in p[o..o + k].iter_mut().enumerate() {
            f(i + j, x);
        }
        if full && p[1..].iter().all(|x| *x == p[0]) {
            p.truncate(1);
            p.shrink_to_fit();
        }
        k
    }
}

impl<T: Clone + Hash> Hash for Pages<T> {
    fn hash<H: Hasher>(&self, h: &mut H) {
        // `len` is fixed at construction and determines the page count.
        self.pages.hash(h);
    }
}

#[derive(Clone, Debug, Hash)]
pub struct Mem {
    pub bytes: Pages<u8>,
    pub sh: Pages<Sh>,
}

impl Mem {
    pub fn new(init: Vec<u8>) -> Mem {
        let n = init.len();
        Mem {
            bytes: Pages::from_vec(init, 0),
            sh: Pages::uniform(n, Sh::default()),
        }
    }

    /// Memory whose every byte holds undefined data until written (S-INIT scratch buffers).
    pub fn new_undef(init: Vec<u8>) -> Mem {
        let n = init.len();
        Mem {
            bytes: Pages::from_vec(init, 0),
            sh: Pages::uniform(
                n,
                Sh {
                    undef: true,
                    ..Sh::default()
                },
            ),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize)]
pub enum St {
    Run,
    Bar,
    /// Blocked in barrier.cluster.wait.
    ClusterWait,
    /// The CTA has not launched yet (CLC kernels, E-CLC).
    Unlaunched,
    /// Waiting in WARPSYNC / bar.warp.sync for the other lanes.
    WarpSync,
    Exited,
}

#[derive(Clone, Debug, Hash)]
pub struct Thread {
    pub cta: u16,
    /// Global warp index.
    pub warp: u32,
    pub lane: u32,
    pub tid: [u32; 3],
    pub pc: u32,
    pub st: St,
    pub regs: Vec<u64>,
    pub preds: Vec<u64>,
    /// Bitset over `preds`: predicate written since launch (S-INIT-PRED: a read of an
    /// unwritten predicate is reported, as for registers; SASS `PT` is always defined).
    pub pdef: Vec<u64>,
    pub vc: VC,
    /// View of this thread's tcgen05.ld/st actor (`Machine::tc_actor_id`): tcgen05 operations
    /// only see synchronization that a `tcgen05.fence::after_thread_sync` imported (PTX
    /// memory model).
    pub tc_vc: VC,
    /// Clock of this thread's in-order tcgen05 async pipe (MMA, commit).
    pub pipe_vc: VC,
    /// `.cta_group` of the thread's first MMA-pipe operation (0: none yet, 1, 2); operations
    /// of the other group go to its second pipe (`Machine::pipe2_actor_id`, `pipe2_vc`).
    pub pipe_grp: u8,
    pub pipe2_vc: VC,
    /// Async actor ids this thread took so far (`Machine::async_actor`; deterministic ids).
    pub tma_count: u32,
    /// Landings of this thread's cp.async copies by waits so far: the epochs of its wait actor
    /// (`Machine::cp_wait_actor`, E-CPASYNC).
    pub cp_epoch: u32,
    /// E-CLOCK opt-in (PTXV_CLOCK_STEP): the value the next clock read returns.
    pub clock: u64,
    /// E-CLOCK: back edges of clock branches taken in a row (clock.rs `MAX_CLOCK_ITERS`).
    pub clock_iters: u8,
    /// E-CLOCK: this thread has read bytes a clock-derived store wrote (`Machine::ctaint`):
    /// its branches on loaded values (`Machine::mem_branch_pcs`) are clock branches.
    pub clock_mem: bool,
    /// Registers written by tcgen05.ld that are undefined until tcgen05.wait::ld.
    pub pending_ld: Vec<(u8, u16)>,
    /// Clock (of the thread's tcgen05.ld/st actor) of its last tcgen05.st / tcgen05.ld: what
    /// wait::st / wait::ld orders before its later MMA-pipe operations (S-TCGEN05).
    pub tc_st: u32,
    pub tc_ld: u32,
    /// Clock (of the same actor) of the thread's first tcgen05.ld since its last
    /// tcgen05.wait::ld, 0 if none: that load and later ones are not complete, so
    /// fence::before_thread_sync does not release them (S-TCGEN05).
    pub tc_ld_unwaited: u32,
    /// Likewise its first tcgen05.st since its last tcgen05.wait::st.
    pub tc_st_unwaited: u32,
    /// Bitset over `regs`: slot written since launch (S-INIT: reads of unwritten registers
    /// are reported).
    pub written: Vec<u64>,
    /// S-INIT, PTX virtual registers: the undefined bits of written registers that are partly
    /// undefined (sorted by index; absent: wholly defined). A 64-bit register packs two f32x2
    /// lanes that may come from different rows (`mov.b64 %rd, {%r1, %r2}`).
    pub vund: Vec<(u16, u64)>,
    /// fence.proxy.async executions of this thread (S-PROXY): its clock then (own entry and
    /// what it had observed of every other thread).
    pub fences: Vec<VC>,
    /// fence.proxy.alias executions of this thread, and its start after a stream-ordered
    /// previous grid (a kernel boundary orders aliases too), as for `fences` (S-VA-ALIAS).
    pub alias_fences: Vec<VC>,
    /// Proxy fences of this thread that order global memory (`fence.proxy.async.global` or
    /// `fence.proxy.async`; SASS `FENCE.VIEW.ASYNC.G`, or `FENCE.VIEW.ASYNC.S` right after
    /// `MEMBAR.ALL.GPU`), and its start after a stream-ordered previous grid, as for `fences`
    /// (S-GPROXY).
    pub gfences: Vec<VC>,
    /// Strong global accesses of this thread to bytes with several mappings since its last
    /// fence.proxy.alias (S-VA-ALIAS), merged per (allocation, mapping, write) into byte
    /// ranges, sorted (canonical state), never truncated. A strong access through another
    /// mapping of an overlapping range, one of the two a write, is an alias race.
    pub alias_pending: Vec<AliasPending>,
    /// This thread has executed a fence.proxy.alias (S-VA-ALIAS: covers its strong accesses
    /// that observe another thread's strong write through the other view).
    pub alias_fenced: bool,
    /// Bulk groups committed so far (the open group's id).
    pub bulk_group: u32,
    /// The read scoreboard of each committed bulk group (SASS `UTMACMDFLUSH`'s rbar, by group
    /// id; `u8::MAX`: none, or PTX). A SASS wait counts the groups of its scoreboard only.
    pub bulk_sb: Vec<u8>,
    /// Completed bulk operations not yet waited for: (group, clock of its shared-memory reads,
    /// clock of its global writes). A `wait_group.read` acquires the first and keeps the entry;
    /// a full wait acquires the second and drops it (SASS without the L1 model: defers it to
    /// `bulk_vis`).
    pub bulk_done: Vec<(u32, VC, VC)>,
    /// SASS front without the L1 model: global-write clocks of bulk groups a `DEPBAR` (or a
    /// control-word wait) on their scoreboard covered, not yet acquired. The thread's next
    /// `CCTL.IVALL` acquires them (ASSUME(E-ASYNC-STEP): ptxas emits it after the `DEPBAR` for
    /// the full `wait_group` only, and until then a line of the destination this SM filled
    /// before the store can still be read, E-HW-L1).
    pub bulk_vis: VC,
    /// Barrier generation key this thread was released from and has not run since (DPOR:
    /// its next transition acquires it).
    pub acq_bar: Option<u64>,
    /// Visibility of mbarrier inits beyond `vc` (E-CLUSTER-BAR): a relaxed cluster arrive
    /// after fence.mbarrier_init releases only the inits. Checked as `init HB vc or vis`.
    pub vis: VC,
    /// Own clock at the last full fence (MEMBAR) and at the last fence.mbarrier_init: what a
    /// relaxed cluster arrive releases.
    pub membar_vc: VC,
    pub minit_vc: VC,
    /// Scope of the last release fence (with `membar_vc`).
    pub membar_scope: Scope,
    /// Own clock at the last `fence.release.sync_restrict::shared::cta.cluster` (SASS: also
    /// `MEMBAR.ALL.CTA`, its lowering): what a later relaxed cluster-scope mbarrier update
    /// releases, for this CTA's shared memory only (S-MBAR-SCOPE).
    pub srel_vc: VC,
    /// `vc_sh` at the last release fence (with `membar_vc`) and at the last
    /// `srel_vc` fence: what those release of shared-memory-only happens-before.
    pub membar_sh: Vec<(Cover, VC)>,
    pub srel_sh: Vec<(Cover, VC)>,
    /// Happens-before that holds for shared-memory accesses only (`Cover::Smem` /
    /// `SmemOf`): acquired through `fence.acquire.sync_restrict::shared::cluster`, a SASS
    /// `SYNCS.PHASECHK`'s cluster-scope acquire, or a restricted release (S-MBAR-SCOPE).
    pub vc_sh: Vec<(Cover, VC)>,
    /// mbarrier phase updates observed by relaxed waits, not yet acquired: (writer, release
    /// clock, release scope, cover) (S-MBAR-SCOPE).
    pub acq_pending_mb: Vec<(u32, VC, Scope, Cover)>,
    /// mbarrier updates of other CTAs this thread observed landing (a completed phase it
    /// waited on), as clocks of their threads: S-CLUSTER-LIFE counts them as ordered before
    /// this thread's exit even when the wait did not synchronize with them (S-MBAR-SCOPE).
    pub landed: VC,
    /// Releases observed by relaxed strong loads, not yet acquired (a later acquire fence
    /// does): (writer, release clock, release scope).
    pub acq_pending: Vec<(u32, VC, Scope)>,
    /// What this thread acquired from other threads' releases (E-SMEM-CUMUL: a SASS shared
    /// store forwards it).
    pub acq_view: VC,
    /// Cluster-barrier generation this thread arrived in and has not waited for yet.
    pub cl_arrived: Option<u32>,
    /// Tensor maps in memory this thread fenced into the tensormap proxy
    /// (`fence.proxy.tensormap::generic.acquire`, SASS `UTMACCTL.IV`): (address, contents
    /// then). A TMA through a device-built map must find it here with unchanged contents.
    pub tmap_fence: Vec<(u64, Vec<u8>, u32)>,
    /// Peeks the explorer runs as failed tests (`explore::peekskip_set`) whose worlds have
    /// not joined yet: (CTA, mbarrier offset, parity, peek pc).
    pub peekwin: Vec<(u16, u32, u32, u32)>,
    /// Local memory (STL/LDL spills), sorted by word address: low 32 bits value, high 32
    /// bits its undefined-bit mask (S-INIT).
    pub local: Vec<(u32, u64)>,
    /// cp.async copies not yet complete (E-CPASYNC): (group, shared offset, bytes, clock at
    /// issue, pc). `cp_group` counts committed groups (the open group's id).
    pub cp_pending: Vec<CpPending>,
    pub cp_group: u32,
    /// Result of the thread's last completed `bar.red` (E-BAR-RED): the reduction, the number
    /// of true contributions, and whether every waiter's was true.
    pub bar_res: Option<(ptxv_ir::BarRedOp, u32, bool)>,
    /// E-COLLECTIVE: the `WARPSYNC.COLLECTIVE` this lane parked at, while it runs an identical
    /// region elsewhere with the rest of its mask group (lanes of one mask at different
    /// collective instructions enter together; GB300: probes/coll_split).
    pub coll_home: Option<u32>,
    /// E-READ-DONE (case `read_done`): loads whose destination registers no instruction has
    /// used yet: (register file, first register, count, clock of the read).
    pub ld_pend: Vec<(u8, u16, u8, u32)>,
    /// SASS: the thread's `LDTM`s whose tensor-memory reads are outstanding, in issue order:
    /// (write scoreboard, 7 for none; clock of its tcgen05.ld/st actor). S-TCGEN05.
    pub ldtm_pend: Vec<(u8, u32)>,
}

impl Thread {
    /// The tcgen05.ld/st actor's clock of the thread's first load or store not yet waited
    /// for (`tc_ld_unwaited`, `tc_st_unwaited`), 0 if none.
    pub fn tc_unwaited(&self) -> u32 {
        match (self.tc_ld_unwaited, self.tc_st_unwaited) {
            (0, x) | (x, 0) => x,
            (x, y) => x.min(y),
        }
    }
}

#[derive(Clone, Debug, Hash)]
pub struct CpPending {
    pub group: u32,
    /// The warp's LDGDEPBAR count when it was issued (SASS: SB0 counts the warp's commits).
    pub wgroup: u32,
    pub off: u32,
    /// The global source (allocation, byte offset); None: zero fill.
    pub src: Option<(usize, u64)>,
    pub data: Vec<u8>,
    pub snapshot: Option<crate::num::ReadSnapshot>,
    pub vc: VC,
    pub pc: u32,
}

#[derive(Clone, Debug, Hash)]
pub struct Warp {
    pub cta: u16,
    pub ur: Vec<u64>,
    pub up: u8,
    /// Bitset over `up`: uniform predicate written since launch (S-INIT-PRED; `UPT` is
    /// always defined).
    pub up_def: u8,
    /// Bitset over `ur`: uniform register written since launch.
    pub ur_written: [u64; 4],
    /// Lanes recorded by `BSSY Bb` for each convergence barrier b (E-BSYNC).
    pub bmask: [u32; 16],
    /// SASS: LDGDEPBAR executions by the warp (any lanes): the cp.async scoreboard counts
    /// the warp's commits, not each thread's (E-CPASYNC).
    pub cp_seq: u32,
    /// SASS: copies a `DEPBAR` of some lanes retired while other live lanes of the warp ran
    /// elsewhere: (the warp's commit count at their issue, their writes' clock, lanes that
    /// have not executed a `DEPBAR` covering them yet). Such a lane's covering wait joins
    /// the clock (E-CPASYNC, `cp_async_wait_warp`). Empty unless waits diverge.
    pub cp_landed: Vec<(u32, VC, u32)>,
}

/// The memory a release or acquire covers (S-MBAR-SCOPE): everything; the SM-local memories
/// (shared and tensor memory: what a SASS `SYNCS.PHASECHK` acquires from another CTA without
/// the `CCTL.IVALL` that invalidates the L1 for global memory); shared memory only
/// (`fence.acquire.sync_restrict::shared::cluster`); or one CTA's shared memory only
/// (`fence.release.sync_restrict::shared::cta`, SASS `MEMBAR.ALL.CTA` before an arrive).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Cover {
    All,
    OnSm,
    Smem,
    SmemOf(u16),
    /// Orders events but covers no memory: an mbarrier update a wait observed without
    /// synchronizing with it (the certificate's phase determinism and the cluster lifetime
    /// use it; the race detector does not).
    Order,
}

impl Cover {
    /// What both cover (None: nothing).
    pub fn and(self, o: Cover) -> Option<Cover> {
        match (self, o) {
            (Cover::Order, _) | (_, Cover::Order) => Some(Cover::Order),
            (Cover::All, x) | (x, Cover::All) => Some(x),
            (Cover::OnSm, x) | (x, Cover::OnSm) => Some(x),
            (Cover::Smem, x) | (x, Cover::Smem) => Some(x),
            (Cover::SmemOf(a), Cover::SmemOf(b)) => (a == b).then_some(Cover::SmemOf(a)),
        }
    }

    /// Does it cover an access to shared memory of CTA `c` (`None`: tensor memory)?
    pub fn covers(self, c: Option<u16>) -> bool {
        match (self, c) {
            (Cover::Order, _) => false,
            (Cover::All | Cover::OnSm, _) => true,
            (Cover::Smem, Some(_)) => true,
            (Cover::SmemOf(a), Some(b)) => a == b,
            _ => false,
        }
    }
}

/// One update of an mbarrier phase (arrive, expect_tx / complete_tx, an async unit's
/// complete-tx or arrive) as a strong read-modify-write of the mbarrier, with what it
/// releases. ASSUME(S-MBAR-SCOPE): a wait synchronizes with it only when the two are morally
/// strong (each one's scope includes the other's thread) and the release's scope includes
/// the waiter.
#[derive(Clone, Debug, Hash)]
pub struct MbRel {
    /// CTA of the thread the update counts as performed by (scope inclusion).
    pub cta: u16,
    /// Scope of the update as a strong operation.
    pub scope: Scope,
    /// What it releases (empty for a relaxed update with no earlier release fence), at
    /// which scope, covering which memory.
    pub rel: VC,
    pub rel_scope: Scope,
    pub cover: Cover,
    /// An async unit's completion (TMA, cp.async.bulk, st.async, tcgen05.commit, CLC),
    /// attributed to its issuing thread (S-MBAR-SCOPE R-ISSUER; `Launch::async_mb_dest`
    /// attributes it to the mbarrier's CTA instead).
    pub asyncx: bool,
}

#[derive(Clone, Debug, Hash, Serialize)]
pub struct Mbar {
    pub expected: u32,
    pub pending: u32,
    pub tx: i64,
    /// Index of the current (incomplete) phase.
    pub phase: u32,
    /// Updates of the current phase (S-MBAR-SCOPE).
    #[serde(skip)]
    pub cur: Vec<MbRel>,
    /// Updates of the most recent completed phase with parity 0 / 1. Parity 1 starts as the
    /// init: `try_wait.parity 1` on phase 0 succeeds immediately.
    #[serde(skip)]
    pub done: [Vec<MbRel>; 2],
    /// Clocks of the threads behind the current / completed phase's updates (a waiter that
    /// observed the phase saw them land: cluster lifetime, `Thread::landed`).
    #[serde(skip)]
    pub cur_land: VC,
    #[serde(skip)]
    pub done_land: [VC; 2],
    #[serde(skip)]
    pub init: Epoch,
    #[serde(skip)]
    pub atomics: Vec<Epoch>,
}

/// Tensor memory of one CTA (one SM in this model): 128 lanes x 512 columns of 32 bits.
#[derive(Clone, Debug, Hash)]
// ASSUME(S-TMEM): see ASSUMPTIONS.md
pub struct Tmem {
    pub words: Pages<u32>,
    pub sh: Pages<Sh>,
    /// Per word: 0 = allocated, never written; 1 = written; 2 = not allocated.
    pub init: Pages<u8>,
    /// Allocator bitmap: bit i = 32-column unit i is allocated.
    pub alloc: u16,
    pub relinquished: bool,
    /// The CTA allocated at least once: it holds its allocation permit until it relinquishes
    /// or exits (E-TMEM-PERMIT, tenant.rs).
    pub permit: bool,
    /// The tensor core's A collector buffer (S-TC-COLLECTOR): after a `.collector::a::fill`
    /// (or a `use` that keeps it), the fingerprint of the A operand it holds and the pc of the
    /// fill; `None` once discarded (`lastuse`, `discard` / no qualifier).
    pub coll_a: Option<(u64, u32)>,
    /// The last MMA that used or changed the collector (its pipe actor and clock at issue),
    /// for the order check (S-TC-COLLECTOR).
    pub coll_last: Option<(u32, u32)>,
}

pub const TMEM_LANES: usize = 128;
pub const TMEM_COLS: usize = 512;

/// Word index of (lane, column) in TMEM storage: column-major, so an MMA writing N columns
/// of all 128 lanes touches N/4 pages instead of 128.
#[inline]
pub fn tmem_idx(lane: usize, col: usize) -> usize {
    col * TMEM_LANES + lane
}

impl Tmem {
    pub fn new() -> Tmem {
        let n = TMEM_LANES * TMEM_COLS;
        Tmem {
            words: Pages::new(n),
            sh: Pages::uniform(n, Sh::default()),
            init: Pages::from_vec(vec![2u8; n], 2),
            alloc: 0,
            relinquished: false,
            permit: false,
            coll_a: None,
            coll_last: None,
        }
    }
}

impl Default for Tmem {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Clone, Debug, Hash)]
pub struct Bar {
    /// Threads waiting (`bar.sync`) in the current generation, sorted.
    pub arrived: Vec<u32>,
    /// Threads that arrived without waiting (`bar.arrive`) in the current generation, sorted;
    /// a repeated arrival is kept (counted twice, E-BAR-ARRIVE) and reported.
    pub arrivers: Vec<u32>,
    /// SASS lanes guarded off at a `BAR` the rest of their warp executed (ASSUME(E-ALIGNED),
    /// E-BAR-EXIT: a BAR arrives per warp), sorted: counted as arrived, neither waiting nor
    /// releasing.
    pub skipped: Vec<u32>,
    /// Join of the arrivers' clocks at their arrivals: released to the waiters.
    pub rel: VC,
    /// Their shared-memory-only happens-before (`Thread::vc_sh`, S-MBAR-SCOPE).
    pub rel_sh: Vec<(Cover, VC)>,
    /// Per waiter, sorted: its `bar.red` reduction (None: a plain `bar.sync`) and its
    /// contribution (E-BAR-RED).
    pub red: Vec<(u32, Option<ptxv_ir::BarRedOp>, bool)>,
    /// Explicit thread count, or None for "all threads of the CTA", which waits only for
    /// threads that have not exited (GB300 probe, ASSUMPTIONS.md E-BAR-EXIT).
    pub count: Option<u32>,
}

#[derive(Clone, Debug, Hash)]
pub enum AsyncKind {
    Tma {
        cta: u16,
        dst: u32,
        /// CTA holding the mbarrier (the peer's for `.cta_group::2`).
        mbar_cta: u16,
        mbar: u32,
        tmap: Arc<TmapDesc>,
        coords: Vec<i64>,
    },
    /// tcgen05.mma, operands decoded at issue.
    Mma {
        cta: u16,
        /// cta_group::2: the odd CTA of the pair (its A rows, B half and D rows).
        peer: Option<u16>,
        d: u32,
        adesc: u64,
        bdesc: u64,
        idesc: u32,
        enable: bool,
        /// `adesc` is A's TMEM address (A in tensor memory).
        a_tmem: bool,
        kind: ptxv_ir::MmaKind,
        /// Block-scaled kinds: TMEM addresses of the A and B scale factors.
        scale: Option<(u32, u32)>,
        /// scale-input-d: D = A B + D * 2^-scale_d.
        scale_d: u8,
        /// `.collector::a::*` (S-TC-COLLECTOR).
        coll: ptxv_ir::MmaCollector,
    },
    /// tcgen05.cp (32x128b.warpx4): the smem matrix `sdesc` -> TMEM `taddr`, 4 columns; with
    /// `peer` (cta_group::2) the same copy in the peer CTA, from its own shared memory.
    Cp {
        cta: u16,
        peer: Option<u16>,
        taddr: u32,
        sdesc: u64,
    },
    /// tcgen05.commit: arrive on `mbar` once every earlier op in the same pipe completed.
    Commit { cta: u16, mbar: u32 },
    /// clusterlaunchcontrol.try_cancel response: 16 bytes to `resp` + complete_tx(16) on `mbar`.
    /// `others`: the rest of the issuer's cluster, for a multicast try_cancel
    /// (UGETNEXTWORKID.BROADCAST): each gets the response and the completion on its own
    /// mbarrier at the same offsets (probes/clc_mcast).
    Clc {
        cta: u16,
        resp: u32,
        mbar: u32,
        others: Vec<u16>,
    },
    /// `cp.async.mbarrier.arrive` (async half): the issuer's cp.async writes then, on `mbar`,
    /// complete one transaction unit (`tx`) or arrive once.
    CpArrive {
        cta: u16,
        mbar: u32,
        writes: Vec<CpPending>,
        tx: bool,
    },
    /// ASSUME(S-REMOTE-INFLIGHT): a remote mbarrier arrival / complete_tx (`mbarrier.arrive
    /// .shared::cluster` to another CTA) or `st.async` into another CTA, in flight from issue
    /// until delivered: `store` = (offset, value) of st.async words, `tx` the transaction
    /// bytes (negative: completed), `n` arrivals. The lanes of one warp instruction that
    /// target the same mbarrier are one operation (`fresh` while that instruction executes).
    Remote {
        cta: u16,
        mbar: u32,
        tx: i64,
        n: u32,
        store: Vec<(u32, u32)>,
        fresh: bool,
        /// What the update releases (S-MBAR-SCOPE), fixed at issue.
        rels: Vec<MbRel>,
        /// The issuing lanes' own clocks at issue (S-CLUSTER-LIFE: landed when observed).
        land: VC,
    },
    /// ASSUME(S-NIC-ORDER): QP `qp`'s NIC executes its oldest announced WQE (nic.rs).
    Nic { qp: u16 },
    /// TMA store (shared::cta -> global) in bulk group `group` of its issuer.
    TmaStore {
        cta: u16,
        src: u32,
        tmap: Arc<TmapDesc>,
        coords: Vec<i64>,
        group: u32,
        /// A bulk f32 add-reduction (`Op::TmaStore { red: true }`).
        red: bool,
    },
}

#[derive(Clone, Debug, Hash)]
pub struct AsyncOp {
    /// In-order pipe this op belongs to, or None (unordered): the issuing thread for tcgen05;
    /// `remote_pipe` for remote mbarrier operations (S-REMOTE-INFLIGHT).
    pub fifo: Option<u64>,
    pub actor: u32,
    pub vc: VC,
    pub issuer: u32,
    pub pc: u32,
    pub kind: AsyncKind,
    /// The issuer's mbarrier-init visibility at issue (`Thread::vis`).
    pub vis: VC,
    /// The issuer's shared-memory-only happens-before at issue (`Thread::vc_sh`): what the
    /// operation's own shared-memory accesses are ordered after (S-MBAR-SCOPE).
    pub sh: Vec<(Cover, VC)>,
}

/// A 2-D/N-D tiled tensor map (what cuTensorMapEncodeTiled was given). `base` may point
/// inside an allocation (device-built descriptors, rung 3a).
#[derive(Clone, Debug, Serialize, Hash, PartialEq, Eq)]
pub struct TmapDesc {
    /// The grid whose parameter block holds it, and its offset there.
    pub grid: usize,
    pub param_off: u32,
    pub buffer: String,
    pub base: u64,
    pub esize: u32,
    /// Innermost first.
    pub dims: Vec<u64>,
    /// Byte strides of dims 1.. (dim 0 is contiguous).
    pub strides: Vec<u64>,
    /// Shared-memory swizzle span in bytes: 0 (none), 32, 64 or 128.
    pub swizzle: u32,
    pub boxd: Vec<u32>,
    /// 4-bit elements (16U4_ALIGN8B): dims / box in bytes, coordinate 0 in elements.
    pub elem4: bool,
    /// The element type when it is f32 / f16 / bf16 (what a tensor add-reduction adds);
    /// None for other types (a reduction through them is unmodeled).
    pub fdtype: Option<ptxv_ir::FFmt>,
    /// A shared-to-shared bulk copy (`UBLKCP.S.S`): the source is this CTA's shared memory at
    /// offset `base`, not global memory.
    pub smem_src: Option<u16>,
    /// im2col-mode map (`boxd` is then [channels, pixels]; im2col.rs).
    pub im2col: Option<crate::tmap::Im2col>,
}

/// The mapping id of a multicast (NVLS) access (`Epoch::view`, S-VA-ALIAS).
pub const MC_VIEW: u16 = u16::MAX;

/// Strong accesses of a thread through one mapping since its last fence.proxy.alias
/// (`Thread::alias_pending`): bytes [lo, hi) of `alloc`, writes or reads, the last at `pc`.
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, PartialOrd, Ord)]
pub struct AliasPending {
    pub alloc: u32,
    pub view: u16,
    pub write: bool,
    pub lo: u64,
    pub hi: u64,
    pub pc: u32,
}

/// A further virtual mapping of part of an allocation (case `views`, S-VA-ALIAS): addresses
/// [base, base + size) reach bytes [alloc_off, alloc_off + size) of `alloc`.
#[derive(Clone, Debug)]
pub struct VaView {
    pub name: String,
    pub base: u64,
    pub size: u64,
    pub alloc: usize,
    pub alloc_off: u64,
}

/// Accesses through mappings used from different address spaces (case `alias_xspace`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum XSpace {
    /// `alias-xspace` finding, graded undecided (default).
    Undecided,
    /// `race:alias`, as within one address space.
    Race,
    /// One location: not checked.
    One,
}

/// An NVLS multicast object: addresses [base, base + size) are a multicast view of one
/// bound range per GPU (S-MULTIMEM).
#[derive(Clone, Debug)]
pub struct McObj {
    pub name: String,
    pub base: u64,
    pub size: u64,
    pub binds: Vec<McBind>,
}

#[derive(Clone, Copy, Debug)]
pub struct McBind {
    pub device: u32,
    /// Offset in the multicast object.
    pub offset: u64,
    pub alloc: usize,
    pub alloc_off: u64,
    pub size: u64,
}

#[derive(Clone, Debug)]
pub struct Alloc {
    pub name: String,
    pub base: u64,
    pub init: Vec<u8>,
    /// S-INIT (scratch buffers): every byte is undefined until written (`init` is a placeholder); a
    /// load of an unwritten byte gives undefined bits, reported where they are used.
    pub undef: bool,
    /// Declared dead workspace (`init = "undef"`, `scratch = true`, not an output): nothing
    /// reads it after the case, so undefined data left in it when the launch ends (an
    /// L2 discard, a partial write) is not an S-INIT violation. ASSUME(S-INIT-DEAD)
    pub dead: bool,
}

/// One grid of a launch. A multi-kernel launch (rung 1e: PDL) runs several grids of one
/// stream in order; their code is concatenated (`Kernel::concat`) and their CTAs numbered
/// globally in launch order.
#[derive(Clone, Debug)]
pub struct Grid {
    pub grid: [u32; 3],
    pub block: [u32; 3],
    pub cluster: [u32; 3],
    /// The launch named a cluster shape (`%is_explicit_cluster`).
    pub explicit_cluster: bool,
    /// The launch's preferred cluster dimension (cudaLaunchAttributePreferredClusterDimension),
    /// if any: the hardware launches each group of CTAs either as one cluster of this shape or
    /// as clusters of the regular one. ASSUME(S-CGA-SHAPE)
    pub preferred: Option<[u32; 3]>,
    /// SR_CgaSize of every CTA of this grid: 0 when `cluster` is the regular shape, 1 when it
    /// is the preferred one (`Launch::preferred_shape`). ASSUME(S-CGA-SHAPE)
    pub cga_shape: u32,
    pub params: Vec<u8>,
    /// First instruction of this grid's kernel.
    pub entry: u32,
    /// Global id of this grid's first CTA.
    pub cta0: u32,
    pub smem_bytes: u32,
    pub param_cbank: u32,
    /// Launched with programmatic stream serialization (E-PDL): its CTAs may launch once
    /// every CTA of the previous grid has triggered or exited, instead of after that grid
    /// completed.
    pub pdl: bool,
    /// GPU this grid runs on. A grid is stream-ordered (and `pdl` applies) only after the
    /// previous grid on the same GPU.
    pub device: u32,
    /// Stream-order override (`NextLaunch::after`): the grid this one waits for instead of
    /// the previous one on its GPU (None inside: nothing).
    pub after: Option<Option<usize>>,
    /// Further grids (earlier, same GPU) that must complete before this one's CTAs launch
    /// (`NextLaunch::waits`: event waits, graph joins); ordered as by a predecessor launched
    /// without PDL (E-PDL).
    pub waits: Vec<usize>,
    /// Memory synchronization domain (`NextLaunch::mem_sync_domain`): `.gpu` scope includes
    /// only the threads of grids on the same GPU in the same domain (S-SCOPE).
    pub mem_domain: u32,
    /// Constant-bank words (bank, byte offset, value) that hold a module data global's
    /// address: its buffer's address + addend (S-GLOBAL-INIT). They override the bank bytes.
    pub cbank_relocs: Vec<(u8, u32, u64)>,
}

impl Grid {
    pub fn threads_per_cta(&self) -> u32 {
        self.block.iter().product()
    }
    pub fn ctas(&self) -> u32 {
        self.grid.iter().product()
    }
    pub fn cluster_size(&self) -> u32 {
        self.cluster.iter().product()
    }
}

pub type MmaCache = std::sync::Arc<
    std::sync::Mutex<std::collections::HashMap<(u64, u64), std::sync::Arc<Vec<u32>>>>,
>;

/// Everything about a launch that does not change during execution.
#[derive(Clone, Debug)]
pub struct Launch {
    /// Launch-level notes (sig, message): S-GLOBAL-INIT's zero-initialized module globals.
    pub notes: Vec<(String, String)>,
    /// C-TICKET-ADDR declarations (case `permuted`): (counter allocation, row allocations).
    pub perm: Vec<(usize, Vec<usize>)>,
    pub grids: Vec<Grid>,
    /// E-LAUNCH-ORDER: the case's launch order was recorded by the capture or declared by its
    /// author (case `launch_order`). Launch-order exclusions apply only then
    /// (`order_completes_before`).
    pub order_recorded: bool,
    pub allocs: Vec<Alloc>,
    /// NVLS multicast objects (case `multicast`, S-MULTIMEM).
    pub mcast: Vec<McObj>,
    /// E-MC-ALIAS-HW: multicast and unicast views of the same bytes are one location (case
    /// `mc_alias = "hw"`); default false: the PTX alias proxy (S-VA-ALIAS).
    pub mc_alias_hw: bool,
    /// Further mappings of allocations (case `views`, S-VA-ALIAS).
    pub views: Vec<VaView>,
    /// Virtual address space of each GPU (case `spaces`; empty: one).
    pub spaces: Vec<u32>,
    pub alias_xspace: XSpace,
    /// Per allocation: reached through more than one mapping (a view or a multicast bind),
    /// so the alias-proxy check applies to it.
    pub aliased: Vec<bool>,
    pub tensormaps: Vec<TmapDesc>,
    pub outputs: Vec<usize>,
    /// Groups of allocations that must end bitwise equal (case `replicas`).
    pub replicas: Vec<Vec<usize>>,
    /// Tensor-core arithmetic model (from calibration).
    pub mma_model: crate::fp::MmaModel,
    /// E-EXIT-FENCE: an exiting thread's fence.acq_rel scope (case `exit_fence`).
    pub exit_fence: Option<ptxv_ir::Scope>,
    /// E-POLL-ACQ: leaving a poll loop acquires (case `poll_acquire`).
    pub poll_acquire: bool,
    /// E-SMEM-CUMUL: a SASS shared-memory store releases what its thread acquired (case
    /// `smem_cumulative`).
    pub smem_cumulative: bool,
    /// E-HW-L1: per-SM non-coherent L1 model (case `hw_l1`).
    pub hw_l1: bool,
    /// E-L1-CORES: co-resident launches share SM L1s; placements enumerated (l1_cores.rs).
    pub l1_cores: bool,
    /// E-L1-HAVOC: an outside agent may fill any line on any SM at any time (l1_cores.rs).
    pub l1_havoc: bool,
    /// Limited residency and the TMEM co-tenant (residency.rs, tenant.rs). Default: off.
    pub env: crate::residency::Env,
    /// E-READ-DONE: no load buffering through completed reads (case `read_done`).
    pub read_done: bool,
    /// E-REMOTE-FIFO: remote arrivals from one CTA to one CTA land in happens-before issue order.
    pub remote_fifo: bool,
    /// E-HW-BSYNC-COLLECTIVE: a BSYNC releases the lanes parked at it while the rest of its
    /// lane mask is parked at a WARPSYNC.COLLECTIVE (GB300, fixtures/e_bsync_parked_collective).
    /// Only the GB300 hardware profile sets it (hwprofile.rs); never set from a case.
    pub hw_bsync_coll: bool,
    /// E-HW-SPLIT-UR (GB300 profile): no switch between a warp's splits inside a
    /// uniform-register window (urwin.rs). Default: off.
    pub hw_split_ur: bool,
    /// The GB300 hardware model's ordering (E-HW-L1, two-level check, twolevel.rs): memory
    /// behind the L1s is sequentially consistent, so every strong write publishes its
    /// writer's clock and every strong read that observes it acquires that clock, at any
    /// scope. Never set from a case; only the second level of `ptxv check` uses it.
    pub hw_order: bool,
    /// E-TC-FENCE-NOP: every instruction of the PTX front runs as if
    /// `tcgen05.fence::after_thread_sync` preceded it and `tcgen05.fence::before_thread_sync`
    /// followed it (GB300's tcgen05 fences are NOPs). Never set from a case: `ptxv check`
    /// reruns a PTX check with TMEM-related races this way, and the hardware model uses it.
    pub tc_fence_implied: bool,
    /// S-MBAR-SCOPE: an async unit's mbarrier completion (TMA, cp.async.bulk, st.async,
    /// tcgen05.commit, CLC) counts as performed by a thread of the mbarrier's CTA instead of
    /// its issuing thread. Never set from a case: `ptxv check` reruns a PTX check with races
    /// this way, and a race that is gone becomes a contract finding (the PTX ISA does not say
    /// which thread performs the complete-tx).
    pub async_mb_dest: bool,
    /// Refuted assumptions this launch was explicitly allowed to use (case `assume_refuted`,
    /// `--assume-refuted`, `PTXV_ASSUME_REFUTED`) and does use.
    pub refuted_opt_in: Vec<String>,
    /// GPU-initiated RDMA queue pairs and NVLink domains (nic.rs, docs/nic.md).
    pub nic: Option<Arc<crate::nic::NicCfg>>,
}

impl Launch {
    /// `PTXV_HW_MODEL=1`: check under the GB300 hardware model directly (the second level of
    /// the two-level check, twolevel.rs: E-HW-L1 with `hw_order`, no E-POLL-ACQ).
    pub fn with_hw_model_env(self) -> Launch {
        if std::env::var("PTXV_HW_MODEL").as_deref() == Ok("1") {
            crate::twolevel::hw_launch(&self)
        } else {
            self
        }
    }

    /// CTAs of all grids.
    pub fn ctas(&self) -> u32 {
        self.grids.iter().map(|g| g.ctas()).sum()
    }
    /// Index of the grid that global CTA `cta` belongs to.
    pub fn grid_index(&self, cta: u32) -> usize {
        self.grids
            .iter()
            .rposition(|g| g.cta0 <= cta)
            .expect("CTA id below the first grid")
    }
    pub fn g(&self, cta: u32) -> &Grid {
        &self.grids[self.grid_index(cta)]
    }
    /// The grid that grid `gi` is stream-ordered after: the previous one on its GPU.
    pub fn prev_grid(&self, gi: usize) -> Option<usize> {
        if let Some(a) = self.grids[gi].after {
            return a;
        }
        (0..gi)
            .rev()
            .find(|&j| self.grids[j].device == self.grids[gi].device)
    }
    /// ASSUME(E-PDL): the grids that must complete before grid `gi`'s CTAs launch: its
    /// predecessor when `gi` was launched without PDL, and its `waits`.
    pub fn full_preds(&self, gi: usize) -> impl Iterator<Item = usize> + '_ {
        let p = self.prev_grid(gi).filter(|_| !self.grids[gi].pdl);
        p.into_iter().chain(self.grids[gi].waits.iter().copied())
    }
    /// Is grid `j` a predecessor (`prev_grid` or `waits`) of grid `i`?
    pub fn is_pred(&self, j: usize, i: usize) -> bool {
        self.prev_grid(i) == Some(j) || self.grids[i].waits.contains(&j)
    }
    /// ASSUME(E-PDL): does grid `j` complete before grid `i` starts, by the order the case
    /// states? Every grid starts after each of its predecessors started (a PDL grid after its
    /// predecessor's CTAs triggered or exited), and after each `full_preds` grid completed.
    /// So `j` completes before `i` starts when `j` is a full predecessor of `i` or of a grid
    /// on a predecessor path to `i`. Predecessors are earlier grids, so the search ends.
    pub fn completes_before(&self, j: usize, i: usize) -> bool {
        if j >= i {
            return false;
        }
        let mut seen = vec![false; i + 1];
        let mut stack = vec![i];
        while let Some(x) = stack.pop() {
            if std::mem::replace(&mut seen[x], true) {
                continue;
            }
            if self.full_preds(x).any(|p| p == j) {
                return true;
            }
            stack.extend(self.prev_grid(x).filter(|&p| p > j));
            stack.extend(self.grids[x].waits.iter().copied().filter(|&p| p > j));
        }
        false
    }
    /// ASSUME(E-LAUNCH-ORDER): `completes_before` for launch-order exclusions (S-NC's
    /// stream-order exclusion): only when the case's launch order is recorded or declared. An
    /// older capture may list grids of different streams one after the other although they
    /// ran at the same time (MISS 113).
    pub fn order_completes_before(&self, j: usize, i: usize) -> bool {
        self.order_recorded && self.completes_before(j, i)
    }
    /// ASSUME(E-PDL): does grid `j` start after grid `i` completed with everything `i` did
    /// happening before it: `i` is reached from `j` through `full_preds` links only (a PDL
    /// link orders no memory)?
    pub fn hb_before(&self, i: usize, j: usize) -> bool {
        if i >= j {
            return false;
        }
        let mut seen = vec![false; j + 1];
        let mut stack = vec![j];
        while let Some(x) = stack.pop() {
            if std::mem::replace(&mut seen[x], true) {
                continue;
            }
            for p in self.full_preds(x) {
                if p == i {
                    return true;
                }
                if p > i {
                    stack.push(p);
                }
            }
        }
        false
    }
    pub fn cluster_size(&self, cta: u32) -> u32 {
        self.g(cta).cluster_size()
    }
    /// ASSUME(S-CGA-SHAPE): the launch with every cluster of every grid that has a preferred
    /// cluster dimension in that shape (SR_CgaSize 1), or None when no grid has one.
    pub fn preferred_shape(&self) -> Option<Launch> {
        if self.grids.iter().all(|g| g.preferred.is_none()) {
            return None;
        }
        let mut l = self.clone();
        for g in &mut l.grids {
            if let Some(p) = g.preferred {
                g.cluster = p;
                g.cga_shape = 1;
            }
        }
        Some(l)
    }
    /// The global CTAs of cluster `c` (an id of `cluster_of`).
    pub fn cluster_ctas(&self, c: u32) -> Vec<u32> {
        let g = &self.grids[(c >> 20) as usize];
        (g.cta0..g.cta0 + g.ctas())
            .filter(|&cta| self.cluster_of(cta).0 == c)
            .collect()
    }
    /// (cluster id, rank in the cluster) of global CTA `cta` (x fastest). Cluster ids are
    /// unique across grids.
    pub fn cluster_of(&self, cta: u32) -> (u32, u32) {
        let gi = self.grid_index(cta);
        let g = &self.grids[gi];
        let c = cta - g.cta0;
        let [gx, gy, _] = g.grid;
        let [cx, cy, cz] = g.cluster;
        let (x, y, z) = (c % gx, (c / gx) % gy, c / (gx * gy));
        let rank = (x % cx) + cx * ((y % cy) + cy * (z % cz));
        let (nx, ny) = (gx / cx, gy / cy);
        let id = (x / cx) + nx * ((y / cy) + ny * (z / cz));
        (((gi as u32) << 20) | id, rank)
    }
    /// Global CTA with the given rank in `cta`'s cluster.
    pub fn cta_of_rank(&self, cta: u32, rank: u32) -> Option<u32> {
        let g = self.g(cta);
        if rank >= g.cluster_size() {
            return None;
        }
        let c = cta - g.cta0;
        let [gx, gy, _] = g.grid;
        let [cx, cy, cz] = g.cluster;
        let (x, y, z) = (c % gx, (c / gx) % gy, c / (gx * gy));
        let (bx, by, bz) = (x - x % cx, y - y % cy, z - z % cz);
        let (rx, ry, rz) = (rank % cx, (rank / cx) % cy, rank / (cx * cy));
        Some(g.cta0 + (bx + rx) + gx * ((by + ry) + gy * (bz + rz)))
    }
    /// (ctaid.x, .y, .z) of global CTA `cta` within its grid.
    pub fn ctaid(&self, cta: u32) -> [u32; 3] {
        let g = self.g(cta);
        let c = cta - g.cta0;
        let [gx, gy, _] = g.grid;
        [c % gx, (c / gx) % gy, c / (gx * gy)]
    }
}

/// `barrier.cluster` state of one cluster (E-CLUSTER-BAR).
#[derive(Clone, Debug, Default, Hash)]
pub struct ClusterBar {
    pub gen: u32,
    /// Threads that arrived in generation `gen` (sorted).
    pub arrived: Vec<u32>,
    /// What the arrivals released (full clocks of release arrives, fence snapshots of
    /// relaxed ones), and what they made visible of mbarrier inits.
    pub rel: VC,
    pub vis: VC,
    /// Shared-memory-only happens-before the arrivals released (S-MBAR-SCOPE).
    pub rel_sh: Vec<(Cover, VC)>,
    /// Completed generations still awaited by someone: (gen, rel, vis).
    pub done: Vec<(u32, VC, VC)>,
    /// Their `rel_sh`, by generation.
    pub done_sh: Vec<(u32, Vec<(Cover, VC)>)>,
}

#[derive(Clone, Debug, Default, Serialize, serde::Deserialize)]
pub struct Finding {
    /// "race" | "deadlock" | "violation" | "fault" | "unmodeled" (a model limit was hit: the
    /// verdict is inconclusive for this path, see ASSUMPTIONS.md)
    pub kind: String,
    /// Deduplication key.
    pub sig: String,
    pub msg: String,
    pub trace: Vec<String>,
    /// Instructions involved (races: the earlier and the later access; stale reads: the read
    /// and the fill); used to match findings across models (twolevel.rs). Empty if unknown.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub pcs: Vec<u32>,
    /// The GB300 classification of a PTX-model race / stale read (twolevel.rs): observable
    /// under the hardware model (E-HW-L1) or latent. Extra information: the finding stands.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gb300: Option<crate::twolevel::Gb300>,
    /// C-TICKET-ADDR (`clean-up-to-permutation`, wellsync/perm.rs): per counter word, per
    /// ticket, the row byte ranges (allocation, first byte, end); outputs compare up to a
    /// permutation of the tickets' rows within each word.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub perm_rows: Vec<Vec<crate::wellsync::perm::Ranges>>,
}

/// The mbarriers, shared between machine copies until one writes (`DerefMut` copies a shared
/// map). Copies that only read them (the certificate's look-ahead copies, `skip_equiv`) cost
/// one reference count instead of a copy of every mbarrier and its vector clocks.
#[derive(Clone, Debug, Default)]
pub struct MbarMap(Arc<BTreeMap<(u16, u32), Mbar>>);

impl std::ops::Deref for MbarMap {
    type Target = BTreeMap<(u16, u32), Mbar>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a> IntoIterator for &'a MbarMap {
    type Item = (&'a (u16, u32), &'a Mbar);
    type IntoIter = std::collections::btree_map::Iter<'a, (u16, u32), Mbar>;
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl std::ops::DerefMut for MbarMap {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.0)
    }
}

#[derive(Clone)]
pub struct Machine {
    pub k: Arc<Kernel>,
    pub launch: Arc<Launch>,
    pub threads: CowVec<Cow<Thread>>,
    pub warps: CowVec<Cow<Warp>>,
    pub smem: CowVec<Mem>,
    pub gmem: Vec<Mem>,
    pub tmem: CowVec<Tmem>,
    pub mbars: MbarMap,
    pub bars: BTreeMap<(u16, u32), Bar>,
    pub pending: Vec<AsyncOp>,
    /// Completed async operations (cta, actor, clock, pc), sorted: checked at exit.
    pub completed: Vec<(u16, u32, u32, u32)>,
    /// ASSUME(S-REMOTE-INFLIGHT): remote operations delivered as async steps (not drained by
    /// their issuer's cluster release): (target CTA, mbarrier, phase it counted toward, pc,
    /// issuer), sorted. At the end, the target must have waited on that phase or a later one
    /// (`mbar_seen`): otherwise nothing kept it from exiting before the delivery. A later
    /// cluster release of the issuer (which would have drained it) moves the entry to
    /// `remote`, the happens-before check. Last field: the issuer's clock at issue
    /// (E-REMOTE-FIFO moves an entry to a later arrival's mbarrier and phase).
    pub remote_obs: Vec<(u16, u32, u64, u32, u32, u32)>,
    /// Per mbarrier (cta, offset): the latest phase whose completion a wait observed.
    pub mbar_seen: BTreeMap<(u16, u32), u64>,
    /// Completed generations per named barrier (cta, id).
    pub bar_gen: BTreeMap<(u16, u32), u32>,
    /// Per named barrier (cta, id): the join of the clocks its last completed generation
    /// released (E-BAR-OVER: an arrival not ordered after it might have landed there).
    pub bar_prev: BTreeMap<(u16, u32), VC>,
    /// barrier.cluster state per cluster id.
    pub clusters: BTreeMap<u32, ClusterBar>,
    /// Accesses to another CTA's shared memory (target CTA, actor, actor clock, pc), sorted:
    /// each must happen-before the exit of some thread of the target (S-CLUSTER-LIFE).
    pub remote: Vec<(u16, u32, u32, u32)>,
    /// CTAs that executed `griddepcontrol.launch_dependents` (bit set, E-PDL).
    pub triggered: Vec<u64>,
    /// Not hashed: `grid_done_vc` of grids that completed (fixed from then on), shared so
    /// that every waiter joins one base (8192-thread routing grids, rung 3b).
    pub done_vcs: Vec<Option<VC>>,
    /// Not hashed: a converged warp's atomic performs its lanes in reverse lane order
    /// (default lane order; `PTXV_ATOM_LANES=rev`). PTX leaves the order open; this shows
    /// what depends on it. ASSUME(E-LOCKSTEP)
    pub atom_lanes_rev: bool,
    /// Last strong write per location (space 0 smem / 1 gmem, id, byte offset), S-SCOPE.
    pub syncs: Syncs,
    /// ASSUME(S-FENCE-SC): the SC fences and strong global writes the forced Fence-SC edges
    /// are derived from (`exec.rs` `sc_fence`); None when the kernel has no SC fence at gpu /
    /// sys scope.
    pub sc: Option<Arc<ScLog>>,
    /// Not hashed, static: strong loads that head a poll loop (spin.rs), load pc -> (loop
    /// head H, back branch b): a thread there runs only when this iteration exits the loop.
    pub polls: Arc<BTreeMap<u32, (u32, u32)>>,
    /// Not hashed, static: warp-level poll loops (spin::wpoll_loops), loop head -> back
    /// branch: a split at the head runs only when an iteration would change something.
    pub wpolls: Arc<BTreeMap<u32, u32>>,
    /// Not hashed, static: try_waits whose retry stub recomputes the parity (spin.rs),
    /// try_wait pc -> (stub entry, stub try_wait pc).
    pub stubs: Arc<BTreeMap<u32, (u32, u32)>>,
    /// Not hashed, static: try_waits without the retry shape (spin.rs), executed as
    /// non-blocking tests (CUTLASS's "peek, work, then wait" idiom).
    pub peeks: Arc<BTreeSet<u32>>,
    /// Not hashed, static: control flow for `bar_oversub` (`explore::bar_cfg`).
    pub bar_cfg: Arc<crate::explore::BarCfg>,
    /// Not hashed: PTX elect picks the highest (true) or lowest active lane (E-ELECT) in a
    /// deterministic run whose active set has no entry in `elect_fn`.
    pub elect_high: bool,
    /// Not hashed: PTX elect of an active set without an entry in `elect_fn` branches over
    /// every active lane (E-ELECT; `explore::per_leader_policy`).
    pub elect_any: bool,
    /// Hashed: the PTX leader function chosen so far (E-ELECT: the leader is a fixed
    /// function of the active set): (active lane mask, leader lane), sorted by mask.
    /// Recorded by `Op::Elect` when `elect_any`; preset entries force a choice.
    pub elect_fn: Vec<(u32, u32)>,
    /// E-ELECT: lane classes of the leader search (`elect_lanes::lane_classes`): with
    /// `elect_any`, an active set's leaders are tried one per group of clones. Not hashed
    /// (fixed for a search).
    pub elect_lanes: Option<Arc<crate::elect_lanes::Part>>,
    /// Not hashed, static: a kernel containing try_cancel whose CTAs were all launched up
    /// front (`explore::clc_eager`, E-CLC); executing a try_cancel is then "unmodeled".
    pub clc_eager: bool,
    /// Not hashed, static: some tcgen05.mma has a `.collector::a::*` qualifier other than
    /// discard (S-TC-COLLECTOR); otherwise the collector buffer stays empty and is not tracked.
    pub coll_used: bool,
    /// Not hashed, static: `%smid` of every CTA (global id) in this run's SM placement, and
    /// `%nsmid` (smid.rs). ASSUME(E-SMID)
    pub smid: Arc<Vec<u32>>,
    pub nsmid: u32,
    /// Not hashed, static: log2 of a thread's async actor ids (`async_shift`, `async_ids`).
    pub async_shift: u32,
    /// Not hashed, static: E-CLOCK's opt-in clock step (`exec::clock_step`, `PTXV_CLOCK_STEP`;
    /// 0: clock reads are 0).
    pub clock_step: u64,
    /// Not hashed, static: back-edge branches of pure delay loops, which fall through
    /// (clock.rs), and of every loop that reads the clock. ASSUME(E-CLOCK)
    pub delay_pcs: Arc<std::collections::HashSet<u32>>,
    /// Not hashed, static: uniform-register windows (urwin.rs; empty: none). ASSUME(E-HW-SPLIT-UR)
    pub ur_window: Arc<Vec<bool>>,
    pub clock_loop_pcs: Arc<std::collections::HashSet<u32>>,
    /// Not hashed, static: branches on the clock, explored both ways, and other instructions
    /// whose guard / control operand derives from the clock (`unmodeled`). clock.rs
    pub clock_choice_pcs: Arc<std::collections::HashSet<u32>>,
    pub clock_effect_pcs: Arc<std::collections::HashSet<u32>>,
    /// Not hashed, static: stores of clock-derived values (clock.rs `clock_stores`), and
    /// branches on loaded values (`mem_branches`). ASSUME(E-CLOCK)
    pub clock_store_pcs: Arc<std::collections::HashSet<u32>>,
    pub mem_branch_pcs: Arc<std::collections::HashSet<u32>>,
    /// Shared (space 0, id = CTA) / global (1, id = allocation) bytes a clock-derived store
    /// wrote last. Hashed when not empty. ASSUME(E-CLOCK)
    pub ctaint: Arc<std::collections::BTreeSet<(u8, u32, u64)>>,
    /// Not hashed: while one instruction executes, the value its deciding predicate takes
    /// (a clock choice, clock.rs `choice_pred`). ASSUME(E-CLOCK)
    pub forced_pred: Option<(ptxv_ir::Pred, bool)>,
    /// Not hashed, static: some instruction reads `%smid` (smid.rs).
    pub smid_used: bool,
    /// Not hashed, static: `smid` was set by an enumeration of placements (smid.rs), so
    /// nested checks run it as is.
    pub smid_fixed: bool,
    /// Not hashed, shared by all clones: every global-memory byte any explored transition
    /// read / wrote (`ptxv check --slices`: the cross-cluster conflict check).
    pub gfoot: Option<Arc<GFoot>>,
    /// Thread slices (E-SLICE, slices.rs): every footprint item of every transition, for the
    /// cross-slice check. Not part of the state.
    pub sfoot: Option<Arc<SFoot>>,
    /// Value-insensitive polls (E-POLL-ANY, exec.rs `vip_ok`): what the search learned about
    /// the values of polled locations. Shared by every state; not part of the state.
    pub vip: Option<Arc<VipLearn>>,
    /// Bit 0: set (by any state of the search) when an mbarrier update landed late or took
    /// more than the pending arrivals (footprint space 25 written, `rec_late` and `rec_count`
    /// in exec.rs); the stateful explorer's local reduction of plain arrivals is then unsound
    /// and the search runs again without it (`explore::local_ample`). Bit 1: that reduction
    /// was applied. Shared by every state; not part of the state.
    pub arv_over: Option<Arc<std::sync::atomic::AtomicU8>>,
    /// Reads run as value-insensitive polls, per global byte (allocation, offset): the latest
    /// epoch of each reader. Never cleared by writes: every later write is checked against
    /// them (a race unless ordered after, or both morally strong).
    pub vipr: Arc<BTreeMap<(u32, u64), Vec<Epoch>>>,
    /// Transient, within one step: 1 = the step's strong global load is a value-insensitive
    /// poll (it skips the observation of the location's release).
    pub vip_read: u8,
    /// Not hashed, transient: the current access is an fp32 atomic add with no returned value
    /// (`red.add.f32`) or a bulk-reduction add: `gfoot` files it as a reduction of kind
    /// `red_kind` (`RED_F32` / `RED_BF16` / `RED_F16`), not a read and a write.
    pub red_access: bool,
    pub red_kind: u8,
    /// Not hashed, diagnostic: per global 2-byte granule, the last bulk reduction into it
    /// (async actor, its own clock, kind). A later one of another kind by another actor that
    /// does not happen after it is reported unmodeled (N-RED16).
    pub red16: std::collections::HashMap<(usize, u64), (u32, u32, u8)>,
    /// Not hashed, transient: the current access goes through a multicast (NVLS) view
    /// (S-VA-ALIAS).
    pub mc_view: bool,
    /// Not hashed, transient: the mapping the current asynchronous unit (bulk copy, TMA, NIC)
    /// accesses global memory through (S-VA-ALIAS); None outside one (`access_view` then
    /// reports the access unmodeled when the allocation has several mappings).
    pub async_view: Option<u16>,
    /// E-HW-WEAK-FLAG: the store being published is a weak flag store (publish the clock at
    /// the thread's last sys-scope fence, not its whole history).
    pub weak_pub: bool,
    /// Not hashed, transient: the `AsyncOp::sh` of the async operation completing now
    /// (S-MBAR-SCOPE).
    pub cur_sh: Vec<(Cover, VC)>,
    /// Not hashed: N-MULTIMEM-ROUND16 provenance, which bytes hold an f16 / bf16
    /// multimem.ld_reduce lane rounded by the calibrated rule (`track_round16`; one
    /// deterministic run, for the comparison with the captured GPU output).
    pub round16: Option<Box<crate::multimem::Round16Shadow>>,
    /// Not hashed, transient: the clock a SASS STTM is ordered with under E-TMEM-QUEUE (its
    /// thread's pending LDTMs joined), while `tmem_access` checks it: a conflict ordered only
    /// by it is a `tmem-queue:` contract finding (exec.rs, the `TmemSt` arm).
    pub tmem_queue_vc: Option<VC>,
    /// Not hashed: per global byte, which actor accessed it first, and whether several did /
    /// any wrote (single deterministic runs, `explore_symmetric`).
    pub gown: Option<Box<GOwn>>,
    /// Not hashed, shared by all clones: MMA results by (inputs, accumulator) fingerprint.
    /// The same MMA recurs on many schedules; its arithmetic is a pure function.
    pub mma_cache: MmaCache,
    /// Not hashed: findings produced by the last transition (drained by the explorer).
    pub findings: Vec<Finding>,
    /// Not hashed: numerics shadow (single-path runs only).
    pub num: Option<Box<NumState>>,
    /// Not hashed, transient: the leader (thread index) the current action picked for the
    /// `elect.sync` it starts at (PTX front-end; see ASSUMPTIONS.md E-ELECT).
    pub elect_choice: Option<usize>,
    /// Not hashed, transient: set (to Some(None)) while a register load runs, so a read of
    /// never-written shared memory is recorded here (first byte) instead of reported: the
    /// loaded registers become undefined, and only a use of them is a finding (S-INIT).
    pub stale: Option<Option<u32>>,
    /// S-INIT: the bytes of the shared / global store running that hold undefined bits (bit i =
    /// byte i of its data; `Sh::undef` per byte).
    pub wundef: u128,
    /// S-INIT: global stores of undefined data (allocation, offset, bytes, pc), checked at
    /// terminal states (`exit_checks`); an index into `Sh::undef`, not part of the state hash.
    pub gundef: Arc<Vec<(usize, u64, u32, u32)>>,
    /// Tensor maps in global memory a TMA used (allocation, offset), for S-TMAP-TAIL.
    pub tmaps_used: Arc<Vec<(usize, u64)>>,
    /// S-TMAP-TAIL: while a tensor map is read, undefined bytes from this offset on are not
    /// a use.
    pub undef_ok_from: Option<usize>,
    /// S-INIT-FILLS (fills.rs): the value unwritten state holds in a multi-fill rerun; None:
    /// 0. Not hashed (constant over a run).
    pub fill: Option<crate::fills::Fill>,
    /// Input coverage (coverage.rs): the provenance shadow of a deterministic run. Not hashed.
    pub prov: Option<Box<crate::coverage::ProvState>>,
    /// Per thread: own clock when its warp last diverged (`Machine::wdiv`); an access with an own
    /// clock at least this one ran after it (intra-warp race labels, E-WARP-ORDER). History
    /// for a label only: not part of the state hash.
    pub div_clk: Arc<Vec<u32>>,
    /// Per warp: it may be diverged on hardware (E-WARP-ORDER): set when the lanes of a split
    /// leave an instruction at different pcs (or a live lane is not in the split), cleared
    /// when a `BSYNC` / `WARPSYNC` releases every live lane together; false at launch. Used
    /// for intra-warp race labels only: not part of the state hash.
    pub wdiv: Arc<Vec<bool>>,
    /// Not hashed: C-ORDER's dead-read probe (wellsync.rs `dead_reads`). Loads by these
    /// (thread, pc) leave their destination registers undefined (S-INIT), so a use of the
    /// loaded value is reported where it reaches a sink.
    pub dead_probe: Option<Arc<BTreeSet<(u32, u32)>>>,
    /// C-TICKET-ADDR reruns (wellsync/perm.rs): an undefined or forced claim result, and
    /// what the rerun records. Not hashed.
    pub perm: Option<Box<crate::wellsync::perm::PermRun>>,
    /// Not hashed, shared by every copy of a machine: the racing reads seen so far, per race
    /// signature (`race_dead.rs`). Some(reads (thread, pc)) while every read of that race is
    /// a thread's non-acquiring load; None once one is not (a write-write race, an atomic,
    /// an asynchronous read).
    pub race_log: Option<RaceLog>,
    /// Not hashed, shared by every copy: relaxed strong reads (and RMWs) whose observed release
    /// a later acquire of the same thread joined, so they synchronized as an acquire would
    /// (a relaxed read or RMW followed by fence.acq_rel / fence.acquire / MEMBAR / CCTL.IVALL).
    /// The certificate's exemptions for non-acquiring accesses (C-ORDER's commutative atomics
    /// and dead reads, C-TICKET, C-POLL-ATOM) and E-RACE-DEAD treat these as acquiring.
    pub fence_acq: Option<FenceAcq>,
    /// Not hashed, transient: set while an operation that read partly undefined registers
    /// runs: the undefined bits of its result (slot i in bits 32i..32i+32), S-INIT.
    pub vmask: Option<u64>,
    /// Not hashed, transient: footprint recording for partial-order reduction (dpor.rs).
    pub fp: Option<Box<Footprint>>,
    /// Not hashed: thread indices outside this range have exited for good (slices: the
    /// other clusters), so scans that only care about live threads skip them (`live`).
    pub live: std::ops::Range<usize>,
    /// Not hashed, fixed within a launch group (groups.rs): accesses of earlier launch groups
    /// a global access must be ordered after (E-LAUNCH-GROUPS).
    pub xg: Option<Arc<XgMap>>,
    /// Not hashed: the launch-group meet this exploration's terminal states go to
    /// (`groups::terminal_sink`; 0: none).
    pub sink: u64,
    /// Not hashed: set on the weak (LDS/STS plain) pass of `smem::two_pass`, whose verdict is
    /// discarded once it finds a race that `smem::needs_strong` sends to the strong pass: the
    /// explorer then stops at the first such race (1: an LDS/STS race; 2: also a global-memory
    /// race, when the strong kernel has acquiring LDS). 0: off.
    pub weak_stop: u8,
    /// Not hashed: the well-synchronization certificate's event log (wellsync.rs); empty
    /// except in a certificate run, and clones are always empty.
    pub wlog: crate::wellsync::LogSlot,
    /// Not hashed: the forced counter reads of a C-RING certificate run (wellsync/ring.rs);
    /// None elsewhere. Clones keep it (dry runs of a poll iteration must see the forcing).
    pub ring: Option<Arc<crate::wellsync::ring::RingCtl>>,
    /// E-HW-L1 (opt-in, `Launch::hw_l1`): per-SM L1 lines and the predictive check's logs
    /// (l1.rs). Hashed (the lines) when present.
    pub l1: Option<Arc<crate::l1::L1>>,
    /// NIC queue pairs' state (nic.rs); None without `Launch::nic` QPs. Hashed.
    pub nic: Option<Box<crate::nic::NicState>>,
}

/// A vector shared copy-on-write in chunks of 64 elements, grouped 64 chunks at a time:
/// cloning copies one pointer per 4096 elements, and a write copies its group's chunk
/// pointers and its chunk. The explorer clones a machine per successor; with tens of
/// thousands of threads a flat `Vec` of thread pointers dominated memory, and a flat list of
/// chunk pointers made every clone bump hundreds of shared reference counts (contended
/// across explorer threads).
#[derive(Clone, Debug)]
pub struct CowVec<T: Clone> {
    groups: Vec<Arc<Vec<Arc<Vec<T>>>>>,
    len: usize,
}

const COWVEC_CHUNK: usize = 64;
const COWVEC_GROUP: usize = 64 * COWVEC_CHUNK;

impl<T: Clone> CowVec<T> {
    pub fn new(v: Vec<T>) -> Self {
        let len = v.len();
        let groups = v
            .chunks(COWVEC_GROUP)
            .map(|g| {
                Arc::new(
                    g.chunks(COWVEC_CHUNK)
                        .map(|c| Arc::new(c.to_vec()))
                        .collect(),
                )
            })
            .collect();
        CowVec { groups, len }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn iter(&self) -> impl Iterator<Item = &T> + Clone {
        self.groups
            .iter()
            .flat_map(|g| g.iter())
            .flat_map(|c| c.iter())
    }
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut T> {
        self.groups
            .iter_mut()
            .flat_map(|g| Arc::make_mut(g).iter_mut())
            .flat_map(|c| Arc::make_mut(c).iter_mut())
    }
}

impl<T: Clone> std::ops::Index<usize> for CowVec<T> {
    type Output = T;
    #[inline]
    fn index(&self, i: usize) -> &T {
        &self.groups[i / COWVEC_GROUP][(i % COWVEC_GROUP) / COWVEC_CHUNK][i % COWVEC_CHUNK]
    }
}

impl<T: Clone> std::ops::IndexMut<usize> for CowVec<T> {
    #[inline]
    fn index_mut(&mut self, i: usize) -> &mut T {
        let g = Arc::make_mut(&mut self.groups[i / COWVEC_GROUP]);
        &mut Arc::make_mut(&mut g[(i % COWVEC_GROUP) / COWVEC_CHUNK])[i % COWVEC_CHUNK]
    }
}

impl<T: Clone + Hash> Hash for CowVec<T> {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.len.hash(h);
        for x in self.iter() {
            x.hash(h);
        }
    }
}

/// Per global byte: (first accessor + 1, flags: 1 = written, 2 = accessed by several actors).
#[derive(Clone, Debug, Default)]
pub struct GOwn {
    pub by: Vec<Vec<(u32, u8)>>,
}

impl GOwn {
    pub fn new(sizes: &[usize]) -> GOwn {
        GOwn {
            by: sizes.iter().map(|&n| vec![(0, 0); n]).collect(),
        }
    }
    pub fn add(&mut self, alloc: usize, off: u64, n: usize, actor: u32, write: bool) {
        let Some(v) = self.by.get_mut(alloc) else {
            return;
        };
        for b in off as usize..(off as usize + n).min(v.len()) {
            let e = &mut v[b];
            if e.0 == 0 {
                e.0 = actor + 1;
            } else if e.0 != actor + 1 {
                e.1 |= 2;
            }
            if write {
                e.1 |= 1;
            }
        }
    }
    /// The first byte written by one actor and accessed by another: (allocation, offset).
    pub fn conflict(&self) -> Option<(usize, usize)> {
        self.by
            .iter()
            .enumerate()
            .find_map(|(a, v)| v.iter().position(|e| e.1 == 3).map(|o| (a, o)))
    }
}

/// Global-memory bytes touched, per allocation: (read bits, write bits). Lock-free: every
/// search thread ORs into the same bitsets.
/// Reduction kinds of `GFoot::red_kind` / `Machine::red_kind`.
pub const RED_F32: u8 = 0;
pub const RED_BF16: u8 = 1;
pub const RED_F16: u8 = 2;

#[derive(Debug)]
pub struct GFoot {
    pub by: Vec<(Vec<AtomicU64>, Vec<AtomicU64>)>,
    /// Bytes updated only by fp32 atomic adds without a returned value (ASSUME(N-ATOMIC)) or
    /// 16-bit bulk-reduction adds (ASSUME(N-RED16)): two clusters' reductions into one byte
    /// commute up to the order the bound covers.
    pub red: Vec<Vec<AtomicU64>>,
    /// The same bytes per reduction kind (`RED_F32`, `RED_BF16`, `RED_F16`): two clusters'
    /// reductions of different kinds into one byte do not commute.
    pub red_kind: Vec<[Vec<AtomicU64>; 3]>,
    /// Record only the accesses of this cluster's actors (a slice of a later grid also runs
    /// the earlier grids, whose accesses the grid boundary orders).
    pub cluster: Option<u32>,
}

impl GFoot {
    /// Bitsets for allocations of the given byte sizes.
    pub fn new(sizes: &[usize]) -> GFoot {
        let bits = |n: usize| (0..n.div_ceil(64)).map(|_| AtomicU64::new(0)).collect();
        GFoot {
            by: sizes.iter().map(|&n| (bits(n), bits(n))).collect(),
            red: sizes.iter().map(|&n| bits(n)).collect(),
            red_kind: sizes.iter().map(|&n| [bits(n), bits(n), bits(n)]).collect(),
            cluster: None,
        }
    }
    pub fn add(&self, alloc: usize, off: u64, n: usize, write: bool) {
        let Some(e) = self.by.get(alloc) else { return };
        let v = if write { &e.1 } else { &e.0 };
        Self::mark(v, off, n);
    }
    /// A commuting reduction of kind `kind` (`RED_F32` / `RED_BF16` / `RED_F16`).
    pub fn add_red(&self, alloc: usize, off: u64, n: usize, kind: u8) {
        if let Some(v) = self.red.get(alloc) {
            Self::mark(v, off, n);
        }
        if let Some(v) = self.red_kind.get(alloc) {
            Self::mark(&v[kind as usize], off, n);
        }
    }
    /// Plain copies of the per-kind reduction bitsets.
    pub fn snapshot_kinds(&self) -> Vec<[Vec<u64>; 3]> {
        let ld = |v: &Vec<AtomicU64>| -> Vec<u64> {
            v.iter()
                .map(|x| x.load(std::sync::atomic::Ordering::Relaxed))
                .collect()
        };
        self.red_kind
            .iter()
            .map(|[a, b, c]| [ld(a), ld(b), ld(c)])
            .collect()
    }
    pub fn mark(v: &[AtomicU64], off: u64, n: usize) {
        let (lo, hi) = (off as usize, off as usize + n);
        let mut b = lo;
        while b < hi {
            let w = b / 64;
            let end = hi.min((w + 1) * 64);
            let len = end - b;
            let mask = if len == 64 {
                !0
            } else {
                ((1u64 << len) - 1) << (b % 64)
            };
            if let Some(x) = v.get(w) {
                x.fetch_or(mask, std::sync::atomic::Ordering::Relaxed);
            }
            b = end;
        }
    }
    /// Plain copies of the bitsets: (read, write, reduction).
    pub fn snapshot(&self) -> Vec<(Vec<u64>, Vec<u64>, Vec<u64>)> {
        let ld = |v: &Vec<AtomicU64>| {
            v.iter()
                .map(|x| x.load(std::sync::atomic::Ordering::Relaxed))
                .collect()
        };
        self.by
            .iter()
            .zip(&self.red)
            .map(|((r, w), x)| (ld(r), ld(w), ld(x)))
            .collect()
    }
}

/// What a search learned for value-insensitive polls (E-POLL-ANY): for every polled global
/// location (allocation, offset, bytes), every value any explored write left there. A new
/// location or value sets `dirty`: the explorer then runs another round, so the last round
/// decides with every value of every execution (`explore_one_model`).
#[derive(Debug, Default)]
pub struct VipLearn {
    pub values: std::sync::Mutex<
        std::collections::HashMap<(u32, u64, u8), std::collections::BTreeSet<u64>>,
    >,
    /// Allocations (mod 64) with a watched location: writes elsewhere skip the lock.
    pub allocs: AtomicU64,
    /// Watched locations some explored write lowered (as an unsigned little-endian integer):
    /// not monotone. A monotone one only takes values at least its current one.
    pub nonmono: std::sync::Mutex<std::collections::HashSet<(u32, u64, u8)>>,
    pub dirty: std::sync::atomic::AtomicBool,
    /// Acquire loads may be value-insensitive polls too (their acquire dropped): the first
    /// search's model. A race found with it is searched again without (`explore::explore_one`).
    pub acquire: bool,
}

impl VipLearn {
    /// The values learned for `key` (watching it from now on, which dirties the round).
    pub fn watch(&self, key: (u32, u64, u8)) -> std::collections::BTreeSet<u64> {
        let mut g = self.values.lock().unwrap();
        g.entry(key)
            .or_insert_with(|| {
                self.allocs
                    .fetch_or(1 << (key.0 % 64), std::sync::atomic::Ordering::Relaxed);
                self.dirty.store(true, std::sync::atomic::Ordering::Relaxed);
                Default::default()
            })
            .clone()
    }
}

/// Space of `SFoot` items for an all-thread `bar.sync` (no count) in a CTA: its completion
/// depends on which threads of the CTA have exited, so it is shared with every slice that has
/// threads in the CTA (written by the barrier, read by `slices::restrict`).
pub const SPACE_CTA_ALL: u8 = 98;

/// The sync cells, shared copy-on-write between a state and its successors: most steps do not
/// publish, so cloning a state for a successor no longer deep-copies every cell's clocks.
#[derive(Clone, Default, Debug)]
pub struct Syncs(Arc<BTreeMap<(u8, u32, u64), SyncCell>>);

impl std::ops::Deref for Syncs {
    type Target = BTreeMap<(u8, u32, u64), SyncCell>;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Syncs {
    pub fn insert(&mut self, k: (u8, u32, u64), c: SyncCell) -> Option<SyncCell> {
        Arc::make_mut(&mut self.0).insert(k, c)
    }
    pub fn remove(&mut self, k: &(u8, u32, u64)) -> Option<SyncCell> {
        Arc::make_mut(&mut self.0).remove(k)
    }
    pub fn get_mut(&mut self, k: &(u8, u32, u64)) -> Option<&mut SyncCell> {
        Arc::make_mut(&mut self.0).get_mut(k)
    }
}

/// Everything a thread slice's transitions touched (E-SLICE, slices.rs): shared memory per
/// CTA as bitsets (read, write), and the other footprint spaces (`Footprint`) as items
/// (space, id, lo, hi, write). Global memory is in `GFoot`. Lock-free for memory; items take
/// a lock, but synchronization steps are rare next to memory accesses.
/// One synchronization footprint item: (space, id, start, end, write).
pub type SFootItem = (u8, u32, u64, u64, bool);

#[derive(Debug)]
pub struct SFoot {
    pub smem: Vec<(Vec<AtomicU64>, Vec<AtomicU64>)>,
    pub items: std::sync::Mutex<std::collections::HashSet<SFootItem>>,
}

impl SFoot {
    /// Bitsets for CTAs with the given shared-memory sizes.
    pub fn new(smem_sizes: &[usize]) -> SFoot {
        let bits = |n: usize| (0..n.div_ceil(64)).map(|_| AtomicU64::new(0)).collect();
        SFoot {
            smem: smem_sizes.iter().map(|&n| (bits(n), bits(n))).collect(),
            items: Default::default(),
        }
    }
    /// One footprint item (`Machine::rec`). Global memory (1) and strong accesses (10, 11,
    /// the same bytes as 0 and 1) are recorded elsewhere or redundant; 4 (CTA launch), 5 (bulk
    /// group of an issuer), 8 (warp-internal) and 9 (grid gates) are not shared between slices
    /// (`slices::restrict` requires every live grid to be launchable on its own).
    pub fn add(&self, space: u8, id: u32, lo: u64, hi: u64, write: bool) {
        match space {
            0 => {
                if let Some(e) = self.smem.get(id as usize) {
                    GFoot::mark(if write { &e.1 } else { &e.0 }, lo, (hi - lo) as usize);
                }
            }
            1 | 4 | 5 | 8 | 9 | 10 | 11 => {}
            _ => {
                // Barrier generations and mbarrier phases: every use conflicts (the
                // co-participants of a barrier must be in one slice).
                let w = write || matches!(space, 14 | 7 | 12);
                self.items.lock().unwrap().insert((space, id, lo, hi, w));
            }
        }
    }
    /// Plain copies: per CTA (read, write) bitsets, and the items.
    #[allow(clippy::type_complexity)]
    pub fn snapshot(&self) -> (Vec<(Vec<u64>, Vec<u64>)>, Vec<(u8, u32, u64, u64, bool)>) {
        let ld = |v: &Vec<AtomicU64>| -> Vec<u64> {
            v.iter()
                .map(|x| x.load(std::sync::atomic::Ordering::Relaxed))
                .collect()
        };
        let mut items: Vec<_> = self.items.lock().unwrap().iter().copied().collect();
        items.sort_unstable();
        (
            self.smem.iter().map(|(r, w)| (ld(r), ld(w))).collect(),
            items,
        )
    }
}

/// Footprint space of the tensor core's A collector buffer (S-TC-COLLECTOR).
pub const SPACE_COLL_A: u8 = 24;
/// Footprint space of mbarrier pending counts (exec.rs `rec_count`).
pub const SPACE_MBAR_COUNT: u8 = 16;
/// Footprint space of mbarrier over-arrivals (exec.rs `rec_late`, `rec_count`); a write is an
/// over-arrival to the reductions (`Machine::arv_over`, dpor.rs `OVER_ARRIVAL`).
pub const SPACE_MBAR_OVER: u8 = 25;
/// Footprint space of NIC queues (nic.rs, id = queue).
pub const SPACE_NIC_QUEUE: u8 = 26;

/// What one transition touched, for the DPOR dependency relation. Spaces: 0 shared memory
/// (id = CTA; mbarrier objects are their 8 bytes: poll = read, arrive/init/complete = write),
/// 1 global memory (id = allocation), 2 TMEM words (id = CTA, index = tmem_idx), 3 TMEM
/// allocator (id = CTA), 4 CTA exit state (id = CTA), 5 bulk-group state (id = issuer),
/// 6 mbarrier phase state (polls read, phase completions and init write), 7 mbarrier
/// arrival-phase keys (see `mbar_maybe_complete`), 8 merge-sensitive pcs (id = warp), 9 grid
/// launch gates, 10/11 strong shared/global accesses, 12 mbarrier phase keys (`rec_phase`),
/// 13 barrier.cluster state, 14 named-barrier generations (id = CTA, index = barrier id << 32
/// | generation: arrivals read theirs, `bar.arrive` writes all; E-BAR-OVER writes, see
/// `Op::BarSync` and `bar_try_release` in exec.rs), 15 mbarrier (re)initialization: init and
/// inval write it (id = CTA, index = offset; sync-only, for the explorer's phase-aware
/// footprints, `sync_ample`), 16 mbarrier pending counts (`SPACE_MBAR_COUNT`, `rec_count`:
/// arrivals read the key of their phase, a counted arrival that takes the last pending
/// arrivals writes it), 20 / 21 L1 lines and SMs (l1_cores.rs `SPACE_LINE` / `SPACE_SM`),
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

/// Key of a synchronization object generation: kind 1 = named barrier (cta, id, generation),
/// 2 = mbarrier (cta, offset, phase), 3 = bulk group (issuer, group).
pub fn sync_key(kind: u8, a: u64, b: u64, c: u64) -> u64 {
    let mut f = FastHasher {
        s: 0x7e57_ab1e_5eed_0000 | kind as u64,
    };
    f.mix(a);
    f.mix(b);
    f.mix(c);
    f.finish()
}

impl Footprint {
    pub fn add(&mut self, space: u8, id: u32, lo: u64, hi: u64, write: bool) {
        if self.sync_only && matches!(space, 0 | 1 | 2 | 6 | 8) {
            return;
        }
        let v = if write { &mut self.w } else { &mut self.r };
        // Coalesce with the previous range when contiguous (the common streaming case).
        if let Some(last) = v.last_mut() {
            if last.0 == space && last.1 == id && lo <= last.3 && hi >= last.2 {
                last.2 = last.2.min(lo);
                last.3 = last.3.max(hi);
                return;
            }
        }
        v.push((space, id, lo, hi));
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
    fn overlaps(a: &[(u8, u32, u64, u64)], b: &[(u8, u32, u64, u64)]) -> bool {
        // Both sorted: merge-style sweep.
        let (mut i, mut j) = (0, 0);
        while i < a.len() && j < b.len() {
            let (x, y) = (a[i], b[j]);
            if (x.0, x.1) < (y.0, y.1) {
                i += 1;
            } else if (x.0, x.1) > (y.0, y.1) {
                j += 1;
            } else if x.3 <= y.2 {
                i += 1;
            } else if y.3 <= x.2 {
                j += 1;
            } else {
                return true;
            }
        }
        false
    }
    /// Dependent: some write of one overlaps a read or write of the other.
    pub fn conflicts(&self, o: &Footprint) -> bool {
        self.all
            || o.all
            || Self::overlaps(&self.w, &o.w)
            || Self::overlaps(&self.w, &o.r)
            || Self::overlaps(&o.w, &self.r)
    }
    pub fn is_empty(&self) -> bool {
        !self.all && self.r.is_empty() && self.w.is_empty()
    }
    /// Does `self` acquire something `o` released?
    pub fn acquires_from(&self, o: &Footprint) -> bool {
        !self.acq.is_empty() && !o.rel.is_empty() && self.acq.iter().any(|k| o.rel.contains(k))
    }
}

impl Hash for Machine {
    fn hash<H: Hasher>(&self, h: &mut H) {
        self.threads.hash(h);
        self.warps.hash(h);
        self.smem.hash(h);
        self.gmem.hash(h);
        self.tmem.hash(h);
        self.mbars.hash(h);
        self.bars.hash(h);
        self.pending.hash(h);
        self.completed.hash(h);
        self.remote_obs.hash(h);
        self.mbar_seen.hash(h);
        self.bar_gen.hash(h);
        self.bar_prev.hash(h);
        self.clusters.hash(h);
        self.remote.hash(h);
        self.triggered.hash(h);
        self.syncs.hash(h);
        self.sc.hash(h);
        if let Some(l1) = &self.l1 {
            l1.hash(h);
        }
        if let Some(n) = &self.nic {
            n.hash(h);
        }
        if !self.vipr.is_empty() {
            self.vipr.hash(h);
        }
        if !self.elect_fn.is_empty() {
            self.elect_fn.hash(h);
        }
        if !self.ctaint.is_empty() {
            self.ctaint.hash(h);
        }
    }
}

impl Machine {
    /// ASSUME(E-CLOCK): is `pc` a clock branch for the threads `members` (clock.rs)?
    pub fn clock_choice_at(&self, pc: u32, members: &[usize]) -> bool {
        self.clock_step == 0
            && (self.clock_choice_pcs.contains(&pc)
                || self.mem_branch_pcs.contains(&pc)
                    && members.iter().any(|&t| self.threads[t].clock_mem))
    }

    /// Record a footprint item if recording (dpor.rs).
    #[inline]
    pub fn rec(&mut self, space: u8, id: u32, lo: u64, hi: u64, write: bool) {
        if let Some(s) = self.sfoot.as_ref() {
            s.add(space, id, lo, hi, write);
        }
        if let Some(f) = self.fp.as_mut() {
            f.add(space, id, lo, hi, write);
        }
    }
    /// Record only for the stateful explorer's synchronization reduction (`sync_only`
    /// footprints): cluster-level dependencies that dpor.rs does not track.
    #[inline]
    pub fn rec_sync(&mut self, space: u8, id: u32, lo: u64, hi: u64, write: bool) {
        if let Some(s) = self.sfoot.as_ref() {
            s.add(space, id, lo, hi, write);
        }
        if let Some(f) = self.fp.as_mut().filter(|f| f.sync_only) {
            f.add(space, id, lo, hi, write);
        }
    }
    #[inline]
    pub fn rec_rel(&mut self, key: u64) {
        if let Some(f) = self.fp.as_mut() {
            f.rel.push(key);
        }
    }
    #[inline]
    pub fn rec_acq(&mut self, key: u64) {
        if let Some(f) = self.fp.as_mut() {
            f.acq.push(key);
        }
    }

    // ASSUME(E-GRID, S-INIT): see ASSUMPTIONS.md
    pub fn new(k: Arc<Kernel>, launch: Arc<Launch>) -> Machine {
        let has_sc = crate::exec::has_sc_fence(&k);
        let coll_used = k.code.iter().any(
            |i| matches!(i.op, Op::Mma { coll, .. } if coll != ptxv_ir::MmaCollector::Discard),
        );
        let smid_used = crate::smid::reads_smid(&k);
        let delay_pcs = Arc::new(crate::clock::delay_loops(&k));
        let ur_window = Arc::new(crate::urwin::windows(&k));
        let clock_loop_pcs = Arc::new(crate::clock::clock_loops(&k));
        let clock_choice_pcs = Arc::new(crate::clock::clock_branches(&k, &delay_pcs));
        let clock_effect_pcs = Arc::new(crate::clock::clock_effects(&k));
        let clock_store_pcs: Arc<std::collections::HashSet<u32>> =
            Arc::new(crate::clock::clock_stores(&k).into_iter().collect());
        let mem_branch_pcs = Arc::new(crate::clock::mem_branches(&k, &delay_pcs));
        let nsmid = crate::smid::nsmid();
        let smid = Arc::new(crate::smid::spread(&launch, nsmid));
        let mut threads = Vec::new();
        let mut warps = Vec::new();
        let nv = k.nregs[File::V as usize].max(k.nregs[File::R as usize]) as usize;
        let np = k.nregs[File::P as usize].max(8) as usize;
        for cta in 0..launch.ctas() {
            let g = launch.g(cta);
            let tpc = g.threads_per_cta();
            let warp0 = warps.len() as u32;
            for _ in 0..tpc.div_ceil(32) {
                warps.push(Cow::new(Warp {
                    cta: cta as u16,
                    ur: vec![0; 255],
                    up: 0,
                    up_def: 0,
                    ur_written: [0; 4],
                    bmask: [0; 16],
                    cp_seq: 0,
                    cp_landed: Vec::new(),
                }));
            }
            for t in 0..tpc {
                let [bx, by, _] = g.block;
                let tid = [t % bx, (t / bx) % by, t / (bx * by)];
                threads.push(Cow::new(Thread {
                    cta: cta as u16,
                    warp: warp0 + t / 32,
                    lane: t % 32,
                    tid,
                    pc: g.entry,
                    st: St::Run,
                    regs: vec![0; nv],
                    preds: vec![0; np.div_ceil(64)],
                    pdef: vec![0; np.div_ceil(64)],
                    vc: VC::new(),
                    tc_vc: VC::new(),
                    pipe_vc: VC::new(),
                    pipe_grp: 0,
                    pipe2_vc: VC::new(),
                    tma_count: 0,
                    cp_epoch: 0,
                    clock: 0,
                    clock_iters: 0,
                    clock_mem: false,
                    pending_ld: vec![],
                    tc_st: 0,
                    tc_ld: 0,
                    tc_ld_unwaited: 0,
                    tc_st_unwaited: 0,
                    ld_pend: vec![],
                    ldtm_pend: vec![],
                    written: vec![0; nv.div_ceil(64)],
                    vund: vec![],
                    fences: vec![],
                    alias_fences: vec![],
                    gfences: vec![],
                    alias_pending: vec![],
                    alias_fenced: false,
                    bulk_group: 0,
                    bulk_sb: Vec::new(),
                    bulk_done: vec![],
                    bulk_vis: VC::new(),
                    acq_bar: None,
                    vis: VC::new(),
                    membar_vc: VC::new(),
                    minit_vc: VC::new(),
                    membar_scope: Scope::Sys,
                    srel_vc: VC::new(),
                    membar_sh: vec![],
                    srel_sh: vec![],
                    vc_sh: vec![],
                    acq_pending_mb: vec![],
                    landed: VC::new(),
                    acq_pending: vec![],
                    acq_view: VC::new(),
                    cl_arrived: None,
                    tmap_fence: Vec::new(),
                    peekwin: Vec::new(),
                    local: Vec::new(),
                    cp_pending: Vec::new(),
                    cp_group: 0,
                    bar_res: None,
                    coll_home: None,
                }));
            }
        }
        let smem: Vec<Mem> = (0..launch.ctas())
            .map(|c| Mem::new(vec![0; launch.g(c).smem_bytes as usize]))
            .collect();
        let gmem = launch
            .allocs
            .iter()
            .map(|a| {
                if a.undef {
                    Mem::new_undef(a.init.clone())
                } else {
                    Mem::new(a.init.clone())
                }
            })
            .collect();
        let tmem: Vec<Tmem> = (0..launch.ctas()).map(|_| Tmem::new()).collect();
        let launch_ctas = launch.ctas();
        let n_grids = launch.grids.len();
        let threads_len = threads.len() as u32;
        let mut m = Machine {
            k,
            launch,
            div_clk: Arc::new(vec![0; threads.len()]),
            wdiv: Arc::new(vec![false; warps.len()]),
            threads: CowVec::new(threads),
            warps: CowVec::new(warps),
            smem: CowVec::new(smem),
            gmem,
            tmem: CowVec::new(tmem),
            mbars: MbarMap::default(),
            bars: BTreeMap::new(),
            pending: Vec::new(),
            completed: Vec::new(),
            remote_obs: Vec::new(),
            mbar_seen: BTreeMap::new(),
            bar_gen: BTreeMap::new(),
            bar_prev: BTreeMap::new(),
            clusters: BTreeMap::new(),
            remote: Vec::new(),
            triggered: vec![0; (launch_ctas as usize).div_ceil(64)],
            done_vcs: vec![None; n_grids],
            atom_lanes_rev: std::env::var("PTXV_ATOM_LANES").is_ok_and(|v| v == "rev"),
            syncs: Default::default(),
            sc: has_sc.then(Default::default),
            polls: Arc::new(Default::default()),
            wpolls: Arc::new(Default::default()),
            stubs: Arc::new(Default::default()),
            peeks: Arc::new(Default::default()),
            bar_cfg: Arc::new(Default::default()),
            elect_high: false,
            clc_eager: false,
            coll_used,
            smid,
            nsmid,
            async_shift: async_shift(threads_len),
            smid_used,
            smid_fixed: false,
            clock_step: crate::exec::clock_step(),
            ur_window,
            delay_pcs,
            clock_loop_pcs,
            clock_choice_pcs,
            clock_effect_pcs,
            clock_store_pcs,
            mem_branch_pcs,
            ctaint: Default::default(),
            forced_pred: None,
            gfoot: None,
            sfoot: None,
            vip: None,
            arv_over: None,
            vipr: Default::default(),
            vip_read: 0,
            red_access: false,
            red_kind: RED_F32,
            red16: Default::default(),
            mc_view: false,
            async_view: None,
            weak_pub: false,
            cur_sh: vec![],
            round16: None,
            gown: None,
            mma_cache: Default::default(),
            findings: Vec::new(),
            tmem_queue_vc: None,
            num: None,
            elect_choice: None,
            elect_any: false,
            elect_fn: Vec::new(),
            elect_lanes: None,
            stale: None,
            dead_probe: None,
            perm: None,
            wundef: 0,
            gundef: Arc::new(Vec::new()),
            tmaps_used: Arc::new(Vec::new()),
            undef_ok_from: None,
            race_log: None,
            fence_acq: None,
            fill: None,
            prov: None,
            vmask: None,
            fp: None,
            live: 0..0,
            xg: None,
            sink: 0,
            weak_stop: 0,
            wlog: Default::default(),
            ring: None,
            l1: None,
            nic: None,
        };
        m.live = 0..m.threads.len();
        if let Some(cfg) = m.launch.nic.as_ref().filter(|c| !c.qps.is_empty()) {
            m.nic = Some(Box::new(crate::nic::NicState::new(cfg)));
        }
        if m.launch.hw_l1 {
            m.l1 = Some(Arc::new(crate::l1::L1::new()));
        }
        // ASSUME(S-RESERVED): the driver initializes the reserved shared memory of a
        // cta_group::2 cubin (EIATTR_TCGEN05_2CTA_USED): words 0x0/0x4, the leader's phase bit
        // at 0x50, and a hidden count-1 mbarrier at 0x48 that ptxas's paired-alloc handshake
        // uses (probes/reserved_smem, probes/mbar_reserved). Everything else starts at 0.
        for cta in 0..m.launch.ctas() {
            if m.k.front != Front::Sass
                || !m.grid_code(cta).iter().any(|i| {
                    matches!(
                        i.op,
                        Op::TmemAllocRaw { pair: true, .. } | Op::Mma { pair: true, .. }
                    )
                })
            {
                continue;
            }
            {
                let (_, rank) = m.launch.cluster_of(cta);
                let mem = &mut m.smem[cta as usize];
                for (off, w) in [
                    (0x0u32, 0x001f_fe00u32),
                    (0x4, 0x7fff_f800),
                    (0x50, (rank % 2 == 0) as u32),
                ] {
                    for (i, b) in w.to_le_bytes().iter().enumerate() {
                        *mem.bytes.get_mut((off as usize) + i) = *b;
                    }
                }
                m.mbars.insert(
                    (cta as u16, 0x48),
                    Mbar {
                        expected: 1,
                        pending: 1,
                        tx: 0,
                        phase: 0,
                        cur: vec![],
                        done: [vec![], vec![]],
                        cur_land: VC::new(),
                        done_land: [VC::new(), VC::new()],
                        init: Epoch {
                            actor: 0,
                            clock: 0,
                            pc: 0,
                            strong: STRONG_SYS,
                            sv: false,
                            view: 0,
                        },
                        atomics: vec![],
                    },
                );
            }
        }
        // ASSUME(E-CLC): a kernel that cancels CTAs starts with every CTA unlaunched; the
        // explorer launches them in any order. ASSUME(E-PDL): so do the CTAs of every grid
        // after the first; they become launchable when the previous grid allows it.
        for t in 0..m.threads.len() {
            let cta = m.threads[t].cta as u32;
            // ASSUME(E-RESIDENCY): under limited residency every CTA starts unlaunched.
            if m.launch.grid_index(cta) > 0
                || m.launch.env.residency.is_some()
                || m.grid_code(cta)
                    .iter()
                    .any(|i| matches!(i.op, Op::ClcTryCancel { .. }))
            {
                m.threads[t].make_mut().st = St::Unlaunched;
            }
        }
        // ASSUME(E-SPIN): poll loops on strong loads block until the iteration exits.
        m.polls = Arc::new(
            crate::spin::poll_loops(&m.k.code, m.k.front)
                .into_iter()
                .filter_map(|(l, r)| r.ok().map(|(h, b)| (l as u32, (h as u32, b as u32))))
                .collect(),
        );
        // ASSUME(E-SPIN): warp-level poll loops block while an iteration is a no-op.
        m.wpolls = Arc::new(
            crate::spin::wpoll_loops(&m.k.code, m.k.front)
                .into_iter()
                .map(|(h, b)| (h as u32, b as u32))
                .collect(),
        );
        // ASSUME(E-SPIN): see ASSUMPTIONS.md
        m.stubs = Arc::new(
            (0..m.k.code.len())
                .filter_map(|w| match crate::spin::retry_shape(&m, w) {
                    Ok(Some((s, w2))) => Some((w as u32, (s as u32, w2 as u32))),
                    _ => None,
                })
                .collect(),
        );
        m.peeks = Arc::new(
            (0..m.k.code.len())
                .filter(|&w| crate::spin::is_peek(&m, w))
                .map(|w| w as u32)
                .collect(),
        );
        m.bar_cfg = Arc::new(crate::explore::bar_cfg(&m));
        // Every thread starts with its own clock at 1, so that "clock 0" means "never".
        for t in 0..m.threads.len() {
            let th = m.threads[t].make_mut();
            vc_tick(&mut th.vc, t as u32);
        }
        m
    }

    /// 128-bit state fingerprint: two differently seeded fast hashes over the (mostly
    /// cached) sub-object hashes.
    /// A deep copy that shares no reference-counted object with `self`, and takes the
    /// static parts (kernel, launch, spin tables, MMA cache) from `like`. Explorer threads
    /// copy the states they take from each other: otherwise every clone, on every thread,
    /// counts references on the same few objects, and the cache lines holding the counts
    /// bounce between cores (the parallel search stopped scaling at ~32 threads).
    pub fn unshared(&self, like: &Machine) -> Machine {
        UNSHARING.set(true);
        let mut m = self.clone();
        UNSHARING.set(false);
        VC_COPIES.with_borrow_mut(|c| c.clear());
        m.k = like.k.clone();
        m.launch = like.launch.clone();
        m.polls = like.polls.clone();
        m.wpolls = like.wpolls.clone();
        m.stubs = like.stubs.clone();
        m.peeks = like.peeks.clone();
        m.bar_cfg = like.bar_cfg.clone();
        m.mma_cache = like.mma_cache.clone();
        m
    }

    /// The threads that may still run, with their indices (see `live`).
    pub fn live_threads(&self) -> impl Iterator<Item = (usize, &Cow<Thread>)> {
        self.live.clone().map(move |i| (i, &self.threads[i]))
    }

    /// This machine, before any step, with the cluster shapes of `launch` (which differs from
    /// the machine's launch only there: `Launch::preferred_shape`). ASSUME(S-CGA-SHAPE)
    pub fn with_cluster_shape(mut self, launch: Arc<Launch>) -> Machine {
        assert_eq!(launch.ctas(), self.launch.ctas());
        self.launch = launch;
        // ASSUME(S-RESERVED): the leader's phase word at 0x50 depends on the rank (the
        // hidden mbarrier at 0x48 marks the CTAs `new` initialized).
        for cta in 0..self.launch.ctas() {
            if self.mbars.contains_key(&(cta as u16, 0x48)) {
                let (_, rank) = self.launch.cluster_of(cta);
                let w = ((rank % 2 == 0) as u32).to_le_bytes();
                let mem = &mut self.smem[cta as usize];
                for (i, b) in w.iter().enumerate() {
                    *mem.bytes.get_mut(0x50 + i) = *b;
                }
            }
        }
        self
    }

    /// `self` with the static parts (kernel, launch, spin tables, MMA cache) of `like`.
    pub fn with_statics(mut self, like: &Machine) -> Machine {
        self.k = like.k.clone();
        self.launch = like.launch.clone();
        self.polls = like.polls.clone();
        self.wpolls = like.wpolls.clone();
        self.stubs = like.stubs.clone();
        self.peeks = like.peeks.clone();
        self.bar_cfg = like.bar_cfg.clone();
        self.mma_cache = like.mma_cache.clone();
        self
    }

    /// A machine with private copies of the static parts, for `unshared`.
    pub fn private_statics(&self) -> Machine {
        let mut m = self.clone();
        m.k = Arc::new((*self.k).clone());
        m.launch = Arc::new((*self.launch).clone());
        m.polls = Arc::new((*self.polls).clone());
        m.wpolls = Arc::new((*self.wpolls).clone());
        m.stubs = Arc::new((*self.stubs).clone());
        m.peeks = Arc::new((*self.peeks).clone());
        m.bar_cfg = Arc::new((*self.bar_cfg).clone());
        m.mma_cache = Default::default();
        m
    }

    pub fn fingerprint(&self) -> u128 {
        let mut a = FastHasher {
            s: 0xa5a5_0f0f_1234_5678,
        };
        self.hash(&mut a);
        let mut b = FastHasher {
            s: 0x5a5a_f0f0_8765_4321,
        };
        0x5a5au16.hash(&mut b);
        self.hash(&mut b);
        ((a.finish() as u128) << 64) | b.finish() as u128
    }

    /// The code of the kernel that CTA `cta` runs.
    pub fn grid_code(&self, cta: u32) -> &[Inst] {
        let gi = self.launch.grid_index(cta);
        let lo = self.launch.grids[gi].entry as usize;
        let hi = self
            .launch
            .grids
            .get(gi + 1)
            .map_or(self.k.code.len(), |g| g.entry as usize);
        &self.k.code[lo..hi]
    }

    pub fn nthreads(&self) -> u32 {
        self.threads.len() as u32
    }

    /// The first async actor id (`async_base`): after every thread's fixed actors.
    pub fn async_base(&self) -> u32 {
        async_base(self.nthreads())
    }

    // Deterministic actor ids: threads 0..n, then per-thread tcgen05.ld/st actors, per-thread
    // tcgen05 pipes, then async operations keyed by (issuer, per-issuer serial).
    pub fn tc_actor_id(&self, t: usize) -> u32 {
        self.nthreads() + t as u32
    }
    pub fn pipe_actor_id(&self, t: usize) -> u32 {
        2 * self.nthreads() + t as u32
    }
    /// The tcgen05 pipe of thread `t` for the `.cta_group` other than the one of its first
    /// MMA-pipe operation (a kernel that mixes them, `Thread::pipe_grp`): a commit tracks
    /// only the operations of its own group (PTX ISA, tcgen05.commit).
    pub fn pipe2_actor_id(&self, t: usize) -> u32 {
        3 * self.nthreads() + t as u32
    }
    /// The tensor-memory writes of thread `t`'s SASS `STTM`s (MISS 107): complete only at the
    /// thread's `FENCE.VIEW.ASYNC.T` (tcgen05.wait::st), so their own actor, apart from its
    /// LDTM reads (`tc_actor_id`, completed at scoreboard waits).
    pub fn sttm_actor_id(&self, t: usize) -> u32 {
        4 * self.nthreads() + t as u32
    }
    /// The async actor ids of a thread (a power of two, `async_shift`); the last is its
    /// cp.async wait actor.
    pub fn async_ids(&self) -> u32 {
        1 << self.async_shift
    }
    pub fn tma_actor_id(&self, t: usize, serial: u32) -> u32 {
        debug_assert!(serial < self.async_ids());
        self.async_base() + ((t as u32) << self.async_shift) + serial
    }
    /// The (issuing thread, serial) of async actor `a` (at least `async_base`, not a NIC's own):
    /// `async_split_of`.
    pub fn async_split(&self, a: u32) -> (usize, u32) {
        async_split_of(a, self.nthreads(), self.async_shift)
    }
    /// The async actor that lands thread `t`'s cp.async copies retired by its waits, at
    /// increasing epochs (`Thread::cp_epoch`): the last of its async actor ids, which the
    /// others (`async_actor`) stop short of (E-CPASYNC).
    pub fn cp_wait_actor(&self, t: usize) -> u32 {
        self.tma_actor_id(t, self.async_ids() - 1)
    }
    /// Thread `t`'s next async actor: one checked allocator for every kind (TMA, bulk and
    /// remote operations, try_cancel, NIC work requests, the actors landing cp.async copies at
    /// an arrive or at the exit). Past the thread's ids an id would alias another actor's (the
    /// wait actor, then the next thread's ids or past the last thread), so the step fails
    /// closed: an `unmodeled` fault that stops the thread. Never wraps. (MISS 68)
    pub fn async_actor(&mut self, t: usize) -> Result<u32, String> {
        let ids = self.async_ids();
        let serial = self.threads[t].tma_count;
        if serial + 1 >= ids {
            let msg = format!(
                "{} starts its asynchronous operation number {}: a thread of this launch ({} threads) has {} async actor ids (the last for its cp.async waits), and another would alias another actor's: not modeled",
                self.actor_name(t as u32),
                serial + 1,
                self.nthreads(),
                ids
            );
            return Err(msg);
        }
        self.threads[t].make_mut().tma_count += 1;
        Ok(self.tma_actor_id(t, serial))
    }

    /// "CTA 3", or "kernel 1 CTA 0" in a multi-kernel launch.
    pub fn cta_name(&self, cta: u16) -> String {
        if self.launch.grids.len() > 1 {
            let gi = self.launch.grid_index(cta as u32);
            format!(
                "kernel {gi} CTA {}",
                cta as u32 - self.launch.grids[gi].cta0
            )
        } else {
            format!("CTA {cta}")
        }
    }

    fn thread_name(&self, t: u32) -> String {
        let th = &self.threads[t as usize];
        if self.launch.grids.len() > 1 {
            let gi = self.launch.grid_index(th.cta as u32);
            let c = th.cta as u32 - self.launch.grids[gi].cta0;
            format!("k{gi}.cta{c}.t{}", th.tid[0])
        } else if self.launch.ctas() > 1 {
            format!("cta{}.t{}", th.cta, th.tid[0])
        } else {
            format!("t{}", th.tid[0])
        }
    }

    pub fn actor_name(&self, a: u32) -> String {
        let n = self.nthreads();
        if a < self.async_base() {
            let t = self.thread_name(a % n);
            match a / n {
                0 => t,
                1 => format!("{t}'s tcgen05.ld/st"),
                2 => format!("{t}'s tcgen05 pipe"),
                3 => format!("{t}'s tcgen05 pipe (its other .cta_group)"),
                4 => format!("{t}'s STTM writes"),
                k => format!("{t}'s fixed actor {k}"),
            }
        } else if let Some(s) = self.nic_actor_name(a) {
            s
        } else {
            let (t, serial) = self.async_split(a);
            if serial == self.async_ids() - 1 {
                return format!("{}'s cp.async copies", self.thread_name(t as u32));
            }
            format!("TMA#{serial} (issued by {})", self.thread_name(t as u32))
        }
    }

    pub fn report(&mut self, kind: &str, sig: String, msg: String) {
        self.report_pcs(kind, sig, msg, vec![]);
    }

    /// `report` naming the instructions involved (`Finding::pcs`).
    pub fn report_pcs(&mut self, kind: &str, sig: String, msg: String, pcs: Vec<u32>) {
        // Verdicts keep the first finding per signature; so does a step. A step that repeats
        // an access (a racy loop of invisible instructions) would otherwise grow without bound.
        if self.findings.iter().any(|f| f.sig == sig) {
            return;
        }
        self.findings.push(Finding {
            kind: kind.into(),
            sig,
            msg,
            trace: vec![],
            pcs,
            gb300: None,
            perm_rows: vec![],
        });
    }

    /// Output buffers' bytes, concatenated (for determinism and hardware comparison).
    pub fn outputs(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for &i in &self.launch.outputs {
            let m = &self.gmem[i];
            for b in 0..m.bytes.len() {
                out.push(*m.bytes.get(b));
            }
        }
        out
    }
}

#[cfg(test)]
mod vc_tests {
    use super::*;

    /// `vc_join` before its covered fast paths (the reference): every path that is not one of
    /// the cheap early returns rebuilds the base by merging.
    fn join_ref(a: &mut VC, b: &VC) {
        if b.base.e.is_empty() && b.own.1 == 0 {
            return;
        }
        if Arc::ptr_eq(&a.base, &b.base) && (b.own.1 == 0 || vc_get(a, b.own.0) >= b.own.1) {
            return;
        }
        if a.base.e.is_empty() && b.own.1 == 0 {
            a.base = b.base.clone();
            return;
        }
        let mut m: BTreeMap<u32, u32> = a.base.e.iter().copied().collect();
        for (k, c) in b.entries() {
            let e = m.entry(k).or_insert(c);
            *e = (*e).max(c);
        }
        a.base = VcBase::from(m.into_iter().collect());
    }

    /// `vc_join` gives exactly the base and own entry the reference gives, whichever side covers
    /// the other, with and without own entries and shared bases; `vc_unify` makes both clocks
    /// the join of the two; `vc_join_memo` gives what the joins in turn give.
    #[test]
    fn join_fast_paths_equal_the_reference() {
        let r = std::cell::Cell::new(0x9e37_79b9_7f4a_7c15u64);
        let next = |n: u64| {
            let mut x = r.get();
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            r.set(x);
            x % n
        };
        for _ in 0..20_000 {
            let mk = |len: u64| {
                let mut e: Vec<(u32, u32)> = (0..len)
                    .map(|_| (next(40) as u32, 1 + next(5) as u32))
                    .collect();
                e.sort();
                e.dedup_by_key(|x| x.0);
                e
            };
            let (la, lb) = (next(40), next(40));
            let ea = mk(la);
            // b: a's entries raised or lowered here and there, or independent.
            let eb = if next(2) == 0 {
                ea.iter()
                    .filter(|_| next(8) != 0)
                    .map(|&(k, c)| (k, (c as i64 + next(3) as i64 - 1).max(1) as u32))
                    .collect()
            } else {
                mk(lb)
            };
            let mut a = vc_from_sorted(ea);
            let mut b = if next(3) == 0 {
                a.clone()
            } else {
                vc_from_sorted(eb)
            };
            if next(2) == 0 {
                a.own = (next(40) as u32, 1 + next(6) as u32);
            }
            if next(2) == 0 {
                b.own = (next(40) as u32, 1 + next(6) as u32);
            }
            let (a0, b0) = (a.clone(), b.clone());
            let mut want = a.clone();
            join_ref(&mut want, &b);
            vc_join(&mut a, &b);
            assert_eq!(a.base.e, want.base.e, "{a0:?} join {b0:?}");
            assert_eq!((a.own, a.base.h), (want.own, want.base.h));
            // vc_unify: both become the join of the two (logically).
            let (mut x, mut y) = (a0.clone(), b0.clone());
            vc_unify(&mut x, &mut y);
            let mut both = a0.clone();
            vc_join(&mut both, &b0);
            vc_join(&mut both, &a0);
            let mut back = b0.clone();
            vc_join(&mut back, &a0);
            vc_join(&mut back, &both);
            assert!(
                x == both && y == both && back == both,
                "{a0:?} unify {b0:?}"
            );
            // vc_join_memo: the joins in turn, from the memo or not (asked twice).
            for _ in 0..2 {
                let m = vc_join_memo(&a0, &[&b0, &a]);
                let mut j = a0.clone();
                vc_join(&mut j, &b0);
                vc_join(&mut j, &a);
                assert_eq!((&m.base.e, m.own), (&j.base.e, j.own));
            }
        }
    }

    /// A clock's representation is well formed: base entries sorted by actor, distinct, no zero
    /// clocks, its hash the fold of its entries; the own entry is (0, 0) or has a clock.
    fn well_formed(v: &VC) -> bool {
        let e = &v.base.e;
        e.windows(2).all(|w| w[0].0 < w[1].0)
            && e.iter().all(|x| x.1 != 0)
            && v.base.h == e.iter().fold(0, |h, &(a, c)| h ^ vc_mix(a, c))
            && (v.own.1 != 0 || v.own.0 == 0)
    }

    fn model_join(a: &mut BTreeMap<u32, u32>, b: &BTreeMap<u32, u32>) {
        for (&k, &c) in b {
            let e = a.entry(k).or_insert(0);
            *e = (*e).max(c);
        }
    }

    /// Model-based: random sequences of tick, copy (shared representations), join, join_many,
    /// join_all, unify, memoized join and meet over a few clocks, each checked after every step
    /// against a plain map from actor to clock: entries, `vc_get`, `vc_covers`, `vc_tail`,
    /// equality and hash (equal clocks compare and hash equal whatever their representation,
    /// different clocks compare unequal). Few actors and frequent copies make the shared-base
    /// and covered fast paths of `vc_join` common, including a joined clock whose own entry is
    /// the only thing it adds (`Cover2::Right` must not take its base then).
    #[test]
    fn clocks_match_a_map_model() {
        let r = std::cell::Cell::new(0x2545_f491_4f6c_dd1du64);
        let next = |n: u64| {
            let mut x = r.get();
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            r.set(x);
            x % n
        };
        const N: usize = 6;
        const ACTORS: u64 = 10;
        for _ in 0..400 {
            let mut v: Vec<VC> = (0..N).map(|_| VC::new()).collect();
            let mut m: Vec<BTreeMap<u32, u32>> = vec![BTreeMap::new(); N];
            for _ in 0..200 {
                let (i, j) = (next(N as u64) as usize, next(N as u64) as usize);
                let op = next(9);
                match op {
                    0 | 1 => {
                        let a = next(ACTORS) as u32;
                        vc_tick(&mut v[i], a);
                        *m[i].entry(a).or_insert(0) += 1;
                    }
                    2 => {
                        v[i] = v[j].clone();
                        m[i] = m[j].clone();
                    }
                    3 => {
                        let b = v[j].clone();
                        vc_join(&mut v[i], &b);
                        let mb = m[j].clone();
                        model_join(&mut m[i], &mb);
                    }
                    4 => {
                        let ks: Vec<usize> =
                            (0..1 + next(4)).map(|_| next(N as u64) as usize).collect();
                        let vs: Vec<VC> = ks.iter().map(|&k| v[k].clone()).collect();
                        let refs: Vec<&VC> = vs.iter().collect();
                        vc_join_many(&mut v[i], &refs);
                        for &k in &ks {
                            let mk = m[k].clone();
                            model_join(&mut m[i], &mk);
                        }
                    }
                    5 => {
                        let ks: Vec<usize> =
                            (0..1 + next(4)).map(|_| next(N as u64) as usize).collect();
                        v[i] = vc_join_all(ks.iter().map(|&k| &v[k]).collect::<Vec<_>>());
                        let mut j = BTreeMap::new();
                        for &k in &ks {
                            model_join(&mut j, &m[k]);
                        }
                        m[i] = j;
                    }
                    6 => {
                        if i != j {
                            let (lo, hi) = (i.min(j), i.max(j));
                            let (x, y) = v.split_at_mut(hi);
                            vc_unify(&mut x[lo], &mut y[0]);
                            let mb = m[j].clone();
                            model_join(&mut m[i], &mb);
                            m[j] = m[i].clone();
                            assert!(vc_unified(&v[i], &v[j]));
                        }
                    }
                    7 => {
                        let ks: Vec<usize> =
                            (0..1 + next(3)).map(|_| next(N as u64) as usize).collect();
                        let xs: Vec<VC> = ks.iter().map(|&k| v[k].clone()).collect();
                        let refs: Vec<&VC> = xs.iter().collect();
                        // Asked twice: computed, then (usually) from the memo.
                        let a = v[i].clone();
                        let r1 = vc_join_memo(&a, &refs);
                        let r2 = vc_join_memo(&a, &refs);
                        assert!(well_formed(&r2) && r1 == r2);
                        v[i] = r2;
                        for &k in &ks {
                            let mk = m[k].clone();
                            model_join(&mut m[i], &mk);
                        }
                    }
                    _ => {
                        v[i] = vc_meet(&v[i], &v[j]);
                        let mj = m[j].clone();
                        let mi = std::mem::take(&mut m[i]);
                        m[i] = mi
                            .into_iter()
                            .filter_map(|(k, c)| {
                                let c = c.min(*mj.get(&k).unwrap_or(&0));
                                (c > 0).then_some((k, c))
                            })
                            .collect();
                    }
                }
                for k in 0..N {
                    let want: Vec<(u32, u32)> = m[k].iter().map(|(&a, &c)| (a, c)).collect();
                    assert!(well_formed(&v[k]), "op {op}: {:?}", v[k]);
                    assert_eq!(vc_entries(&v[k]), want, "op {op} on {i}, {j}: {:?}", v[k]);
                    for a in 0..ACTORS as u32 + 1 {
                        assert_eq!(vc_get(&v[k], a), *m[k].get(&a).unwrap_or(&0));
                    }
                    let from = next(ACTORS + 1) as u32;
                    let tail: Vec<(u32, u32)> =
                        want.iter().copied().filter(|x| x.0 >= from).collect();
                    assert_eq!(vc_tail(&v[k], from), tail);
                }
                for x in 0..N {
                    for y in 0..N {
                        let same = m[x] == m[y];
                        assert_eq!(v[x] == v[y], same, "{:?} vs {:?}", v[x], v[y]);
                        if same {
                            assert_eq!(v[x].logical_hash(), v[y].logical_hash());
                        }
                        let covers = m[y]
                            .iter()
                            .all(|(a, &c)| m[x].get(a).is_some_and(|&d| d >= c));
                        assert_eq!(vc_covers(&v[x], &v[y]), covers);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod footprint_space_tests {
    /// Named footprint spaces differ from each other and from every literal space id passed to
    /// `rec` in the simulator's sources (a shared id makes unrelated transitions conflict, and
    /// a write in `SPACE_MBAR_OVER` stops DPOR).
    #[test]
    fn footprint_spaces_are_disjoint() {
        let named = [
            ("SPACE_MBAR_COUNT", super::SPACE_MBAR_COUNT),
            ("SPACE_MBAR_OVER", super::SPACE_MBAR_OVER),
            ("SPACE_NIC_QUEUE", super::SPACE_NIC_QUEUE),
            ("SPACE_COLL_A", super::SPACE_COLL_A),
            ("SPACE_LINE", crate::l1_cores::SPACE_LINE),
            ("SPACE_SM", crate::l1_cores::SPACE_SM),
        ];
        for (i, a) in named.iter().enumerate() {
            for b in &named[i + 1..] {
                assert_ne!(
                    a.1, b.1,
                    "{} and {} share footprint space {}",
                    a.0, b.0, a.1
                );
            }
        }
        let srcs = [
            include_str!("exec.rs"),
            include_str!("nic.rs"),
            include_str!("l1_cores.rs"),
            include_str!("machine.rs"),
            include_str!("dpor.rs"),
            include_str!("explore.rs"),
        ];
        let mut literal = std::collections::BTreeSet::new();
        for s in srcs {
            for (i, _) in s.match_indices("self.rec(") {
                let rest = &s[i + "self.rec(".len()..];
                let digits: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                if !digits.is_empty() && rest[digits.len()..].starts_with(',') {
                    literal.insert(digits.parse::<u8>().unwrap());
                }
            }
        }
        assert!(literal.contains(&0), "no literal spaces found: {literal:?}");
        for (name, v) in named {
            assert!(
                !literal.contains(&v),
                "{name} = {v} is also used as a literal footprint space"
            );
        }
    }
}
