/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Launch groups (`ptxv check --groups`): a multi-launch case checked one group of launches
//! at a time. ASSUME(E-LAUNCH-GROUPS): see ASSUMPTIONS.md.
//!
//! The launches are partitioned into groups G1 < G2 < ... < Gn, closed under stream order (a
//! launch's predecessor on its GPU is in its group or an earlier one; no PDL edge crosses
//! groups). Group i is checked on the machine that group i-1's check ended in, with the
//! launches of later groups held back (their threads count as exited, as in a slice), by the
//! certificate (E-WELLSYNC) or, failing that, by exploration. Before group i+1 starts, the
//! race shadow of global memory is *demoted*: every access recorded so far counts as plain
//! (S-SCOPE's morally-strong exemption no longer applies to it), and global words that
//! group i wrote with unordered strong writes (commutative atomics, C-FADD) are *poisoned*:
//! any later access to them is a race. So a later group's access to a byte an earlier group
//! accessed is a race unless every conflicting earlier access happens-before it.
//!
//! A certified group's final state is the state of every schedule of it (E-WELLSYNC), with
//! clocks that are lower bounds; groups after an explored one would need one continuation
//! per terminal state, which is not implemented: an explored group must be the last.

use std::collections::BTreeSet;
use std::time::Instant;

use crate::explore::Verdict;
use crate::machine::*;
use crate::wellsync::Carry;

/// Check stream-order closure; returns why the partition is not valid.
pub fn validate(m: &Machine, groups: &[Vec<usize>]) -> Result<(), String> {
    let n = m.launch.grids.len();
    let mut of = vec![usize::MAX; n];
    for (i, g) in groups.iter().enumerate() {
        for &x in g {
            if x >= n {
                return Err(format!("launch {x} does not exist ({n} launches)"));
            }
            if of[x] != usize::MAX {
                return Err(format!("launch {x} is in two groups"));
            }
            of[x] = i;
        }
    }
    if let Some(x) = of.iter().position(|&g| g == usize::MAX) {
        return Err(format!("launch {x} is in no group"));
    }
    for x in 0..n {
        if let Some(p) = m.launch.prev_grid(x) {
            if of[p] > of[x] {
                return Err(format!(
                    "launch {x} is in group {} but its stream predecessor {p} in the later group {}",
                    of[x] + 1,
                    of[p] + 1
                ));
            }
            if of[p] < of[x] && m.launch.grids[x].pdl {
                return Err(format!(
                    "launch {x} depends on {p} by programmatic dependent launch across groups"
                ));
            }
        }
    }
    Ok(())
}

/// `m` with the threads of every launch not in `keep` exited.
fn hold(mut m: Machine, keep: &BTreeSet<usize>) -> Machine {
    for g in 0..m.launch.grids.len() {
        if keep.contains(&g) {
            continue;
        }
        for t in m.grid_threads(g) {
            m.threads[t].make_mut().st = St::Exited;
        }
    }
    relive(&mut m);
    m
}

fn relive(m: &mut Machine) {
    let live: Vec<usize> = (0..m.threads.len())
        .filter(|&t| m.threads[t].st != St::Exited)
        .collect();
    m.live = match (live.first(), live.last()) {
        (Some(&a), Some(&b)) => a..b + 1,
        _ => 0..0,
    };
}

/// An obligation on later groups: a global byte range, an access of an earlier group, and
/// whether it was a write (then every later access must happen after it; else a later
/// write must).
pub type Oblig = (u32, u64, u64, Epoch, bool);

fn add_oblig(map: &mut XgMap, (alloc, lo, hi, e, w): Oblig) {
    for b in lo..hi {
        let v = map.entry((alloc, b)).or_default();
        match v.iter_mut().find(|(x, xw)| x.actor == e.actor && *xw == w) {
            // An actor's accesses are in program order: the latest covers the earlier ones.
            Some((x, _)) => {
                if x.clock < e.clock {
                    *x = e;
                }
            }
            None => v.push((e, w)),
        }
    }
}

/// Obligations from a certified group's `Carry`: its strong reads (a later write must follow
/// them: the race shadow dropped those before a byte's last write, and a strong read, unlike
/// a plain one, need not be ordered before that write) and its strong writes that another
/// strong write of the same word does not order (in another schedule either can be the last).
fn carry_obligs(c: &Carry) -> Vec<Oblig> {
    let mut v: Vec<Oblig> = Vec::new();
    for &(a, lo, hi, e) in &c.reads {
        v.push((a as u32, lo, hi, e, false));
    }
    for &(a, lo, hi, e) in &c.poison {
        v.push((a as u32, lo, hi, e, true));
    }
    v
}

/// The machine for the next group: `fin` (the end state of the groups so far) with the
/// launches of `next` back in their initial state (from `full`), the global race shadow
/// demoted to plain accesses, and `obligs` added to the earlier groups' (`Machine::xg`).
pub fn boundary(fin: Machine, full: &Machine, next: &[usize], obligs: &[Oblig]) -> Machine {
    // The explorer edits static tables of its states (`peekwait`, `peekskip` in `bar_cfg`)
    // and the certificate may have run the weak kernel: back to the case's own.
    let mut fin = fin.with_statics(full);
    fin.sink = 0;
    fin.weak_stop = 0;
    fold_completed(&mut fin);
    fin.wlog = Default::default();
    fin.ring = None;
    fin.findings.clear();
    fin.fp = None;
    fin.num = None;
    fin.elect_high = full.elect_high;
    fin.elect_any = full.elect_any;
    fin.elect_fn = full.elect_fn.clone();
    fin.elect_lanes = full.elect_lanes.clone();
    fin.elect_choice = full.elect_choice;
    fin.stale = full.stale;
    fin.vmask = full.vmask;
    for &g in next {
        for t in full.grid_threads(g) {
            fin.threads[t] = full.threads[t].clone();
        }
    }
    relive(&mut fin);
    for mem in fin.gmem.iter_mut() {
        let n = mem.sh.len();
        let mut i = 0;
        while i < n {
            i += mem.sh.for_run(i, n - i, |_, s| s.demote());
        }
    }
    let mut map: XgMap = fin.xg.as_deref().cloned().unwrap_or_default();
    // E-POLL-ANY reads of the groups so far: obligations like strong reads.
    for (&(a, b), v) in fin.vipr.iter() {
        for e in v {
            add_oblig(&mut map, (a, b, b + 1, Epoch { strong: 0, ..*e }, false));
        }
    }
    fin.vipr = Default::default();
    fin.vip = None;
    for &o in obligs {
        let (alloc, lo, hi, e, w) = o;
        let hi = hi.min(fin.gmem[alloc as usize].sh.len() as u64);
        add_oblig(&mut map, (alloc, lo, hi, Epoch { strong: 0, ..e }, w));
    }
    fin.xg = (!map.is_empty()).then(|| std::sync::Arc::new(map));
    fin
}

/// Completed async operations of finished CTAs (every thread exited) become part of their
/// threads' clocks (what `grid_done_vc` joins for the next grid on the GPU) and leave
/// `completed`: their exit checks were done at the group's end, and the clocks of exited
/// threads are read only by `grid_done_vc` (and a meet must not depend on which operation an
/// async unit's clock value names in one terminal state or another).
fn fold_completed(m: &mut Machine) {
    let done = |m: &Machine, cta: u16| m.threads.iter().all(|t| t.cta != cta || t.st == St::Exited);
    let recs = std::mem::take(&mut m.completed);
    let mut keep = Vec::new();
    for r in recs {
        let (cta, actor, clock, _) = r;
        if !done(m, cta) {
            keep.push(r);
            continue;
        }
        if let Some(t) = (0..m.threads.len()).find(|&t| m.threads[t].cta == cta) {
            let th = m.threads[t].make_mut();
            vc_join(&mut th.vc, &vc_single(actor, clock));
        }
    }
    m.completed = keep;
}

/// The meet of an explored group's terminal states (`terminal_sink`): one state whose clocks
/// are lower bounds of every terminal state's (pointwise minimum per thread, completed async
/// operations and synchronization sources kept only where all agree), with the global race
/// shadow of the first and, for every byte whose shadow differs between terminal states,
/// every terminal state's last write and reads as obligations.
#[derive(Default)]
pub struct Meet {
    pub(crate) base: Option<Machine>,
    /// The fingerprint of the terminal state `base` started from: the base is the terminal
    /// state with the least fingerprint, so that the meet does not depend on the order in
    /// which terminal states arrive (multi-node: every rank computes the same one).
    pub(crate) base_fp: u128,
    pub(crate) obligs: XgMap,
    /// Allocations whose final bytes differ between terminal states.
    pub(crate) differ: BTreeSet<usize>,
    /// Per actor of a finished CTA (thread or async unit): its CTA and the largest final
    /// clock over the terminal states (`finish`).
    pub(crate) finals: std::collections::HashMap<u32, (u16, u32)>,
    pub count: usize,
}

/// Active meets by id (`Machine::sink`): explorations of other cases in the same process
/// (parallel tests) do not mix in.
static SINK: std::sync::Mutex<Option<std::collections::HashMap<u64, Meet>>> =
    std::sync::Mutex::new(None);
static SINK_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Hook (explore.rs, at every terminal state): fold it into its `Meet`, if any.
pub fn terminal_sink(m: &Machine) {
    if m.sink == 0 {
        return;
    }
    let mut g = SINK.lock().unwrap();
    if let Some(meet) = g.as_mut().and_then(|h| h.get_mut(&m.sink)) {
        meet.add(m);
    }
}

fn sh_obligs(alloc: u32, i: u64, s: &Sh, map: &mut XgMap) {
    for w in s.writes() {
        add_oblig(map, (alloc, i, i + 1, Epoch { strong: 0, ..*w }, true));
    }
    for r in &s.r {
        add_oblig(map, (alloc, i, i + 1, Epoch { strong: 0, ..*r }, false));
    }
}

impl Meet {
    fn add(&mut self, m: &Machine) {
        self.count += 1;
        let mut m = m.clone();
        // E-POLL-ANY: reads of value-insensitive polls are strong reads a later write must
        // follow, in every terminal state.
        for (&(a, b), v) in m.vipr.iter() {
            for e in v {
                add_oblig(
                    &mut self.obligs,
                    (a, b, b + 1, Epoch { strong: 0, ..*e }, false),
                );
            }
        }
        m.vipr = Default::default();
        let done =
            |m: &Machine, cta: u16| m.threads.iter().all(|t| t.cta != cta || t.st == St::Exited);
        let mut fin = |a: u32, cta: u16, c: u32| {
            let e = self.finals.entry(a).or_insert((cta, c));
            e.1 = e.1.max(c);
        };
        for t in 0..m.threads.len() {
            let th = &m.threads[t];
            if th.st == St::Exited {
                fin(t as u32, th.cta, vc_get(&th.vc, t as u32));
            }
        }
        for &(cta, actor, clock, _) in &m.completed {
            if done(&m, cta) {
                fin(actor, cta, clock);
            }
        }
        fold_completed(&mut m);
        let fp = m.fingerprint();
        let Some(base) = self.base.as_mut() else {
            self.base = Some(m);
            self.base_fp = fp;
            return;
        };
        // Everything below is symmetric in `base` and `m` except which machine's own fields
        // (registers, shadow) the result keeps: keep the least fingerprint's.
        let m = if fp < self.base_fp {
            self.base_fp = fp;
            std::mem::replace(base, m)
        } else {
            m
        };
        let m = &m;
        for (a, (x, y)) in base.gmem.iter().zip(&m.gmem).enumerate() {
            let per = Pages::<u8>::PAGE;
            for pi in 0..x.bytes.npages() {
                if !x.bytes.same_page(&y.bytes, pi) {
                    let lo = pi * per;
                    let hi = (lo + per).min(x.bytes.len());
                    if (lo..hi).any(|i| x.bytes.get(i) != y.bytes.get(i)) {
                        self.differ.insert(a);
                    }
                }
                if !x.sh.same_page(&y.sh, pi) {
                    let lo = pi * per;
                    let hi = (lo + per).min(x.sh.len());
                    for i in lo..hi {
                        let (p, q) = (x.sh.get(i), y.sh.get(i));
                        if p != q {
                            sh_obligs(a as u32, i as u64, p, &mut self.obligs);
                            sh_obligs(a as u32, i as u64, q, &mut self.obligs);
                        }
                    }
                }
            }
        }
        for t in 0..base.threads.len() {
            if base.threads[t].vc != m.threads[t].vc {
                let v = vc_meet(&base.threads[t].vc, &m.threads[t].vc);
                base.threads[t].make_mut().vc = v;
            }
        }

        let keys: Vec<_> = base.syncs.keys().copied().collect();
        for k in keys {
            let keep = match (base.syncs.get(&k), m.syncs.get(&k)) {
                (Some(c), Some(d))
                    if (c.writer, c.scope, c.pc, c.rel_scope)
                        == (d.writer, d.scope, d.pc, d.rel_scope) =>
                {
                    Some(vc_meet(&c.rel, &d.rel))
                }
                _ => None,
            };
            match keep {
                Some(r) => base.syncs.get_mut(&k).unwrap().rel = r,
                None => {
                    base.syncs.remove(&k);
                }
            }
        }
    }
}

impl Meet {
    /// The continuation state. An actor of a finished CTA is known, through the grid
    /// completion its CTA's next grid joins (`grid_done_vc`), at its largest final clock
    /// over the terminal states: knowing an actor's *final* clock means knowing all of its
    /// events, whatever their numbering in one terminal state or another (the number of
    /// steps of a thread, or of bulk copies of an async unit, can depend on the schedule).
    /// Partial knowledge (other threads' clocks, synchronization sources) stays the minimum.
    fn finish(mut self) -> Option<Machine> {
        let mut base = self.base.take()?;
        for (&a, &(cta, c)) in &self.finals {
            if let Some(t) = (0..base.threads.len()).find(|&t| base.threads[t].cta == cta) {
                let th = base.threads[t].make_mut();
                vc_join(&mut th.vc, &vc_single(a, c));
            }
        }
        Some(base)
    }
}

/// The continuation of a group whose certificate took several C-RING runs (one per sequence
/// of counter choices, wellsync/ring.rs): their meet, as for an explored group's terminal
/// states, with every run's `Carry` and the meet's obligations as one `Carry`. Err if the
/// runs leave different global memory (later groups would need one continuation each).
pub(crate) fn meet_runs(runs: Vec<(Machine, Carry)>) -> Result<(Machine, Carry), String> {
    let mut meet = Meet::default();
    let mut carry = Carry::default();
    for (m, c) in runs {
        meet.add(&m);
        carry.poison.extend(c.poison);
        carry.reads.extend(c.reads);
    }
    if !meet.differ.is_empty() {
        return Err(format!(
            "the C-RING runs leave different bytes in allocation(s) {:?}",
            meet.differ
        ));
    }
    for (&(a, i), v) in &meet.obligs {
        for &(e, w) in v {
            let x = (a as usize, i, i + 1, e);
            if w {
                carry.poison.push(x);
            } else {
                carry.reads.push(x);
            }
        }
    }
    let m = meet.finish().ok_or("no run")?;
    Ok((m, carry))
}

impl Meet {
    /// Thread slices (E-SLICE, slices.rs): the meet of every combination of the slices'
    /// terminal states, from `t0` (one terminal state of the whole group, a deterministic run),
    /// `cur` (the group's start) and each slice's meet. A thread's clock in a combination is the
    /// join of its clocks in the slices' terminal states (its own slice's, plus the async
    /// completions another slice folded into it as the first thread of a shared CTA), so the
    /// join of the slices' lower bounds is a lower bound. Final clocks and obligations are the
    /// slices' (disjoint actors and bytes). A synchronization source is the one slice's that
    /// changed it (strong writes of one location are in one slice), else unchanged.
    pub(crate) fn compose(cur: &Machine, t0: &Machine, parts: Vec<Meet>) -> Meet {
        let mut out = Meet::default();
        out.add(t0);
        out.count = parts.iter().fold(1usize, |n, p| n.saturating_mul(p.count));
        let Some(base) = out.base.as_mut() else {
            return out;
        };
        let bases: Vec<&Machine> = parts.iter().filter_map(|p| p.base.as_ref()).collect();
        for t in 0..base.threads.len() {
            let mut v = VC::new();
            for b in &bases {
                vc_join(&mut v, &b.threads[t].vc);
            }
            if bases.is_empty() {
                continue;
            }
            base.threads[t].make_mut().vc = v;
        }
        let same = |a: Option<&SyncCell>, b: Option<&SyncCell>| match (a, b) {
            (Some(a), Some(b)) => {
                (a.writer, a.scope, a.pc, a.rel_scope) == (b.writer, b.scope, b.pc, b.rel_scope)
                    && a.rel == b.rel
            }
            (None, None) => true,
            _ => false,
        };
        let keys: Vec<_> = base.syncs.keys().copied().collect();
        for k in keys {
            let c0 = cur.syncs.get(&k);
            let changed: Vec<&Machine> = bases
                .iter()
                .copied()
                .filter(|b| !same(b.syncs.get(&k), c0))
                .collect();
            match changed.as_slice() {
                [] => {}
                [b] => match (b.syncs.get(&k), base.syncs.get(&k)) {
                    (Some(c), Some(d))
                        if (c.writer, c.scope, c.pc, c.rel_scope)
                            == (d.writer, d.scope, d.pc, d.rel_scope) =>
                    {
                        let r = vc_meet(&c.rel, &d.rel);
                        base.syncs.get_mut(&k).unwrap().rel = r;
                    }
                    _ => {
                        base.syncs.remove(&k);
                    }
                },
                _ => {
                    base.syncs.remove(&k);
                }
            }
        }
        out.finals.clear();
        for p in parts {
            for (a, (cta, c)) in p.finals {
                let e = out.finals.entry(a).or_insert((cta, c));
                e.1 = e.1.max(c);
            }
            for (k, v) in p.obligs {
                for (e, w) in v {
                    let list = out.obligs.entry(k).or_default();
                    if !list.contains(&(e, w)) {
                        list.push((e, w));
                    }
                }
            }
            out.differ.extend(p.differ);
        }
        out
    }
}

/// Explore `m` (both shared-memory models, `smem::two_pass`), folding every terminal state
/// into a `Meet`.
pub(crate) fn explore_meet(mut m: Machine, max_states: usize) -> (Verdict, Meet) {
    let id = SINK_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    SINK.lock()
        .unwrap()
        .get_or_insert_with(Default::default)
        .insert(id, Meet::default());
    m.sink = id;
    let v = crate::smem::two_pass(m, |m| crate::explore::explore(m, max_states));
    let meet = SINK
        .lock()
        .unwrap()
        .as_mut()
        .and_then(|h| h.remove(&id))
        .unwrap_or_default();
    (v, meet)
}

/// A group certificate's verdict (or why it failed) and the end state on success.
type GroupCert = (Result<Verdict, Vec<String>>, Option<(Machine, Carry)>);

/// Multi-node: every rank explored launch group `i` on its own; check that all of them
/// continue from the same state (the continuation's fingerprint and obligations, the verdict's
/// completeness and finding signatures). Panics (ending the job) if they do not.
fn agree(i: usize, v: &Verdict, next: &Option<(Machine, Vec<Oblig>)>) {
    use crate::dist::Body;
    use crate::dist::Msg;
    let Some(d) = crate::dist::get() else {
        return;
    };
    let mut sigs: Vec<&str> = v.findings.iter().map(|f| f.sig.as_str()).collect();
    sigs.sort();
    let mut ob: Vec<String> = next
        .iter()
        .flat_map(|(_, o)| o.iter().map(|x| format!("{x:?}")))
        .collect();
    ob.sort();
    let digest = format!(
        "complete={} findings={sigs:?} next={:?} obligs={ob:?}",
        v.complete,
        next.as_ref()
            .map(|(m, _)| format!("{:032x}", m.fingerprint()))
    );
    // A call number no search uses (they count up from 1).
    let call = u64::MAX - 1 - i as u64;
    let sink = std::sync::Arc::new(crate::dist::MailSink::default());
    d.set_round(call, 0, Some(sink.clone()));
    let msg = |body| Msg {
        call,
        round: 0,
        body,
    };
    if d.rank != 0 {
        d.send(0, msg(Body::Report(digest.clone().into_bytes())));
    }
    let verdict: Vec<u8> = if d.rank == 0 {
        let others: Vec<(usize, Vec<u8>)> = sink.0.wait(|q| {
            let got: Vec<(usize, Vec<u8>)> = (1..d.size)
                .filter_map(|r| {
                    q.get(&r)?.iter().find_map(|x| match x {
                        Body::Report(b) => Some((r, b.clone())),
                        _ => None,
                    })
                })
                .collect();
            (got.len() == d.size - 1).then_some(got)
        });
        let bad: Vec<usize> = others
            .iter()
            .filter(|(_, b)| b.as_slice() != digest.as_bytes())
            .map(|(r, _)| *r)
            .collect();
        let dec = if bad.is_empty() {
            b"ok".to_vec()
        } else {
            let (r, b) = others.iter().find(|(r, _)| *r == bad[0]).unwrap();
            format!(
                "ranks {bad:?} disagree with rank 0\n  rank 0: {digest}\n  rank {r}: {}",
                String::from_utf8_lossy(b)
            )
            .into_bytes()
        };
        d.broadcast(&msg(Body::Decision(dec.clone())));
        dec
    } else {
        sink.0.wait(|q| {
            q.get(&0)?.iter().find_map(|x| match x {
                Body::Decision(b) => Some(b.clone()),
                _ => None,
            })
        })
    };
    // Unregister without dropping early messages of the next search (`set_round` keeps
    // those at or after the call it is given).
    d.set_round(0, 0, None);
    if verdict != b"ok" {
        panic!(
            "launch group {}: the ranks' explorations end in different states: {}",
            i + 1,
            String::from_utf8_lossy(&verdict)
        );
    }
    eprintln!(
        "launch group {}: all {} ranks continue from the same state",
        i + 1,
        d.size
    );
}

/// The certificate on the weak kernel (LDS/STS plain), then on the kernel itself, as
/// `wellsync::check_certified`; the verdict and the end state on success.
fn certify_group(m: &Machine) -> GroupCert {
    let mut why = Vec::new();
    if let Some(w) = crate::smem::weak(&m.k) {
        let mut mw = m.clone();
        mw.k = std::sync::Arc::new(w);
        let (o, keep) = crate::wellsync::certify_keep(mw);
        if let Some(v) = o.verdict {
            return (Ok(v), keep);
        }
        why.extend(o.reasons);
    }
    let (o, keep) = crate::wellsync::certify_keep(m.clone());
    match o.verdict {
        Some(v) => (Ok(v), keep),
        None => {
            why.extend(
                o.reasons
                    .into_iter()
                    .map(|r| format!("(LDS/STS strong) {r}")),
            );
            (Err(why), None)
        }
    }
}

/// `ptxv check --groups`: each group by certificate, else (last group only) exploration.
pub fn check_groups(
    full: Machine,
    groups: &[Vec<usize>],
    max_states: usize,
    report: &mut dyn FnMut(usize, &Verdict),
) -> Verdict {
    check_groups_pre(full, groups, max_states, report, None, &Opts::from_env())
}

/// `ptxv check --groups SPEC --slices`: explore each group by thread slices (E-SLICE), as
/// `PTXV_SLICES=1` does.
pub static SLICES: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// Debugging and bounded-run options of `check_groups_opts` (the CLI reads them from
/// `PTXV_GROUPS_RUN`, `PTXV_GROUPS_UPTO`, `PTXV_GROUPS_BOUNDED`). Each makes the verdict
/// incomplete.
#[derive(Clone, Debug, Default)]
pub struct Opts {
    /// Groups 1..run_first are one deterministic run each (explore::run_once), not checked.
    pub run_first: usize,
    /// Stop after this group (later groups stay held back).
    pub upto: Option<usize>,
    /// Continue past an incomplete explored group from the meet of the terminal states found.
    pub bounded: bool,
    /// Explore a group one thread slice at a time when it splits (E-SLICE, slices.rs).
    pub slices: bool,
}

impl Opts {
    pub fn from_env() -> Opts {
        let num = |k: &str| std::env::var(k).ok().and_then(|x| x.parse().ok());
        Opts {
            run_first: num("PTXV_GROUPS_RUN").unwrap_or(0),
            upto: num("PTXV_GROUPS_UPTO"),
            bounded: std::env::var("PTXV_GROUPS_BOUNDED").is_ok(),
            slices: std::env::var("PTXV_SLICES").is_ok_and(|x| x != "0")
                || SLICES.load(std::sync::atomic::Ordering::Relaxed),
        }
    }
}

/// `check_groups` with explicit `Opts`.
pub fn check_groups_opts(
    full: Machine,
    groups: &[Vec<usize>],
    max_states: usize,
    report: &mut dyn FnMut(usize, &Verdict),
    opts: &Opts,
) -> Verdict {
    check_groups_pre(full, groups, max_states, report, None, opts)
}

type Cert = (Result<Verdict, Vec<String>>, Option<(Machine, Carry)>);

fn check_groups_pre(
    full: Machine,
    groups: &[Vec<usize>],
    max_states: usize,
    report: &mut dyn FnMut(usize, &Verdict),
    mut pre: Option<Cert>,
    opts: &Opts,
) -> Verdict {
    let t0 = Instant::now();
    let mut all = Verdict {
        states: 0,
        transitions: 0,
        complete: true,
        terminal_states: 0,
        distinct_outputs: 0,
        findings: vec![],
        assumptions: vec![],
        notes: vec![],
        seconds: 0.0,
        outputs: vec![],
    };
    if let Err(e) = validate(&full, groups) {
        all.complete = false;
        all.notes.push(format!("launch groups not valid: {e}"));
        return all;
    }
    let first: BTreeSet<usize> = groups[0].iter().copied().collect();
    let mut cur = hold(full.clone(), &first);
    let mut how = Vec::new();
    // Debugging: PTXV_GROUPS_UPTO=k stops after group k (later groups stay held back; the
    // verdict covers only groups 1..k and is marked incomplete).
    let upto = opts.upto;
    for (i, g) in groups.iter().enumerate() {
        let last = i + 1 == groups.len() || upto == Some(i + 1);
        if upto == Some(i + 1) && i + 1 < groups.len() {
            all.complete = false;
            all.notes.push(format!(
                "PTXV_GROUPS_UPTO={}: later groups not checked",
                i + 1
            ));
        }
        // Debugging: PTXV_GROUPS_RUN=k replaces the check of groups 1..k by one
        // deterministic run (explore::run_once): later groups are then checked from that
        // run's end state only, and the verdict is incomplete.
        let run_k = opts.run_first;
        if i < run_k && !last {
            let (fin, fs, ok) = crate::explore::run_once(cur.clone());
            all.complete = false;
            all.notes.push(format!(
                "PTXV_GROUPS_RUN: group {} is one deterministic run",
                i + 1
            ));
            how.push(format!("group {} ({} launches): one run", i + 1, g.len()));
            for f in fs {
                if !all.findings.iter().any(|x| x.sig == f.sig) {
                    all.findings.push(f);
                }
            }
            if !ok {
                all.notes
                    .push(format!("group {}: the run did not terminate", i + 1));
                break;
            }
            cur = boundary(fin, &full, &groups[i + 1], &[]);
            continue;
        }
        let (r, keep) = match pre.take() {
            Some(c) if i == 0 => c,
            _ => certify_group(&cur),
        };
        let (v, next): (Verdict, Option<(Machine, Vec<Oblig>)>) = match r {
            Ok(v) => {
                how.push(format!("group {} ({} launches): certified", i + 1, g.len()));
                (v, keep.map(|(m, c)| (m, carry_obligs(&c))))
            }
            Err(why) => {
                eprintln!(
                    "certificate for launch group {} does not hold; exploring instead. Reasons:",
                    i + 1
                );
                let shown = if std::env::var("PTXV_WS_ALL").is_ok() {
                    usize::MAX
                } else {
                    12
                };
                for r in why.iter().take(shown) {
                    eprintln!("  - {r}");
                }
                // Debugging: PTXV_GROUPS_CERT_ONLY=all|k stops at a refused group (any, or
                // group k) instead of exploring it (the verdict is incomplete).
                let cert_only = std::env::var("PTXV_GROUPS_CERT_ONLY")
                    .ok()
                    .is_some_and(|v| v.parse::<usize>().map_or(true, |k| k == i + 1));
                if cert_only {
                    all.complete = false;
                    all.notes.push(format!(
                        "PTXV_GROUPS_CERT_ONLY: launch group {} not certified ({}), not explored",
                        i + 1,
                        why.first().map(String::as_str).unwrap_or("?")
                    ));
                    break;
                }
                // Multi-node: a group that later groups continue from is explored by every
                // rank on its own (its meet is a machine, which ranks do not exchange), and
                // the ranks check that they continue from the same state (`agree`).
                let local = !last && crate::dist::get().is_some();
                crate::dist::set_local(local);
                // E-SLICE (`PTXV_SLICES=1`, `ptxv check --groups ... --slices`): one thread
                // slice at a time when the group splits.
                let (mut v, meet) = if opts.slices {
                    crate::slices::explore_meet_sliced(&cur, max_states, &mut |k, name, v| {
                        eprintln!(
                            "  group {} slice {k} ({name}): {:?} states={} terminal={}{} {:.1}s",
                            i + 1,
                            v.classes(),
                            v.states,
                            v.terminal_states,
                            if v.complete { "" } else { " INCOMPLETE" },
                            v.seconds
                        )
                    })
                } else {
                    explore_meet(cur.clone(), max_states)
                };
                crate::dist::set_local(false);
                v.notes.push(format!(
                    "launch group {}: certificate failed ({}), explored",
                    i + 1,
                    why.first().map(String::as_str).unwrap_or("?")
                ));
                how.push(format!(
                    "group {} ({} launches): explored, {} states, {} terminal",
                    i + 1,
                    g.len(),
                    v.states,
                    meet.count
                ));
                // Debugging / bounded runs: PTXV_GROUPS_BOUNDED=1 continues past an incomplete
                // group from the meet of the terminal states found (the verdict stays
                // incomplete: later groups see only those continuations).
                let bounded = opts.bounded && meet.count > 0;
                if bounded && !v.complete {
                    all.complete = false;
                }
                let next = if last
                    || (!v.complete && !bounded)
                    || v.findings
                        .iter()
                        .any(|f| f.kind != "contract" && f.kind != "probe" && f.kind != "note")
                {
                    None
                } else if !meet.differ.is_empty() {
                    all.complete = false;
                    all.notes.push(format!(
                        "launch group {}: its terminal states differ in allocation(s) {:?}; later groups would need one continuation per final memory (not implemented)",
                        i + 1,
                        meet.differ
                    ));
                    None
                } else {
                    let ob: Vec<Oblig> = meet
                        .obligs
                        .iter()
                        .flat_map(|(&(a, i), v)| v.iter().map(move |&(e, w)| (a, i, i + 1, e, w)))
                        .collect();
                    meet.finish().map(|b| (b, ob))
                };
                if local {
                    agree(i, &v, &next);
                }
                (v, next)
            }
        };
        report(i, &v);
        all.states += v.states;
        all.transitions += v.transitions;
        all.complete &= v.complete;
        for f in &v.findings {
            if !all.findings.iter().any(|x| x.sig == f.sig) {
                all.findings.push(f.clone());
            }
        }
        for a in &v.assumptions {
            if !all.assumptions.contains(a) {
                all.assumptions.push(a.clone());
            }
        }
        for n in &v.notes {
            if !all.notes.contains(n) {
                all.notes.push(n.clone());
            }
        }
        if last {
            all.terminal_states = v.terminal_states;
            all.distinct_outputs = v.distinct_outputs;
            all.outputs = v.outputs;
            break;
        }
        let Some((fin, obligs)) = next else {
            all.complete = false;
            all.notes.push(format!(
                "launch group {}: no end state to continue from",
                i + 1
            ));
            break;
        };
        cur = boundary(fin, &full, &groups[i + 1], &obligs);
    }
    if !all.assumptions.iter().any(|a| a == "E-LAUNCH-GROUPS") {
        all.assumptions.push("E-LAUNCH-GROUPS".into());
    }
    all.notes.insert(
        0,
        format!("launch groups (E-LAUNCH-GROUPS): {}", how.join("; ")),
    );
    all.findings
        .sort_by(|a, b| (&a.kind, &a.sig).cmp(&(&b.kind, &b.sig)));
    all.seconds = t0.elapsed().as_secs_f64();
    all
}

/// Groups from a spec: "a,b;c,d" lists the launches of groups 2, 3, ...; group 1 is the
/// rest.
pub fn parse(spec: &str, n: usize) -> Result<Vec<Vec<usize>>, String> {
    let mut later: Vec<Vec<usize>> = Vec::new();
    for part in spec.split(';') {
        let mut g = Vec::new();
        for x in part.split(',').map(str::trim).filter(|x| !x.is_empty()) {
            if let Some((a, b)) = x.split_once('-') {
                let (a, b): (usize, usize) = (
                    a.parse().map_err(|_| format!("bad launch {a}"))?,
                    b.parse().map_err(|_| format!("bad launch {b}"))?,
                );
                g.extend(a..=b);
            } else {
                g.push(x.parse().map_err(|_| format!("bad launch {x}"))?);
            }
        }
        if !g.is_empty() {
            later.push(g);
        }
    }
    let used: BTreeSet<usize> = later.iter().flatten().copied().collect();
    let first: Vec<usize> = (0..n).filter(|x| !used.contains(x)).collect();
    let mut out = vec![first];
    out.extend(later);
    Ok(out)
}

/// `--groups SPEC`: `auto` tries the certificate on the whole case first; on refusal it
/// moves the launches a refusal names (with every launch of the same kernel and their stream
/// successors) to a second group and certifies the rest, repeating while the rest is refused
/// for reasons naming its own launches. Otherwise SPEC lists the later groups (`parse`).
pub fn check_spec(
    full: Machine,
    spec: &str,
    max_states: usize,
    report: &mut dyn FnMut(usize, &[usize], &Verdict),
) -> Verdict {
    let n = full.launch.grids.len();
    // NIC queue pairs (nic.rs) carry state across launches: the whole launch at once.
    if full.nic.is_some() {
        let mut v = crate::explore::explore(full, max_states);
        v.notes
            .push("--groups ignored: NIC queue pairs are checked as one launch".into());
        return v;
    }
    if spec != "auto" {
        return match parse(spec, n) {
            Ok(groups) => {
                let gs = groups.clone();
                check_groups(full, &groups, max_states, &mut |i, v| report(i, &gs[i], v))
            }
            Err(e) => {
                let mut v = check_groups(full, &[], max_states, &mut |_, _| {});
                v.notes = vec![format!("--groups {spec}: {e}")];
                v
            }
        };
    }
    let mut cert = certify_group(&full);
    let mut later: BTreeSet<usize> = BTreeSet::new();
    loop {
        let why = match &cert.0 {
            Ok(v) if later.is_empty() => {
                report(0, &(0..n).collect::<Vec<_>>(), v);
                let mut v = v.clone();
                v.notes
                    .insert(0, "launch groups: the whole case is certified".into());
                return v;
            }
            Ok(_) => break,
            Err(why) => why.clone(),
        };
        let named = named_launches(&full, &why);
        let grow: BTreeSet<usize> = named.difference(&later).copied().collect();
        if grow.is_empty() {
            break;
        }
        later.extend(grow);
        close(&full, &mut later);
        if later.len() == n {
            break;
        }
        let rest: BTreeSet<usize> = (0..n).filter(|x| !later.contains(x)).collect();
        eprintln!(
            "launch groups: certifying launches {:?} without {:?} ({})",
            rest,
            later,
            why.first().map(String::as_str).unwrap_or("?")
        );
        cert = certify_group(&hold(full.clone(), &rest));
    }
    if later.is_empty() || later.len() == n {
        // Nothing to split off: the certificate, else exploration, of the whole case.
        let groups = vec![(0..n).collect::<Vec<_>>()];
        let gs = groups.clone();
        return check_groups_pre(
            full,
            &groups,
            max_states,
            &mut |i, v| report(i, &gs[i], v),
            Some(cert),
            &Opts::from_env(),
        );
    }
    let groups = vec![
        (0..n).filter(|x| !later.contains(x)).collect::<Vec<_>>(),
        later.into_iter().collect::<Vec<_>>(),
    ];
    eprintln!("launch groups: {groups:?}");
    let gs = groups.clone();
    check_groups_pre(
        full,
        &groups,
        max_states,
        &mut |i, v| report(i, &gs[i], v),
        Some(cert),
        &Opts::from_env(),
    )
}

/// The launches named in certificate reasons (`k<launch>.`), with every launch of the same
/// kernel (the other ranks' instances of a collective).
fn named_launches(m: &Machine, reasons: &[String]) -> BTreeSet<usize> {
    let n = m.launch.grids.len();
    let mut bad = BTreeSet::new();
    for r in reasons {
        let b = r.as_bytes();
        for i in 0..b.len() {
            if b[i] == b'k' && (i == 0 || !b[i - 1].is_ascii_alphanumeric()) {
                let mut j = i + 1;
                while j < b.len() && b[j].is_ascii_digit() {
                    j += 1;
                }
                if j > i + 1 && j < b.len() && b[j] == b'.' {
                    if let Ok(x) = r[i + 1..j].parse::<usize>() {
                        if x < n {
                            bad.insert(x);
                        }
                    }
                }
            }
        }
    }
    let entries: BTreeSet<u32> = bad.iter().map(|&x| m.launch.grids[x].entry).collect();
    for x in 0..n {
        if entries.contains(&m.launch.grids[x].entry) {
            bad.insert(x);
        }
    }
    bad
}

/// Close a set of launches under stream successors.
fn close(m: &Machine, set: &mut BTreeSet<usize>) {
    let n = m.launch.grids.len();
    loop {
        let before = set.len();
        for x in 0..n {
            if let Some(p) = m.launch.prev_grid(x) {
                if set.contains(&p) {
                    set.insert(x);
                }
            }
        }
        if set.len() == before {
            break;
        }
    }
}
