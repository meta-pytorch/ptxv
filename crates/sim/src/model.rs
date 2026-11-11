/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A target-independent model interface and a stateful explorer over it.
//!
//! The GB300 explorer (`explore.rs`) is specialised to `Machine`; other targets implement `Model`
//! and reuse this search: states cached by a 128-bit fingerprint, depth-first with an explicit
//! stack, a shared visited set for parallel workers, findings deduplicated by signature with the
//! schedule that reached them, and the `Verdict` and `Finding` types of the GB300 path so reports
//! look the same.
//!
//! Reduction hooks: `enabled` may return a subset of the enabled actions (a persistent set);
//! `all_enabled` returns all of them. Cycle proviso: a state whose reduced successor was already
//! discovered at a depth not greater than its own, with the same `progress`, is fully expanded,
//! so every cycle of the reduced graph has a fully expanded state (depths cannot increase all
//! around a cycle, and every transition of a cycle keeps the progress measure).
//! Progress check (`check_progress`): every explored state must reach a terminal or stuck state
//! (or an unexplored one, when capped); a set of states that cannot is reported as a
//! `livelock` (actors that cycle forever: a spin on a flag nobody writes).

use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Debug;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::time::Instant;

use crate::explore::Verdict;
use crate::machine::Finding;

pub trait Model: Clone + Send + Sync + 'static {
    type Action: Clone + Debug + Send + Sync + 'static;

    /// The actions enabled in this state. Empty in a terminal or stuck state.
    fn enabled(&self) -> Vec<Self::Action>;
    /// Apply one action; findings raised while doing it.
    fn step(&mut self, a: &Self::Action) -> Vec<Finding>;
    /// Everything that affects the future (and the outputs).
    fn fingerprint(&self) -> u128;
    /// Normal termination (every actor finished).
    fn is_terminal(&self) -> bool;
    /// A finding for a non-terminal state with no enabled action (deadlock), or for a state
    /// that ended abnormally.
    fn stuck(&self) -> Option<Finding>;
    /// Output bytes of a terminal state (counts distinct outcomes).
    fn outputs(&self) -> Vec<u8>;
    fn describe(&self, a: &Self::Action) -> String;
    /// Assumption IDs the verdict rests on.
    fn assumptions(&self) -> Vec<String> {
        Vec::new()
    }
    /// Every enabled action, when `enabled` returns a reduced (persistent) set.
    fn all_enabled(&self) -> Vec<Self::Action> {
        self.enabled()
    }
    /// A measure that never decreases along a transition. Only transitions that keep it equal
    /// can lie on a cycle, so only they are subject to the cycle proviso (0: all are).
    fn progress(&self) -> u64 {
        0
    }
    /// Report livelocks: keep the explored state graph and report the states from which no end
    /// state (terminal, or stuck with or without a finding) is reachable: every schedule from
    /// them loops forever.
    fn detect_livelock(&self) -> bool {
        false
    }
    /// Check that every explored state can reach a terminal or stuck state, or an unexplored
    /// state when the search is capped (livelocks). Defaults to `detect_livelock`.
    fn check_progress(&self) -> bool {
        self.detect_livelock()
    }
    /// The finding for a livelock found by the progress check, describing a state on a cycle
    /// that never reaches an end (the explorer appends the trace, one turn of the cycle and the
    /// count of such states).
    fn livelock(&self) -> Finding {
        Finding {
            kind: "deadlock".into(),
            sig: "livelock".into(),
            msg: "livelock: reachable states cannot reach termination or a deadlock: every schedule from them loops forever (a poll or spin that nothing will satisfy)".into(),
            ..Default::default()
        }
    }
}

struct Path {
    parent: Option<Arc<Path>>,
    desc: String,
}

impl Drop for Path {
    fn drop(&mut self) {
        let mut cur = self.parent.take();
        while let Some(n) = cur {
            match Arc::try_unwrap(n) {
                Ok(mut n) => cur = n.parent.take(),
                Err(_) => break,
            }
        }
    }
}

fn trace(p: &Option<Arc<Path>>) -> Vec<String> {
    let mut v = Vec::new();
    let mut cur = p.as_ref();
    while let Some(n) = cur {
        v.push(n.desc.clone());
        cur = n.parent.as_ref();
    }
    v.reverse();
    v
}

/// A work item: state, path, depth.
type Item<M> = (M, Option<Arc<Path>>, u32);

/// The explored graph (progress check): edges, first parents, states with an end.
#[derive(Default)]
struct Graph {
    edges: Vec<(u128, u128)>,
    parent: HashMap<u128, u128>,
    /// Terminal or stuck states, and states left unexpanded (cap / deadline).
    ends: HashSet<u128>,
    expanded: HashSet<u128>,
}

struct Shared<M: Model> {
    stack: Mutex<Vec<Item<M>>>,
    /// Fingerprint -> discovery depth.
    visited: Mutex<HashMap<u128, u32>>,
    graph: Option<Mutex<Graph>>,
    findings: Mutex<Vec<Finding>>,
    outputs: Mutex<HashSet<Vec<u8>>>,
    states: AtomicUsize,
    transitions: AtomicUsize,
    terminal: AtomicUsize,
    busy: AtomicUsize,
    truncated: AtomicBool,
}

fn add_findings(sh: &Mutex<Vec<Finding>>, new: Vec<Finding>, path: &Option<Arc<Path>>) {
    if new.is_empty() {
        return;
    }
    let mut f = sh.lock().expect("findings lock");
    for mut x in new {
        if f.iter().any(|y| y.sig == x.sig) {
            continue;
        }
        if x.trace.is_empty() {
            x.trace = trace(path);
        }
        f.push(x);
    }
}

/// Explore every schedule of `m0` (up to `max_states` distinct states) with `workers` threads.
pub fn explore<M: Model>(m0: M, max_states: usize, workers: usize) -> Verdict {
    explore_until(m0, max_states, workers, None)
}

/// `explore` with a wall-clock budget: past `deadline` the search stops (verdict incomplete).
pub fn explore_until<M: Model>(
    m0: M,
    max_states: usize,
    workers: usize,
    deadline: Option<std::time::Duration>,
) -> Verdict {
    let t0 = Instant::now();
    let assumptions = m0.assumptions();
    let progress = m0.check_progress();
    let root = m0.clone();
    let sh = Arc::new(Shared {
        stack: Mutex::new(Vec::new()),
        visited: Mutex::new(HashMap::new()),
        graph: progress.then(|| Mutex::new(Graph::default())),
        findings: Mutex::new(Vec::new()),
        outputs: Mutex::new(HashSet::new()),
        states: AtomicUsize::new(0),
        transitions: AtomicUsize::new(0),
        terminal: AtomicUsize::new(0),
        busy: AtomicUsize::new(0),
        truncated: AtomicBool::new(false),
    });
    sh.visited
        .lock()
        .expect("visited")
        .insert(m0.fingerprint(), 0);
    sh.states.store(1, Ordering::Relaxed);
    sh.stack.lock().expect("stack").push((m0, None, 0));
    let workers = workers.max(1);
    std::thread::scope(|s| {
        for _ in 0..workers {
            let sh = sh.clone();
            s.spawn(move || worker(&sh, max_states, t0, deadline));
        }
    });
    let mut findings = std::mem::take(&mut *sh.findings.lock().expect("findings"));
    if let Some(g) = &sh.graph {
        let g = std::mem::take(&mut *g.lock().expect("graph"));
        if let Some(f) = livelock(&root, g) {
            findings.push(f);
        }
    }
    let outs: Vec<Vec<u8>> = sh
        .outputs
        .lock()
        .expect("outputs")
        .iter()
        .cloned()
        .collect();
    Verdict {
        states: sh.states.load(Ordering::Relaxed),
        transitions: sh.transitions.load(Ordering::Relaxed),
        complete: !sh.truncated.load(Ordering::Relaxed),
        terminal_states: sh.terminal.load(Ordering::Relaxed),
        distinct_outputs: outs.len(),
        findings,
        assumptions,
        notes: vec![format!("model explorer, {workers} worker(s)")],
        seconds: t0.elapsed().as_secs_f64(),
        outputs: outs,
    }
}

fn worker<M: Model>(
    sh: &Shared<M>,
    max_states: usize,
    t0: Instant,
    deadline: Option<std::time::Duration>,
) {
    loop {
        if deadline.is_some_and(|d| t0.elapsed() > d) {
            sh.truncated.store(true, Ordering::Relaxed);
            let mut st = sh.stack.lock().expect("stack");
            if let Some(g) = &sh.graph {
                let mut g = g.lock().expect("graph");
                for (m, _, _) in st.iter() {
                    g.ends.insert(m.fingerprint());
                }
            }
            st.clear();
        }
        let item = {
            let mut st = sh.stack.lock().expect("stack");
            let it = st.pop();
            if it.is_some() {
                sh.busy.fetch_add(1, Ordering::SeqCst);
            }
            it
        };
        let Some((m, path, depth)) = item else {
            if sh.busy.load(Ordering::SeqCst) == 0 {
                return;
            }
            std::thread::yield_now();
            continue;
        };
        let fp = sh.graph.as_ref().map_or(0, |_| m.fingerprint());
        let acts = m.enabled();
        let mut ends = Vec::new();
        if acts.is_empty() {
            if m.is_terminal() {
                sh.terminal.fetch_add(1, Ordering::Relaxed);
                sh.outputs.lock().expect("outputs").insert(m.outputs());
            }
            if let Some(f) = m.stuck() {
                add_findings(&sh.findings, vec![f], &path);
            }
            ends.push(fp);
        }
        let mut next = Vec::new();
        let mut edges = Vec::new();
        let n_acts = acts.len();
        let prog = m.progress();
        let mut todo = acts;
        let mut full = false;
        loop {
            // cycle proviso: a successor discovered at a depth <= ours may close a cycle
            let mut back = false;
            for a in todo {
                let mut m2 = m.clone();
                let f = m2.step(&a);
                sh.transitions.fetch_add(1, Ordering::Relaxed);
                let p = Some(Arc::new(Path {
                    parent: path.clone(),
                    desc: m.describe(&a),
                }));
                add_findings(&sh.findings, f, &p);
                let fp2 = m2.fingerprint();
                if sh.graph.is_some() {
                    edges.push(fp2);
                }
                {
                    let mut vis = sh.visited.lock().expect("visited");
                    if let Some(&d) = vis.get(&fp2) {
                        back |= d <= depth && m2.progress() == prog;
                        continue;
                    }
                    vis.insert(fp2, depth + 1);
                }
                if sh.states.fetch_add(1, Ordering::Relaxed) >= max_states {
                    sh.truncated.store(true, Ordering::Relaxed);
                    ends.push(fp2);
                    continue;
                }
                if let Some(g) = &sh.graph {
                    g.lock().expect("graph").parent.insert(fp2, fp);
                }
                next.push((m2, p, depth + 1));
            }
            if full || !back {
                break;
            }
            full = true;
            let all = m.all_enabled();
            if all.len() <= n_acts {
                break;
            }
            todo = all;
        }
        if let Some(g) = &sh.graph {
            let mut g = g.lock().expect("graph");
            g.expanded.insert(fp);
            g.ends.extend(ends);
            g.edges.extend(edges.into_iter().map(|t| (fp, t)));
        }
        sh.stack.lock().expect("stack").extend(next);
        sh.busy.fetch_sub(1, Ordering::SeqCst);
    }
}

/// Explored states that cannot reach an end (terminal, stuck or unexplored state): a livelock.
/// The finding describes a state on a cycle among them, with the schedule that reaches it from
/// the root and the cycle's actions.
fn livelock<M: Model>(root: &M, g: Graph) -> Option<Finding> {
    let mut radj: HashMap<u128, Vec<u128>> = HashMap::new();
    let mut fwd: HashMap<u128, Vec<u128>> = HashMap::new();
    let mut ok: HashSet<u128> = g.ends.clone();
    for &(a, b) in &g.edges {
        radj.entry(b).or_default().push(a);
        fwd.entry(a).or_default().push(b);
        if !g.expanded.contains(&b) {
            ok.insert(b);
        }
    }
    let mut work: Vec<u128> = ok.iter().copied().collect();
    while let Some(x) = work.pop() {
        for &p in radj.get(&x).map_or(&[][..], |v| &v[..]) {
            if ok.insert(p) {
                work.push(p);
            }
        }
    }
    let bad: HashSet<u128> = g
        .expanded
        .iter()
        .filter(|x| !ok.contains(x))
        .copied()
        .collect();
    let start = *bad.iter().min()?;
    // walk to a state on a cycle (every successor of a bad state is bad)
    let next_bad = |x: u128| -> Option<u128> {
        fwd.get(&x)?
            .iter()
            .copied()
            .filter(|y| bad.contains(y))
            .min()
    };
    let mut seen = HashSet::new();
    let mut cur = start;
    while seen.insert(cur) {
        match next_bad(cur) {
            Some(y) => cur = y,
            None => break,
        }
    }
    let mut chain = vec![cur];
    while let Some(&p) = g.parent.get(chain.last().expect("nonempty")) {
        if chain.len() > 1_000_000 {
            break;
        }
        chain.push(p);
    }
    chain.reverse();
    let step_to = |m: &M, target: u128| -> Option<(M, String)> {
        for a in m.all_enabled() {
            let mut m2 = m.clone();
            m2.step(&a);
            if m2.fingerprint() == target {
                return Some((m2, m.describe(&a)));
            }
        }
        None
    };
    let mut m = root.clone();
    let mut trace = Vec::new();
    if m.fingerprint() == chain[0] {
        for &t in &chain[1..] {
            match step_to(&m, t) {
                Some((m2, d)) => {
                    trace.push(d);
                    m = m2;
                }
                None => break,
            }
        }
    }
    let mut f = m.livelock();
    // one turn of the cycle
    let mut c = m.clone();
    let mut at = m.fingerprint();
    for _ in 0..16 {
        let Some(y) = next_bad(at) else { break };
        let Some((c2, d)) = step_to(&c, y) else {
            break;
        };
        trace.push(format!("(cycle) {d}"));
        c = c2;
        at = y;
        if at == cur {
            break;
        }
    }
    f.trace = trace;
    f.msg = format!("{} ({} states cannot reach the end)", f.msg, bad.len());
    Some(f)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two counters incremented in any order up to 2 each: 9 states, one outcome.
    #[derive(Clone)]
    struct Two([u8; 2]);

    impl Model for Two {
        type Action = usize;
        fn enabled(&self) -> Vec<usize> {
            (0..2).filter(|&i| self.0[i] < 2).collect()
        }
        fn step(&mut self, a: &usize) -> Vec<Finding> {
            self.0[*a] += 1;
            Vec::new()
        }
        fn fingerprint(&self) -> u128 {
            (self.0[0] as u128) << 8 | self.0[1] as u128
        }
        fn is_terminal(&self) -> bool {
            self.0 == [2, 2]
        }
        fn stuck(&self) -> Option<Finding> {
            None
        }
        fn outputs(&self) -> Vec<u8> {
            self.0.to_vec()
        }
        fn describe(&self, a: &usize) -> String {
            format!("inc {a}")
        }
    }

    #[test]
    fn counts_states() {
        for w in [1, 4] {
            let v = explore(Two([0, 0]), 1000, w);
            assert_eq!(v.states, 9, "3x3 grid of counter values");
            assert_eq!(v.distinct_outputs, 1);
            assert!(v.complete);
        }
    }

    /// A state that toggles forever between two values: no terminal state, a livelock.
    #[derive(Clone)]
    struct Spin(u8);

    impl Model for Spin {
        type Action = ();
        fn enabled(&self) -> Vec<()> {
            vec![()]
        }
        fn step(&mut self, _: &()) -> Vec<Finding> {
            self.0 ^= 1;
            Vec::new()
        }
        fn fingerprint(&self) -> u128 {
            self.0 as u128
        }
        fn is_terminal(&self) -> bool {
            false
        }
        fn stuck(&self) -> Option<Finding> {
            None
        }
        fn outputs(&self) -> Vec<u8> {
            Vec::new()
        }
        fn describe(&self, _: &()) -> String {
            "toggle".into()
        }
        fn check_progress(&self) -> bool {
            true
        }
    }

    #[test]
    fn livelock_reported() {
        let v = explore(Spin(0), 100, 1);
        assert!(v.complete);
        assert_eq!(v.terminal_states, 0);
        assert_eq!(v.findings.len(), 1);
        assert_eq!(v.findings[0].sig, "livelock");
        assert!(v.findings[0].trace.iter().any(|t| t.contains("toggle")));
    }
}
