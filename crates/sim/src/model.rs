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
    /// Report livelocks: keep the explored state graph and, at the end of a complete search,
    /// report the states from which no end state (terminal, or stuck with or without a finding)
    /// is reachable: every schedule from them loops forever.
    fn detect_livelock(&self) -> bool {
        false
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

/// The explored graph for livelock detection: successors per state, end states, and one path
/// to each state.
#[derive(Default)]
struct Graph {
    succ: std::collections::HashMap<u128, Vec<u128>>,
    ends: HashSet<u128>,
    path: std::collections::HashMap<u128, Option<Arc<Path>>>,
}

struct Shared<M: Model> {
    graph: Option<Mutex<Graph>>,
    stack: Mutex<Vec<(M, Option<Arc<Path>>)>>,
    visited: Mutex<HashSet<u128>>,
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
    let fp0 = m0.fingerprint();
    let sh = Arc::new(Shared {
        graph: m0.detect_livelock().then(|| {
            let mut g = Graph::default();
            g.path.insert(fp0, None);
            Mutex::new(g)
        }),
        stack: Mutex::new(Vec::new()),
        visited: Mutex::new(HashSet::new()),
        findings: Mutex::new(Vec::new()),
        outputs: Mutex::new(HashSet::new()),
        states: AtomicUsize::new(0),
        transitions: AtomicUsize::new(0),
        terminal: AtomicUsize::new(0),
        busy: AtomicUsize::new(0),
        truncated: AtomicBool::new(false),
    });
    sh.visited.lock().expect("visited").insert(m0.fingerprint());
    sh.states.store(1, Ordering::Relaxed);
    sh.stack.lock().expect("stack").push((m0, None));
    let workers = workers.max(1);
    std::thread::scope(|s| {
        for _ in 0..workers {
            let sh = sh.clone();
            s.spawn(move || worker(&sh, max_states, t0, deadline));
        }
    });
    let complete = !sh.truncated.load(Ordering::Relaxed);
    if let (Some(g), true) = (&sh.graph, complete) {
        let g = g.lock().expect("graph");
        if let Some(f) = livelock(&g) {
            sh.findings.lock().expect("findings").push(f);
        }
    }
    let findings = std::mem::take(&mut *sh.findings.lock().expect("findings"));
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
            sh.stack.lock().expect("stack").clear();
        }
        let item = {
            let mut st = sh.stack.lock().expect("stack");
            let it = st.pop();
            if it.is_some() {
                sh.busy.fetch_add(1, Ordering::SeqCst);
            }
            it
        };
        let Some((m, path)) = item else {
            if sh.busy.load(Ordering::SeqCst) == 0 {
                return;
            }
            std::thread::yield_now();
            continue;
        };
        let acts = m.enabled();
        let fp_m = sh.graph.as_ref().map(|_| m.fingerprint());
        if acts.is_empty() {
            if let (Some(g), Some(f)) = (&sh.graph, fp_m) {
                g.lock().expect("graph").ends.insert(f);
            }
            if m.is_terminal() {
                sh.terminal.fetch_add(1, Ordering::Relaxed);
                sh.outputs.lock().expect("outputs").insert(m.outputs());
            }
            if let Some(f) = m.stuck() {
                add_findings(&sh.findings, vec![f], &path);
            }
        }
        let mut next = Vec::new();
        for a in acts {
            let mut m2 = m.clone();
            let f = m2.step(&a);
            sh.transitions.fetch_add(1, Ordering::Relaxed);
            let p = Some(Arc::new(Path {
                parent: path.clone(),
                desc: m.describe(&a),
            }));
            add_findings(&sh.findings, f, &p);
            let fp = m2.fingerprint();
            if let (Some(g), Some(from)) = (&sh.graph, fp_m) {
                let mut g = g.lock().expect("graph");
                g.succ.entry(from).or_default().push(fp);
                g.path.entry(fp).or_insert_with(|| p.clone());
            }
            if !sh.visited.lock().expect("visited").insert(fp) {
                continue;
            }
            if sh.states.fetch_add(1, Ordering::Relaxed) >= max_states {
                sh.truncated.store(true, Ordering::Relaxed);
                continue;
            }
            next.push((m2, p));
        }
        sh.stack.lock().expect("stack").extend(next);
        sh.busy.fetch_sub(1, Ordering::SeqCst);
    }
}

/// States from which no end state is reachable (reverse reachability over the explored graph).
fn livelock(g: &Graph) -> Option<Finding> {
    let mut pred: std::collections::HashMap<u128, Vec<u128>> = std::collections::HashMap::new();
    for (a, bs) in &g.succ {
        for b in bs {
            pred.entry(*b).or_default().push(*a);
        }
    }
    let mut live: HashSet<u128> = g.ends.clone();
    let mut todo: Vec<u128> = g.ends.iter().copied().collect();
    while let Some(x) = todo.pop() {
        for y in pred.get(&x).into_iter().flatten() {
            if live.insert(*y) {
                todo.push(*y);
            }
        }
    }
    let stuck: Vec<&u128> = g.path.keys().filter(|s| !live.contains(s)).collect();
    let first = stuck
        .iter()
        .min_by_key(|s| trace(&g.path[s]).len())
        .map(|s| trace(&g.path[s]))?;
    Some(Finding {
        kind: "deadlock".into(),
        sig: "livelock".into(),
        msg: format!(
            "livelock: {} reachable state(s) cannot reach termination or a deadlock: every schedule from them loops forever (a poll or spin that nothing will satisfy)",
            stuck.len()
        ),
        trace: first,
        ..Default::default()
    })
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
}
