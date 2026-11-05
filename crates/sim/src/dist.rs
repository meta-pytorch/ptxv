/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Multi-node exploration: the ranks of one `ptxv` job (Slurm tasks) connect in a full TCP
//! mesh and run the same program; each stateful search (`explore_one`) is a collective in
//! which every rank explores part of the state space (explore.rs, `DistRound`).
//!
//! - Visited states are partitioned by fingerprint: the owner of a state is
//!   `fp >> 64 mod ranks`, and only the owner's set says whether it is new. A rank that
//!   reaches a state owned elsewhere asks the owner (16 bytes) and keeps the state until the
//!   answer comes; states never travel, except when an idle rank steals work: then the
//!   victim sends schedules (action sequences) and the thief replays them.
//! - The synchronization reduction's learned sets are broadcast as they grow.
//! - Rank 0 coordinates: termination (all idle, no work in flight, two identical waves),
//!   the global state cap, and the end of each round (verdicts merged and sent back, so every
//!   rank continues with the same verdict and takes the same path through the program).
//!
//! Rendezvous through a shared directory (`PTXV_DIST_DIR`): rank r writes `addr.r` with the
//! host and port it listens on. Rank and size come from `SLURM_PROCID` / `SLURM_NTASKS`
//! (or `PTXV_RANK` / `PTXV_RANKS`).

use std::collections::HashMap;
use std::io::BufReader;
use std::io::BufWriter;
use std::io::Read;
use std::io::Write;
use std::net::TcpListener;
use std::net::TcpStream;
use std::net::ToSocketAddrs;
use std::sync::Arc;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::mpsc;
use std::time::Duration;
use std::time::Instant;

/// One message, with the collective it belongs to (`call`: the n-th search of the program;
/// `round`: the n-th round of that search).
#[derive(Debug, Clone)]
pub struct Msg {
    pub call: u64,
    pub round: u32,
    pub body: Body,
}

#[derive(Debug, Clone)]
pub enum Body {
    /// Is each fingerprint new? (fingerprint, the asker's token)
    Query(Vec<(u128, u64)>),
    /// The owner's answers: (token, new).
    Reply(Vec<(u64, bool)>),
    /// An idle rank asks for work.
    StealReq,
    /// Schedules to replay (possibly none).
    Work(Vec<Vec<u8>>),
    /// A learned set grew (encoded by explore.rs).
    Learn(Vec<u8>),
    /// To rank 0: idle, work messages sent / received, states owned, terminal states, wave.
    Status {
        idle: bool,
        sent: u64,
        recv: u64,
        states: u64,
        terminal: u64,
        wave: u64,
    },
    /// From rank 0: 1 = stop (state cap), 2 = the round's search terminated.
    Control(u8),
    /// Flush barrier: everything the sender sent before it has arrived.
    Marker,
    /// To rank 0 at the end of a round (encoded by explore.rs).
    Report(Vec<u8>),
    /// From rank 0: the merged round result (encoded by explore.rs).
    Decision(Vec<u8>),
}

/// Little-endian byte writer / reader for the wire format.
#[derive(Default)]
pub struct W(pub Vec<u8>);

impl W {
    pub fn u8(&mut self, x: u8) {
        self.0.push(x);
    }
    pub fn u32(&mut self, x: u32) {
        self.0.extend_from_slice(&x.to_le_bytes());
    }
    pub fn u64(&mut self, x: u64) {
        self.0.extend_from_slice(&x.to_le_bytes());
    }
    pub fn u128(&mut self, x: u128) {
        self.0.extend_from_slice(&x.to_le_bytes());
    }
    pub fn bytes(&mut self, b: &[u8]) {
        self.u64(b.len() as u64);
        self.0.extend_from_slice(b);
    }
    pub fn str(&mut self, s: &str) {
        self.bytes(s.as_bytes());
    }
}

pub struct R<'a>(pub &'a [u8]);

impl R<'_> {
    fn take(&mut self, n: usize) -> &[u8] {
        let (a, b) = self.0.split_at(n);
        self.0 = b;
        a
    }
    pub fn u8(&mut self) -> u8 {
        self.take(1)[0]
    }
    pub fn u32(&mut self) -> u32 {
        u32::from_le_bytes(self.take(4).try_into().unwrap())
    }
    pub fn u64(&mut self) -> u64 {
        u64::from_le_bytes(self.take(8).try_into().unwrap())
    }
    pub fn u128(&mut self) -> u128 {
        u128::from_le_bytes(self.take(16).try_into().unwrap())
    }
    pub fn bytes(&mut self) -> Vec<u8> {
        let n = self.u64() as usize;
        self.take(n).to_vec()
    }
    pub fn str(&mut self) -> String {
        String::from_utf8(self.bytes()).unwrap()
    }
}

fn encode(m: &Msg) -> Vec<u8> {
    let mut w = W::default();
    w.u64(m.call);
    w.u32(m.round);
    match &m.body {
        Body::Query(v) => {
            w.u8(1);
            w.u64(v.len() as u64);
            for &(f, t) in v {
                w.u128(f);
                w.u64(t);
            }
        }
        Body::Reply(v) => {
            w.u8(2);
            w.u64(v.len() as u64);
            for &(t, n) in v {
                w.u64(t);
                w.u8(n as u8);
            }
        }
        Body::StealReq => w.u8(3),
        Body::Work(v) => {
            w.u8(4);
            w.u64(v.len() as u64);
            for p in v {
                w.bytes(p);
            }
        }
        Body::Learn(b) => {
            w.u8(5);
            w.bytes(b);
        }
        Body::Status {
            idle,
            sent,
            recv,
            states,
            terminal,
            wave,
        } => {
            w.u8(6);
            w.u8(*idle as u8);
            w.u64(*sent);
            w.u64(*recv);
            w.u64(*states);
            w.u64(*terminal);
            w.u64(*wave);
        }
        Body::Control(k) => {
            w.u8(7);
            w.u8(*k);
        }
        Body::Marker => w.u8(8),
        Body::Report(b) => {
            w.u8(9);
            w.bytes(b);
        }
        Body::Decision(b) => {
            w.u8(10);
            w.bytes(b);
        }
    }
    w.0
}

fn decode(b: &[u8]) -> Msg {
    let mut r = R(b);
    let call = r.u64();
    let round = r.u32();
    let body = match r.u8() {
        1 => {
            let n = r.u64() as usize;
            Body::Query((0..n).map(|_| (r.u128(), r.u64())).collect())
        }
        2 => {
            let n = r.u64() as usize;
            Body::Reply((0..n).map(|_| (r.u64(), r.u8() != 0)).collect())
        }
        3 => Body::StealReq,
        4 => {
            let n = r.u64() as usize;
            Body::Work((0..n).map(|_| r.bytes()).collect())
        }
        5 => Body::Learn(r.bytes()),
        6 => Body::Status {
            idle: r.u8() != 0,
            sent: r.u64(),
            recv: r.u64(),
            states: r.u64(),
            terminal: r.u64(),
            wave: r.u64(),
        },
        7 => Body::Control(r.u8()),
        8 => Body::Marker,
        9 => Body::Report(r.bytes()),
        10 => Body::Decision(r.bytes()),
        k => panic!("dist: bad message kind {k}"),
    };
    Msg { call, round, body }
}

/// Where a rank's incoming messages go: the current collective's handler, or a buffer for
/// messages of a collective this rank has not reached yet.
pub trait Sink: Send + Sync {
    fn handle(&self, from: usize, m: Msg);
}

struct Router {
    /// (call, round) of the current round, and its handler.
    cur: Option<(u64, u32, Arc<dyn Sink>)>,
    /// Learned-set messages of the current call go here whatever their round.
    learn: Option<(u64, Arc<dyn Sink>)>,
    early: Vec<(usize, Msg)>,
}

pub struct Dist {
    pub rank: usize,
    pub size: usize,
    out: Vec<Option<Mutex<mpsc::Sender<Vec<u8>>>>>,
    router: Mutex<Router>,
    /// The number of searches started so far (the `call` of the next one).
    pub calls: Mutex<u64>,
}

static DIST: OnceLock<Option<Arc<Dist>>> = OnceLock::new();
static LOCAL: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

/// While set, searches run within this process only (`get` is None): sharded slices, where
/// each rank explores its own clusters.
pub fn set_local(on: bool) {
    LOCAL.store(on, std::sync::atomic::Ordering::SeqCst);
}

/// This process's rank context, when `PTXV_DIST_DIR` is set (connects on first use).
pub fn get() -> Option<Arc<Dist>> {
    if LOCAL.load(std::sync::atomic::Ordering::SeqCst) {
        return None;
    }
    raw()
}

/// The rank context even inside `set_local`.
pub fn raw() -> Option<Arc<Dist>> {
    DIST.get_or_init(|| {
        let dir = std::env::var("PTXV_DIST_DIR").ok()?;
        let env = |a: &str, b: &str| {
            std::env::var(a)
                .or_else(|_| std::env::var(b))
                .ok()
                .and_then(|s| s.parse::<usize>().ok())
        };
        let rank = env("PTXV_RANK", "SLURM_PROCID").expect("dist: no rank");
        let size = env("PTXV_RANKS", "SLURM_NTASKS").expect("dist: no size");
        if size <= 1 {
            return None;
        }
        Some(Dist::connect(&dir, rank, size))
    })
    .clone()
}

fn hostname() -> String {
    std::env::var("SLURMD_NODENAME").unwrap_or_else(|_| {
        std::fs::read_to_string("/proc/sys/kernel/hostname")
            .unwrap()
            .trim()
            .to_string()
    })
}

impl Dist {
    fn connect(dir: &str, rank: usize, size: usize) -> Arc<Dist> {
        std::fs::create_dir_all(dir).unwrap();
        // Dual-stack where available: host names may resolve to IPv6 first.
        let l = TcpListener::bind("[::]:0")
            .or_else(|_| TcpListener::bind("0.0.0.0:0"))
            .expect("dist: bind");
        let port = l.local_addr().unwrap().port();
        let tmp = format!("{dir}/.addr.{rank}");
        std::fs::write(&tmp, format!("{}:{port}", hostname())).unwrap();
        std::fs::rename(&tmp, format!("{dir}/addr.{rank}")).unwrap();
        let addr_of = |r: usize| -> String {
            let t0 = Instant::now();
            loop {
                if let Ok(s) = std::fs::read_to_string(format!("{dir}/addr.{r}")) {
                    return s;
                }
                assert!(
                    t0.elapsed() < Duration::from_secs(600),
                    "dist: rank {r} never came up"
                );
                std::thread::sleep(Duration::from_millis(100));
            }
        };
        let mut streams: Vec<Option<TcpStream>> = (0..size).map(|_| None).collect();
        // Connect to lower ranks, accept higher ones; a hello carries the connecting rank.
        for (r, slot) in streams.iter_mut().enumerate().take(rank) {
            let a = addr_of(r);
            let t0 = Instant::now();
            let mut s = 'conn: loop {
                for sa in a.to_socket_addrs().into_iter().flatten() {
                    if let Ok(s) = TcpStream::connect_timeout(&sa, Duration::from_secs(5)) {
                        break 'conn s;
                    }
                }
                assert!(
                    t0.elapsed() < Duration::from_secs(600),
                    "dist: cannot connect to {a}"
                );
                std::thread::sleep(Duration::from_millis(200));
            };
            s.write_all(&(rank as u32).to_le_bytes()).unwrap();
            *slot = Some(s);
        }
        for _ in rank + 1..size {
            let (mut s, _) = l.accept().unwrap();
            let mut b = [0u8; 4];
            s.read_exact(&mut b).unwrap();
            streams[u32::from_le_bytes(b) as usize] = Some(s);
        }
        let mut outs: Vec<Option<Mutex<mpsc::Sender<Vec<u8>>>>> = (0..size).map(|_| None).collect();
        let mut ends = Vec::new();
        for (r, s) in streams.into_iter().enumerate() {
            let Some(s) = s else { continue };
            s.set_nodelay(true).unwrap();
            let (tx, rx) = mpsc::channel::<Vec<u8>>();
            outs[r] = Some(Mutex::new(tx));
            ends.push((r, s, rx));
        }
        let d = Arc::new(Dist {
            rank,
            size,
            out: outs,
            router: Mutex::new(Router {
                cur: None,
                learn: None,
                early: Vec::new(),
            }),
            calls: Mutex::new(0),
        });
        for (r, s, rx) in ends {
            let ws = s.try_clone().unwrap();
            std::thread::spawn(move || {
                let mut w = BufWriter::with_capacity(1 << 20, ws);
                while let Ok(f) = rx.recv() {
                    let mut f = Some(f);
                    while let Some(b) = f.take() {
                        w.write_all(&(b.len() as u32).to_le_bytes()).unwrap();
                        w.write_all(&b).unwrap();
                        f = rx.try_recv().ok();
                    }
                    if w.flush().is_err() {
                        return;
                    }
                }
            });
            let d2 = d.clone();
            std::thread::spawn(move || {
                let mut rd = BufReader::with_capacity(1 << 20, s);
                let mut len = [0u8; 4];
                while rd.read_exact(&mut len).is_ok() {
                    let mut b = vec![0u8; u32::from_le_bytes(len) as usize];
                    rd.read_exact(&mut b).unwrap();
                    d2.deliver(r, decode(&b));
                }
            });
        }
        d
    }

    fn deliver(&self, from: usize, m: Msg) {
        let sink = {
            let mut g = self.router.lock().unwrap();
            let s = match (&m.body, &g.cur, &g.learn) {
                (Body::Learn(_), _, Some((c, s))) if *c == m.call => Some(s.clone()),
                (Body::Learn(_), _, _) => None,
                (_, Some((c, r, s)), _) if *c == m.call && *r == m.round => Some(s.clone()),
                _ => None,
            };
            if s.is_none() {
                g.early.push((from, m.clone()));
            }
            s
        };
        if let Some(s) = sink {
            s.handle(from, m);
        }
    }

    pub fn send(&self, to: usize, m: Msg) {
        if to == self.rank {
            self.deliver(to, m);
            return;
        }
        let b = encode(&m);
        let _ = self.out[to].as_ref().unwrap().lock().unwrap().send(b);
    }

    pub fn broadcast(&self, m: &Msg) {
        for r in 0..self.size {
            if r != self.rank {
                self.send(r, m.clone());
            }
        }
    }

    /// Route the current call's learned-set messages to `s` (and deliver early ones).
    pub fn set_learn(&self, call: u64, s: Option<Arc<dyn Sink>>) {
        let early = {
            let mut g = self.router.lock().unwrap();
            g.learn = s.clone().map(|s| (call, s));
            let (mine, rest): (Vec<_>, Vec<_>) = std::mem::take(&mut g.early)
                .into_iter()
                .partition(|(_, m)| matches!(m.body, Body::Learn(_)) && m.call == call);
            g.early = rest;
            mine
        };
        if let Some(s) = s {
            for (f, m) in early {
                s.handle(f, m);
            }
        }
    }

    /// Route round (`call`, `round`)'s messages to `s` (and deliver early ones).
    pub fn set_round(&self, call: u64, round: u32, s: Option<Arc<dyn Sink>>) {
        let early = {
            let mut g = self.router.lock().unwrap();
            g.cur = s.clone().map(|s| (call, round, s));
            // Messages of rounds that are over (late status reports, steal requests).
            g.early.retain(|(_, m)| {
                (m.call, m.round) >= (call, round) || matches!(m.body, Body::Learn(_))
            });
            let (mine, rest): (Vec<_>, Vec<_>) = std::mem::take(&mut g.early)
                .into_iter()
                .partition(|(_, m)| {
                    !matches!(m.body, Body::Learn(_)) && m.call == call && m.round == round
                });
            g.early = rest;
            mine
        };
        if let Some(s) = s {
            for (f, m) in early {
                s.handle(f, m);
            }
        }
    }

    /// The owner of a state.
    pub fn owner(&self, fp: u128) -> usize {
        ((fp >> 64) as u64 % self.size as u64) as usize
    }
}

/// A Mailbox as a round's message handler.
#[derive(Default)]
pub struct MailSink(pub Mailbox);

impl Sink for MailSink {
    fn handle(&self, from: usize, m: Msg) {
        self.0.put(from, m.body);
    }
}

/// Collects messages of one kind for a blocking wait (used for the round-end exchange).
#[derive(Default)]
pub struct Mailbox {
    pub q: Mutex<HashMap<usize, Vec<Body>>>,
    pub cv: std::sync::Condvar,
}

impl Mailbox {
    pub fn put(&self, from: usize, b: Body) {
        self.q.lock().unwrap().entry(from).or_default().push(b);
        self.cv.notify_all();
    }
    /// Wait until `pred` holds on the collected messages; return its value.
    pub fn wait<T>(&self, mut pred: impl FnMut(&mut HashMap<usize, Vec<Body>>) -> Option<T>) -> T {
        let mut g = self.q.lock().unwrap();
        loop {
            if let Some(t) = pred(&mut g) {
                return t;
            }
            g = self
                .cv
                .wait_timeout(g, Duration::from_millis(50))
                .unwrap()
                .0;
        }
    }
}
