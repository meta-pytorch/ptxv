/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! MISS 58: the GPU validation refuses a numerics DAG that GB300 contradicts
//! (`errdist::validate::gpu_dag_refusal`), except where another order of an output's atomic
//! chain explains GB300's value (`Prog::atomic_chains`, `AtomChain::order_explains`,
//! N-ERRDIST-DAG). The decision on synthetic mismatch counts and programs, without a GPU; the
//! recorded GB300 runs are in the fixes' reports. Targets of the source mutants
//! gpu-refuse-off, gpu-refuse-atomic, gpu-fed-off, atom-sources-off, atom-operands-off and
//! atom-bound-off (tools/mutate/srcmut.py). The follow-up: a validation refuses Tier 3 itself
//! on the analytic tiers' criteria (`validate::tier3_refusal`; mutants t3-refusal-off,
//! t3-moments-off), and a run's inputs bound P(differ) exactly against a claimed interval
//! (`validate::p_contradicts`; mutant p-exact-off). The MISS 84 follow-up: no excuse where the
//! bound is vacuous (mutant atom-vacuous-excuses) or where the run's happens-before orders
//! every add (mutant atom-free-off).

use ptxv_sim::errdist::chain::Op;
use ptxv_sim::errdist::chain::Prog;
use ptxv_sim::errdist::mc::Obs;
use ptxv_sim::errdist::mc::p_interval;
use ptxv_sim::errdist::mc::p_interval_obs;
use ptxv_sim::errdist::report::Estimate;
use ptxv_sim::errdist::spec::Format;
use ptxv_sim::errdist::validate::against_estimate;
use ptxv_sim::errdist::validate::gpu_dag_refusal;
use ptxv_sim::errdist::validate::p_contradicts;
use ptxv_sim::errdist::validate::tier3_refusal;

/// A DAG mismatch at an output without atomic adds refuses (the review's runs printed DISAGREE
/// next to the headline with 10 and 320 such outputs).
#[test]
fn gpu_mismatch_refuses() {
    for (m, inputs) in [(1, 400), (10, 400), (320, 400)] {
        let e = gpu_dag_refusal(m, 0, inputs).expect("a mismatch refuses");
        assert!(
            e.contains(&format!("{m} tracked outputs over {inputs}")) && e.contains("refusing"),
            "{e}"
        );
    }
    assert_eq!(gpu_dag_refusal(0, 0, 400), None);
}

/// Mismatches that another order of their atomic chain explains are no refusal. One more
/// mismatch elsewhere refuses, naming both counts.
#[test]
fn atomic_order_mismatch_is_not_a_refusal() {
    assert_eq!(gpu_dag_refusal(64, 64, 400), None);
    let e = gpu_dag_refusal(65, 64, 400).expect("the unexplained output refuses");
    assert!(
        e.contains("on 1 tracked outputs") && e.contains("64 outputs whose atomic adds GB300"),
        "{e}"
    );
}

fn leaf(idx: u64) -> Op {
    Op::Leaf {
        buf: 0,
        idx,
        fmt: Format::F32,
    }
}

fn add(a: u32, b: u32) -> Op {
    Op::Add {
        a,
        b,
        rz: false,
        ftz: true,
    }
}

/// `Prog::atomic_fed` marks the outputs whose DAG holds an atomic add, through later ops.
#[test]
fn atomic_fed_follows_the_dag() {
    let p = Prog {
        // 0, 1: leaves; 2: an atomic add; 3: a plain add; 4: the atomic add's rounding;
        // 5: a plain add of the atomic sum.
        ops: vec![
            leaf(0),
            leaf(1),
            add(0, 1),
            add(0, 1),
            Op::Round { arg: 2, f16: false },
            add(3, 2),
        ],
        hw: vec![0; 6],
        outputs: vec![Some(2), Some(3), Some(4), Some(5), None],
        out_fmt: Format::F32,
        bufs: vec!["x".into()],
        dep_untracked: vec![],
        atomic: vec![2],
        atomic_src: vec![0],
        atomic_free: vec![false],
        pre_rounded: vec![],
    };
    assert_eq!(p.atomic_fed(), [true, false, true, true, false]);
    let q = Prog {
        atomic: vec![],
        ..p.clone()
    };
    assert_eq!(q.atomic_fed(), [false; 5]);
}

/// One location's chain of fp32 atomic adds (`srcs`: each add's issuing thread) from a
/// constant `init`, on leaves 0..3, and a plain add of the sum: outputs [the sum, the later
/// add]. Adds of two threads are unordered by the run's happens-before (no barrier).
fn chain(init: f64, srcs: &[u32]) -> Prog {
    let mut ops = vec![leaf(0), leaf(1), leaf(2), leaf(3), Op::Const(init)];
    let mut cur = 4;
    let mut atomic = vec![];
    for k in 0..srcs.len() as u32 {
        ops.push(add(cur, k));
        cur = ops.len() as u32 - 1;
        atomic.push(cur);
    }
    ops.push(add(cur, 3));
    let n = ops.len();
    Prog {
        ops,
        hw: vec![0; n],
        outputs: vec![Some(cur), Some(n as u32 - 1)],
        out_fmt: Format::F32,
        bufs: vec!["x".into()],
        dep_untracked: vec![],
        atomic,
        atomic_src: srcs.to_vec(),
        atomic_free: (0..srcs.len())
            .map(|i| i > 0 && srcs[i] != srcs[i - 1])
            .collect(),
        pre_rounded: vec![],
    }
}

/// A bf16 atomic chain of `n` adds from `n` threads, from 0, on leaves 0..n-1, each add rounded
/// to bf16 (`Op::Round`): the output is the last rounding.
fn chain16(n: u32) -> Prog {
    let mut ops: Vec<Op> = (0..n as u64).map(leaf).collect();
    ops.push(Op::Const(0.0));
    let mut cur = n;
    let mut atomic = vec![];
    for k in 0..n {
        ops.push(add(cur, k));
        atomic.push(ops.len() as u32 - 1);
        ops.push(Op::Round {
            arg: ops.len() as u32 - 1,
            f16: false,
        });
        cur = ops.len() as u32 - 1;
    }
    let len = ops.len();
    Prog {
        ops,
        hw: vec![0; len],
        outputs: vec![Some(cur)],
        out_fmt: Format::Bf16,
        bufs: vec!["x".into()],
        dep_untracked: vec![],
        atomic_src: (0..n).collect(),
        atomic_free: (0..n).map(|i| i > 0).collect(),
        atomic,
        pre_rounded: vec![],
    }
}

/// MISS 58, narrowed: GB300's order of the adds explains a different value only at an output
/// that is an atomic location's value, whose adds come from two or more threads and add three
/// or more operands counting a nonzero initial value, and only within twice the reordering
/// bound gamma_k * sum |x| + k * eta. Before, every output whose DAG held an atomic add was
/// exempt: an FTZ the DAG lacked (y = x[j] + x[j + 32] from 0, one rounding in either order)
/// and errors upstream of one thread's adds went unchecked.
#[test]
fn atomic_order_explains_only_a_free_order_within_its_bound() {
    let v = |i: u32| [1.0, 3.0, -2.0, 0.5, 0.0][i.min(4) as usize];
    let ulp1 = 2f32.powi(-23);
    // Three adds from three threads: a value one ulp of 2 off the DAG's is another order's.
    let p = chain(0.0, &[0, 1, 2]);
    let ch = p.atomic_chains();
    let c = ch[0].as_ref().expect("the output is the location's value");
    assert_eq!((c.adds.len(), c.init, c.sources), (3, 4, 3));
    assert!(c.order_explains(&p, &v, 2.0, 2.0 + 2.0 * ulp1));
    // Far outside the bound (6 * 2^-24 * 6 + eta, doubled): no order gives it.
    assert!(!c.order_explains(&p, &v, 2.0, 2.0 + 0.01));
    // The plain add of the sum is no atomic location's value: no chain, no exemption.
    assert!(ch[1].is_none(), "{:?}", ch[1]);
    // Two operands from zero round once in either order.
    let two = chain(0.0, &[0, 1]);
    let c2 = two.atomic_chains()[0].clone().unwrap();
    assert!(!c2.order_explains(&two, &v, 4.0, 4.0 + 4.0 * ulp1));
    // ... unless the location held a nonzero value first.
    let two_init = chain(0.5, &[0, 1]);
    let v2 = |i: u32| [1.0, 3.0, -2.0, 0.5, 0.5][i.min(4) as usize];
    let c3 = two_init.atomic_chains()[0].clone().unwrap();
    assert!(c3.order_explains(&two_init, &v2, 4.5, 4.5 + 4.0 * ulp1));
    // One thread's adds are in program order: GB300's is the DAG's.
    let one = chain(0.0, &[7, 7, 7]);
    let c4 = one.atomic_chains()[0].clone().unwrap();
    assert_eq!(c4.sources, 1);
    assert!(!c4.order_explains(&one, &v, 2.0, 2.0 + 2.0 * ulp1));
    // Three threads whose adds the run's happens-before orders (a barrier between each two,
    // fixtures/errdist/atomord/bar): one order, the DAG's (MISS 84 follow-up).
    let ordered = Prog {
        atomic_free: vec![false; 3],
        ..p.clone()
    };
    let c5 = ordered.atomic_chains()[0].clone().unwrap();
    assert_eq!((c5.sources, c5.free), (3, false));
    assert!(!c5.order_explains(&ordered, &v, 2.0, 2.0 + 2.0 * ulp1));
}

/// MISS 84 follow-up, point (a): gamma_k has no bound at k u >= 1/2 (a bf16 chain of 128 adds),
/// and the excuse then took any value as another order's. It now excuses none; at 64 adds the
/// bound stands and a difference of one bf16 ulp is another order's.
#[test]
fn atomic_order_explains_nothing_past_a_vacuous_bound() {
    let v = |_: u32| 1.0;
    let long = chain16(128);
    let c = long.atomic_chains()[0].clone().expect("a bf16 chain");
    assert_eq!((c.adds.len(), c.sources, c.free), (128, 128, true));
    assert_eq!(c.u, 2f64.powi(-8));
    assert!(!c.order_explains(&long, &v, 128.0, 129.0));
    assert!(!c.order_explains(&long, &v, 128.0, 1.0e6));
    let short = chain16(64);
    let c = short.atomic_chains()[0].clone().unwrap();
    assert!(c.order_explains(&short, &v, 64.0, 64.5));
    assert!(!c.order_explains(&short, &v, 64.0, 1.0e6));
}

/// Observations of `n` inputs of 64 outputs, `k` of them with `d` differing outputs each.
fn obs(n: u64, k: u64, d: u64) -> Obs {
    let mut o = Obs {
        inputs: n,
        compared: 64 * n,
        differ: k * d,
        ..Default::default()
    };
    o.input_differ = (0..n)
        .map(|i| if i < k { d as f64 / 64.0 } else { 0.0 })
        .collect();
    o
}

/// MISS 120 counts inputs in a claim's P(differ) interval where few differ; an observed run
/// that refutes a claim keeps the outputs' and batch-means intervals (`mc::p_interval_obs`).
/// With the input-level bound on that side too, a kernel_db run's simulator validation (4
/// inputs of 786,432 outputs; cute.SplitKReduce, P(differ) 0.220 on every input) held
/// P(differ) to [1.4e-3, 0.84], and its Tier 2 of 0.228 (4% high) stood, as did the estimates
/// of 19 other refusals in 18 of 199 runs. Mutant obs-input-bound puts the bound back there.
#[test]
fn an_observed_run_refutes_with_the_outputs_and_inputs_intervals() {
    let c = 786_432u64;
    let f = [0.21990, 0.22004, 0.21985, 0.21997];
    let mut run = Obs {
        inputs: 4,
        compared: 4 * c,
        ..Default::default()
    };
    run.differ = f.iter().map(|x| (x * c as f64).round() as u64).sum();
    run.input_differ = f.to_vec();
    run.input_compared = vec![c; 4];
    let t2 = Estimate {
        tier: 2,
        p_differ: Some((0.2284, 0.2201, 0.2308)),
        ..Default::default()
    };
    let v = against_estimate("sim", &t2, &run, 0);
    assert!(!v.agree, "{:?}", v.notes);
    let t2 = Estimate {
        tier: 2,
        p_differ: Some((0.21995, 0.2199, 0.2200)),
        ..Default::default()
    };
    assert!(against_estimate("sim", &t2, &run, 0).agree);
    // As a claim (a Tier 3 of these 4 inputs) the run counts inputs: 4 < SMALL_K.
    let (lo, hi) = p_interval(&run, 0.05);
    assert!(lo < 0.01 && hi > 0.8, "[{lo}, {hi}]");
    let (lo, hi) = p_interval_obs(&run, 0.05);
    assert!(lo > 0.219 && hi < 0.221, "[{lo}, {hi}]");
}

/// The follow-up (MISS 42's rule for Tier 3): a run whose P(differ) Tier 3's contradicts
/// refuses Tier 3; so does a bitwise mismatch that no other order of an atomic chain explains.
/// Explained mismatches alone (GB300's own order of the adds), agreeing P(differ), and a side
/// that compared nothing (out of domain) do not. Since MISS 120 a Tier 3 of 5 inputs with
/// nothing differing claims [0, 0.52] (inputs counted), which holds the run's 0.05, and stands;
/// a narrower claim is refused by the run's exact bounds.
#[test]
fn validation_refuses_tier3_on_p_differ_or_bits() {
    // Tier 3: 400 inputs, 40 with 1 of 64 differing (P(differ) 1.6e-3). The run: 1000 inputs,
    // a fifth with 16 of 64 differing (0.05).
    let (t3, run) = (obs(400, 40, 1), obs(1000, 200, 16));
    let e = tier3_refusal("sim", &t3, &run, 0, 0, (1.0e-3, 2.2e-3))
        .expect("P(differ) 1.6e-3 against 0.05 refuses");
    assert!(
        e.contains("Tier 3's P(differ)")
            && e.contains("DISAGREES FAILS")
            && e.contains("1000 inputs"),
        "{e}"
    );
    let t3 = obs(5, 0, 0);
    let own = p_interval(&t3, 0.05);
    assert!(own.0 == 0.0 && own.1 > 0.5, "{own:?}");
    assert_eq!(tier3_refusal("sim", &t3, &run, 0, 0, own), None);
    let e = tier3_refusal("sim", &t3, &run, 0, 0, (0.0, 0.01)).expect("a narrower claim");
    assert!(
        e.contains("exact bounds") && e.contains("DISAGREES FAILS"),
        "{e}"
    );
    // The same rate on both sides agrees.
    let same = obs(300, 60, 16);
    let claim = (0.045, 0.055);
    assert_eq!(tier3_refusal("sim", &same, &run, 0, 0, claim), None);
    // Bits: explained mismatches alone pass; one more refuses.
    assert_eq!(tier3_refusal("gpu", &same, &run, 218, 218, claim), None);
    let e = tier3_refusal("gpu", &same, &run, 219, 218, claim).expect("an unexplained one");
    assert!(
        e.contains("1 outputs differ bit for bit from the DAG where no other order"),
        "{e}"
    );
    // Nothing compared on either side: out of domain, not this rule's.
    let none = Obs {
        inputs: 4,
        ..Default::default()
    };
    assert_eq!(tier3_refusal("sim", &t3, &none, 0, 0, (0.0, 0.01)), None);
    assert_eq!(tier3_refusal("sim", &none, &run, 0, 0, claim), None);
}

/// A discrepancy confined to a few inputs: GB300 runs of the atomic sums without the DAG's FTZ
/// differ on every output of 4 of 400 inputs (1.00e-2), where Tier 3 (300 inputs, nothing
/// differs) claims P(differ) in [0, 3.84e-4]. The batch-means test passes it (its standard
/// error is about sqrt(4) / 400); the inputs' exact bounds, Clopper-Pearson over inputs, put
/// P(differ) at least 5e-4 (99.9%): refused. The same claim against 0 of 400 inputs, or a
/// claim the bounds contain, stands. Since MISS 120 Tier 3's own claim at 0 of 300 inputs is
/// [0, 1.2e-2] (inputs counted), which holds the run's 1.00e-2, and stands.
#[test]
fn few_inputs_contradict_a_claimed_interval() {
    let (t3, run) = (obs(300, 0, 0), obs(400, 4, 64));
    let own = p_interval(&t3, 0.05);
    assert!(own.1 > 1.0e-2, "{own:?}");
    assert_eq!(tier3_refusal("gpu", &t3, &run, 0, 0, own), None);
    let claim = (0.0, 3.84e-4);
    let (lo, hi) = p_contradicts(claim, &run).expect("4 of 400 inputs contradict [0, 3.84e-4]");
    assert!(lo > 3.84e-4 && lo < 1e-2 && hi > 1e-2, "{lo} {hi}");
    let e = tier3_refusal("gpu", &t3, &run, 0, 0, claim).expect("refused");
    assert!(
        e.contains("exact bounds") && e.contains("DISAGREES FAILS"),
        "{e}"
    );
    assert_eq!(p_contradicts(claim, &obs(400, 0, 0)), None);
    assert_eq!(p_contradicts((0.0, 2e-2), &run), None);
    // Agreeing row sums (P about 0.67 on both sides) stand.
    let claim = (0.656, 0.679);
    assert_eq!(p_contradicts(claim, &obs(200, 200, 44)), None);
}

/// With mismatches explained as GB300's order of atomic adds the two sides no longer sample one
/// DAG, and an error distribution the run contradicts refuses Tier 3; without them the moments
/// raise no refusal (the follow-up's rule).
#[test]
fn moments_count_where_atomic_order_differs() {
    let mut t3 = obs(300, 60, 16);
    let mut run = obs(400, 80, 16);
    t3.input_means = (0..300).map(|i| (0.01 * (i % 7) as f64, 0.0)).collect();
    run.input_means = (0..400)
        .map(|i| (5.0 + 0.01 * (i % 7) as f64, 0.0))
        .collect();
    let claim = (0.045, 0.055);
    assert_eq!(tier3_refusal("gpu", &t3, &run, 0, 0, claim), None);
    let e = tier3_refusal("gpu", &t3, &run, 12, 12, claim).expect("moments refuse");
    assert!(e.contains("error distributions disagree"), "{e}");
}
