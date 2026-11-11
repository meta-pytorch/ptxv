// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

//! Concolic expressions (symb.rs): every expression agrees with the simulator, on the captured
//! input and, where the path condition holds, on random other inputs.

mod cc_common;

use cc_common::*;
use ptxv_case::Case;
use ptxv_sim::machine::Machine;

/// Sweep (manual): the expressions of every fixture's good variant against the simulator.
/// `PTXV_CC_SWEEP=name1,name2` limits it.
#[test]
#[ignore]
fn sweep_fixtures_mismatches() {
    let only: Option<Vec<String>> = std::env::var("PTXV_CC_SWEEP")
        .ok()
        .map(|s| s.split(',').map(|x| x.to_string()).collect());
    let mut names: Vec<String> = std::fs::read_dir(root().join("fixtures"))
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("case.toml").exists())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    names.sort();
    let (mut mism, mut gaps) = (Vec::new(), Vec::new());
    for n in names {
        if only.as_ref().is_some_and(|o| !o.contains(&n)) {
            continue;
        }
        for front in ["ptx", "sass"] {
            let dir = root().join("fixtures").join(&n);
            let Ok(case) = Case::load(&dir) else { continue };
            let v = case
                .m
                .variants
                .first()
                .map(|v| v.id.clone())
                .unwrap_or("good".into());
            if case.m.cubin.is_some() && front == "ptx" {
                continue;
            }
            let Ok(m) = machine(&dir, &v, front) else {
                continue;
            };
            if m.threads.len() > 4096 {
                continue;
            }
            let t = std::time::Instant::now();
            let (cov, s) = ptxv_sim::coverage::analyze_with(&m, Some(Default::default()));
            let (sym, _) = s.unwrap();
            println!(
                "SWEEP {n} {front}: nodes={} inst={} mismatch={} prov_gap={} sites={} untaken={} atoms={} opaque={:?} {:.1}s",
                sym.ar.len(),
                sym.n_inst,
                sym.n_mismatch,
                sym.n_prov_gap,
                cov.sites,
                cov.untaken.len(),
                sym.path.values().map(|p| p.len()).sum::<usize>(),
                sym.n_opaque,
                t.elapsed().as_secs_f64()
            );
            for (_, x) in sym.mismatches.iter().take(4) {
                println!("    MISMATCH {x}");
            }
            for (_, x) in sym.prov_gaps.iter().take(4) {
                println!("    PROV-GAP {x}");
            }
            if sym.n_mismatch > 0 {
                mism.push(format!("{n}/{front}"));
            }
            if sym.n_prov_gap > 0 {
                gaps.push(format!("{n}/{front}"));
            }
        }
    }
    // A provenance gap (coverage.rs `Prov` clean, the expression with input leaves) is a miss
    // of the provenance, flagged here.
    assert!(
        mism.is_empty() && gaps.is_empty(),
        "mismatches: {mism:?}\nprovenance gaps: {gaps:?}"
    );
}

/// Debugging (manual): the path atoms of thread `PTXV_CC_T` (default 0) on a fixture.
#[test]
#[ignore]
fn dump_paths() {
    let name = std::env::var("PTXV_CC_FIX").unwrap_or("e_concolic".into());
    let front = std::env::var("PTXV_CC_FRONT").unwrap_or("ptx".into());
    let t: u32 = std::env::var("PTXV_CC_T")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(0);
    let m = match std::env::var("PTXV_CC_CASE") {
        Ok(dir) => machine(std::path::Path::new(&dir), "good", &front).unwrap(),
        Err(_) => fixture(&name, "good", &front),
    };
    let (cov, s) = ptxv_sim::coverage::analyze_with(&m, Some(Default::default()));
    let (sym, _) = s.unwrap();
    for (pc, s) in &sym.sites {
        println!(
            "site {pc} {}: {} / {}",
            m.k.describe(*pc as usize),
            s[0].len(),
            s[1].len()
        );
    }
    for u in &cov.untaken {
        let obs = sym
            .sites
            .get(&u.pc)
            .map_or(0, |s| s[!u.never as usize].len());
        println!("untaken {} never={} observations={obs}", u.at, u.never);
    }
    println!(
        "truncated paths: {}, threads with paths: {}",
        sym.truncated.len(),
        sym.path.len()
    );
    for a in sym.path.get(&t).cloned().unwrap_or_default() {
        println!(
            "{:?} {} {} = {}",
            a.kind,
            m.k.describe(a.pc as usize),
            sym.ar.show(a.cond, &|x| format!("a{x}"), 8),
            a.val
        );
    }
    for (_, x) in &sym.mismatches {
        println!("MISMATCH {x}");
    }
    if let Ok(r) = std::env::var("PTXV_CC_OPS") {
        let (a, b) = r.split_once('-').unwrap();
        for pc in a.parse::<usize>().unwrap()..b.parse::<usize>().unwrap() {
            println!("{pc}: {:?}", m.k.code[pc].op);
        }
    }
}

/// xorshift64*.
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_f491_4f6c_dd1d)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n as u64) as usize
    }
    fn pick<'a>(&mut self, v: &[&'a str]) -> &'a str {
        v[self.below(v.len())]
    }
}

/// A random straight-line PTX kernel: each thread loads four u32 and four f32 inputs, runs
/// `n` random integer / float / conversion statements, and stores every value it made.
fn random_kernel(rng: &mut Rng, n: usize) -> (String, usize) {
    let mut body = String::new();
    let (mut nr, mut nf, mut nh, mut nd, mut np) = (4usize, 4usize, 0usize, 0usize, 0usize);
    let mut stores: Vec<String> = Vec::new();
    for i in 0..4 {
        body += &format!("\tld.global.u32 %r{i}, [%rd4+{}];\n", 4 * i);
        body += &format!("\tld.global.f32 %f{i}, [%rd5+{}];\n", 4 * i);
    }
    for _ in 0..n {
        let r = |rng: &mut Rng, nr: usize| format!("%r{}", rng.below(nr));
        let f = |rng: &mut Rng, nf: usize| format!("%f{}", rng.below(nf));
        let imm = |rng: &mut Rng| match rng.below(4) {
            0 => format!("{}", rng.below(40)),
            1 => format!("0x{:x}", rng.next() as u32),
            2 => format!("{}", rng.below(4096)),
            _ => "1".to_string(),
        };
        let kind = rng.below(10);
        if kind < 5 {
            let d = format!("%r{nr}");
            let line = match rng.below(17) {
                0 => {
                    let op = rng.pick(&[
                        "add.u32",
                        "sub.u32",
                        "mul.lo.u32",
                        "mul.hi.u32",
                        "mul.hi.s32",
                        "and.b32",
                        "or.b32",
                        "xor.b32",
                        "min.u32",
                        "max.u32",
                        "min.s32",
                        "max.s32",
                    ]);
                    format!("{op} {d}, {}, {}", r(rng, nr), r(rng, nr))
                }
                1 => {
                    let op = rng.pick(&[
                        "add.u32",
                        "sub.u32",
                        "mul.lo.u32",
                        "and.b32",
                        "or.b32",
                        "xor.b32",
                        "min.s32",
                        "max.u32",
                    ]);
                    format!("{op} {d}, {}, {}", r(rng, nr), imm(rng))
                }
                2 => {
                    let op = rng.pick(&["shl.b32", "shr.u32", "shr.s32"]);
                    format!("{op} {d}, {}, {}", r(rng, nr), rng.below(40))
                }
                3 => {
                    let op = rng.pick(&["shl.b32", "shr.u32", "shr.s32"]);
                    format!("{op} {d}, {}, {}", r(rng, nr), r(rng, nr))
                }
                4 => format!(
                    "mad.lo.u32 {d}, {}, {}, {}",
                    r(rng, nr),
                    r(rng, nr),
                    r(rng, nr)
                ),
                5 => {
                    let op = rng.pick(&[
                        "not.b32", "neg.s32", "popc.b32", "brev.b32", "clz.b32", "abs.s32",
                    ]);
                    format!("{op} {d}, {}", r(rng, nr))
                }
                6 => format!(
                    "prmt.b32 {d}, {}, {}, 0x{:x}",
                    r(rng, nr),
                    r(rng, nr),
                    rng.next() as u16
                ),
                7 => {
                    let op = rng.pick(&["bfe.u32", "bfe.s32"]);
                    format!(
                        "{op} {d}, {}, {}, {}",
                        r(rng, nr),
                        rng.below(34),
                        rng.below(34)
                    )
                }
                8 | 9 => {
                    let p = format!("%p{np}");
                    np += 1;
                    let c = rng.pick(&[
                        "setp.lt.u32",
                        "setp.ge.s32",
                        "setp.eq.u32",
                        "setp.ne.u32",
                        "setp.gt.s32",
                        "setp.le.u32",
                    ]);
                    format!(
                        "{c} {p}, {}, {};\n\tselp.b32 {d}, {}, {}, {p}",
                        r(rng, nr),
                        r(rng, nr),
                        r(rng, nr),
                        r(rng, nr)
                    )
                }
                10 => {
                    let op = rng.pick(&[
                        "cvt.rzi.s32.f32",
                        "cvt.rni.s32.f32",
                        "cvt.rmi.u32.f32",
                        "cvt.rpi.s32.f32",
                    ]);
                    format!("{op} {d}, {}", f(rng, nf))
                }
                11 => format!("mov.b32 {d}, {}", f(rng, nf)),
                12 => {
                    let w = format!("%rd{}", 10 + nd);
                    let w2 = format!("%rd{}", 11 + nd);
                    nd += 2;
                    let op = rng.pick(&["mul.wide.u32", "mul.wide.s32"]);
                    format!(
                        "{op} {w}, {}, {};\n\tshr.u64 {w2}, {w}, {};\n\tcvt.u32.u64 {d}, {w2}",
                        r(rng, nr),
                        r(rng, nr),
                        rng.below(40)
                    )
                }
                13 => format!(
                    "mul.lo.u32 {d}, {}, 0x{:x}",
                    r(rng, nr),
                    rng.next() as u32 | 1
                ),
                14 => format!("add.u32 {d}, {}, {}", r(rng, nr), r(rng, nr)),
                15 => {
                    let h = format!("%h{nh}");
                    nh += 1;
                    format!("cvt.u16.u32 {h}, {};\n\tcvt.s32.s16 {d}, {h}", r(rng, nr))
                }
                _ => format!("add.u32 {d}, {}, {}", r(rng, nr), imm(rng)),
            };
            body += &format!("\t{line};\n");
            stores.push(d);
            nr += 1;
        } else {
            let d = format!("%f{nf}");
            let mut extra = None;
            let line = match rng.below(15) {
                0 => {
                    let op = rng.pick(&[
                        "add.rn.f32",
                        "add.rz.f32",
                        "add.rm.f32",
                        "add.rp.f32",
                        "sub.rn.f32",
                        "mul.rn.f32",
                        "mul.rz.f32",
                        "add.rn.ftz.f32",
                        "mul.rn.ftz.f32",
                        "min.f32",
                        "max.f32",
                        "add.rn.sat.f32",
                    ]);
                    format!("{op} {d}, {}, {}", f(rng, nf), f(rng, nf))
                }
                1 => {
                    let op = rng.pick(&["fma.rn.f32", "fma.rz.f32", "fma.rn.ftz.f32"]);
                    format!("{op} {d}, {}, {}, {}", f(rng, nf), f(rng, nf), f(rng, nf))
                }
                2 => {
                    let op = rng.pick(&[
                        "abs.f32",
                        "neg.f32",
                        "ex2.approx.ftz.f32",
                        "rcp.approx.ftz.f32",
                        "sqrt.approx.f32",
                        "lg2.approx.f32",
                        "rsqrt.approx.f32",
                        "cvt.rni.f32.f32",
                        "cvt.rzi.f32.f32",
                    ]);
                    format!("{op} {d}, {}", f(rng, nf))
                }
                3 => {
                    let op = rng.pick(&["cvt.rn.f32.s32", "cvt.rn.f32.u32", "cvt.rz.f32.s32"]);
                    format!("{op} {d}, {}", r(rng, nr))
                }
                4 => format!("mov.b32 {d}, {}", r(rng, nr)),
                5 | 6 => {
                    let p = format!("%p{np}");
                    np += 1;
                    let c = rng.pick(&[
                        "setp.lt.f32",
                        "setp.ge.f32",
                        "setp.equ.f32",
                        "setp.neu.f32",
                        "setp.gtu.f32",
                        "setp.num.f32",
                    ]);
                    format!(
                        "{c} {p}, {}, {};\n\tselp.f32 {d}, {}, {}, {p}",
                        f(rng, nf),
                        f(rng, nf),
                        f(rng, nf),
                        f(rng, nf)
                    )
                }
                7 => {
                    let h = format!("%h{nh}");
                    nh += 1;
                    let (to, from) = if rng.below(2) == 0 {
                        ("f16", "f16")
                    } else {
                        ("bf16", "bf16")
                    };
                    format!(
                        "cvt.rn.{to}.f32 {h}, {};\n\tcvt.f32.{from} {d}, {h}",
                        f(rng, nf)
                    )
                }
                8 => {
                    let a = format!("%fd{nd}");
                    let b = format!("%fd{}", nd + 1);
                    nd += 2;
                    format!(
                        "cvt.f64.f32 {a}, {};\n\tfma.rn.f64 {b}, {a}, {a}, {a};\n\tcvt.rn.f32.f64 {d}, {b}",
                        f(rng, nf)
                    )
                }
                9 => format!(
                    "mul.rn.f32 {d}, {}, 0f{:08X}",
                    f(rng, nf),
                    (rng.next() as u32) & 0xbfff_ffff
                ),
                10 => format!(
                    "add.rn.f32 {d}, {}, 0f{:08X}",
                    f(rng, nf),
                    (rng.next() as u32) & 0xbfff_ffff
                ),
                11 => {
                    let h = format!("%h{nh}");
                    let h2 = format!("%h{}", nh + 1);
                    nh += 2;
                    format!(
                        "cvt.rn.f16.f32 {h}, {};\n\tfma.rn.f16 {h2}, {h}, {h}, {h};\n\tcvt.f32.f16 {d}, {h2}",
                        f(rng, nf)
                    )
                }
                12 => format!("min.NaN.f32 {d}, {}, {}", f(rng, nf), f(rng, nf)),
                13 => {
                    // Packed f32x2 FMA (SASS FFMA2): both lanes stored.
                    let (a, b, c) = (
                        format!("%rd{}", 10 + nd),
                        format!("%rd{}", 11 + nd),
                        format!("%rd{}", 12 + nd),
                    );
                    nd += 3;
                    let t = format!("%f{}", nf + 1);
                    extra = Some(t.clone());
                    let op = rng.pick(&["fma.rn.f32x2", "fma.rz.f32x2"]);
                    format!(
                        "mov.b64 {a}, {{{}, {}}};\n\tmov.b64 {b}, {{{}, {}}};\n\t{op} {c}, {a}, {b}, {a};\n\tmov.b64 {{{d}, {t}}}, {c}",
                        f(rng, nf),
                        f(rng, nf),
                        f(rng, nf),
                        f(rng, nf)
                    )
                }
                _ => format!("div.rn.f32 {d}, {}, {}", f(rng, nf), f(rng, nf)),
            };
            body += &format!("\t{line};\n");
            stores.push(d);
            nf += 1;
            if let Some(t) = extra {
                stores.push(t);
                nf += 1;
            }
        }
    }
    for (j, x) in stores.iter().enumerate() {
        let ty = if x.starts_with("%f") { "f32" } else { "u32" };
        body += &format!("\tst.global.{ty} [%rd6+{}], {x};\n", 4 * j);
    }
    let k = stores.len();
    let ptx = format!(
        ".version 9.0\n.target sm_103a\n.address_size 64\n\n.visible .entry rk(.param .u64 pu, .param .u64 pf, .param .u64 po)\n.reqntid 64\n{{\n\t.reg .pred %p<{}>;\n\t.reg .b32 %r<{}>;\n\t.reg .f32 %f<{}>;\n\t.reg .b16 %h<{}>;\n\t.reg .b64 %rd<{}>;\n\t.reg .f64 %fd<{}>;\n\tld.param.u64 %rd1, [pu];\n\tld.param.u64 %rd2, [pf];\n\tld.param.u64 %rd3, [po];\n\tcvta.to.global.u64 %rd1, %rd1;\n\tcvta.to.global.u64 %rd2, %rd2;\n\tcvta.to.global.u64 %rd3, %rd3;\n\tmov.u32 %r100, %tid.x;\n\tmul.wide.u32 %rd7, %r100, 16;\n\tadd.s64 %rd4, %rd1, %rd7;\n\tadd.s64 %rd5, %rd2, %rd7;\n\tmul.lo.u32 %r101, %r100, {};\n\tmul.wide.u32 %rd8, %r101, 4;\n\tadd.s64 %rd6, %rd3, %rd8;\n{body}\tret;\n}}\n",
        np + 1,
        nr.max(102) + 1,
        nf + 1,
        nh + 1,
        nd + 12,
        nd + 2,
        k
    );
    (ptx, k)
}

fn write_case(dir: &std::path::Path, ptx: &str, k: usize, seed: u64) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join("kernel.ptx"), ptx).unwrap();
    let toml = format!(
        "name = \"rk\"\ndescription = \"random\"\nptx = \"kernel.ptx\"\nentry = \"rk\"\ngrid = [1, 1, 1]\nblock = [64, 1, 1]\noutputs = [\"out\"]\n\n[[buffers]]\nname = \"u\"\ndtype = \"u32\"\nshape = [256]\ninit = \"bits\"\nseed = {seed}\n\n[[buffers]]\nname = \"f\"\ndtype = \"f32\"\nshape = [256]\ninit = \"normal\"\nseed = {seed}\n\n[[buffers]]\nname = \"out\"\ndtype = \"u32\"\nshape = [{}]\ninit = \"zeros\"\n\n[[params]]\nkind = \"ptr\"\nbuffer = \"u\"\n\n[[params]]\nkind = \"ptr\"\nbuffer = \"f\"\n\n[[params]]\nkind = \"ptr\"\nbuffer = \"out\"\n\n[[variants]]\nid = \"good\"\nexpect = [\"clean\"]\n",
        64 * k
    );
    std::fs::write(dir.join("case.toml"), toml).unwrap();
}

/// Random input bytes for allocations `u` (bit patterns and small integers) and `f` (floats
/// of several magnitudes and specials): (allocation, offset, bytes, value) writes.
fn random_input(m: &Machine, rng: &mut Rng) -> Vec<(usize, u64, u8, u64)> {
    let mut w = Vec::new();
    let ua = m.launch.allocs.iter().position(|a| a.name == "u").unwrap();
    let fa = m.launch.allocs.iter().position(|a| a.name == "f").unwrap();
    for i in 0..256u64 {
        let u = match rng.below(4) {
            0 => rng.next() & 0xffff_ffff,
            1 => rng.next() % 64,
            2 => (rng.next() as u32 as i32 % 1000) as u32 as u64,
            _ => [0u64, 1, 0xffff_ffff, 0x8000_0000, 0x7fff_ffff, 31, 32, 33][rng.below(8)],
        };
        w.push((ua, 4 * i, 4u8, u));
        let fv: u32 = match rng.below(5) {
            0 => ((rng.next() % 2000) as f32 / 100.0 - 10.0).to_bits(),
            1 => (rng.next() as u32) & 0x7fff_ffff | ((rng.below(2) as u32) << 31),
            2 => [
                0u32,
                0x8000_0000,
                0x7f80_0000,
                0xff80_0000,
                0x7fc0_0000,
                0x0000_0001,
                0x0080_0000,
                0x3f80_0000,
            ][rng.below(8)],
            3 => (((rng.next() % 200) as f32 - 100.0) * 1e6).to_bits(),
            _ => ((rng.next() % 1000) as f32 / 1e4).to_bits(),
        };
        w.push((fa, 4 * i, 4u8, fv as u64));
    }
    w
}

/// Property: for random straight-line kernels (integer, float, conversion and select
/// statements), every stored value's expression, evaluated at a fresh random input, equals
/// the simulator's value on that input, bitwise (both fronts). `PTXV_CC_PROP_N` programs.
#[test]
fn expressions_equal_simulator_on_random_inputs() {
    let n: u64 = std::env::var("PTXV_CC_PROP_N")
        .ok()
        .and_then(|x| x.parse().ok())
        .unwrap_or(12);
    let mut checked = 0usize;
    let mut skipped_opaque = 0usize;
    let mut skipped_path = 0usize;
    let mut opq: std::collections::BTreeMap<String, usize> = Default::default();
    for seed in 1..=n {
        let mut rng = Rng(0x9e37_79b9_7f4a_7c15 ^ seed.wrapping_mul(0x1000_0000_01b3));
        let (ptx, k) = random_kernel(&mut rng, 14);
        let dir = scratch_dir("rk", &format!("{seed}"));
        write_case(&dir, &ptx, k, seed);
        for front in ["ptx", "sass"] {
            let m0 = match machine(&dir, "good", front) {
                Ok(m) => m,
                // A SASS form the lifter does not take (not the expressions' concern).
                Err(e) if front == "sass" => {
                    println!(
                        "seed {seed} sass: skipped: {}",
                        format!("{e:#}").lines().next().unwrap_or("")
                    );
                    continue;
                }
                Err(e) => panic!("seed {seed} {front}: {e:#}\n{ptx}"),
            };
            let (_, s) = ptxv_sim::coverage::analyze_with(&m0, Some(Default::default()));
            let (mut sym, mfin) = s.unwrap();
            assert_eq!(
                sym.n_mismatch, 0,
                "seed {seed} {front}: {:#?}\n{ptx}",
                sym.mismatches
            );
            let oa = m0
                .launch
                .allocs
                .iter()
                .position(|a| a.name == "out")
                .unwrap() as u32;
            let out0 = mfin.outputs();
            // The expressions of every stored word.
            let mut es = Vec::new();
            for t in 0..64usize {
                for j in 0..k {
                    let off = ((t * k + j) * 4) as u64;
                    let v = u32::from_le_bytes(
                        out0[off as usize..off as usize + 4].try_into().unwrap(),
                    ) as u64;
                    let e = sym.global_expr(&m0, oa, off, 4, v);
                    assert_eq!(
                        sym.ar.conc[e as usize], v,
                        "seed {seed} {front}: t{t} out{j}"
                    );
                    es.push((off, e));
                }
            }
            let mut ev = ptxv_sim::symb::Evaluator::new(&sym.ar, &mfin);
            for trial in 0..3 {
                let writes = random_input(&m0, &mut rng);
                let mu = ptxv_sim::coverage::Mutation {
                    pc: 0,
                    never: false,
                    writes: writes.clone(),
                    label: String::new(),
                    within_invariants: false,
                };
                let m1 = ptxv_sim::coverage::apply(&m0, &mu);
                let (m1f, _, ok) = ptxv_sim::explore::run_once(m1.clone());
                assert!(ok);
                let out1 = m1f.outputs();
                // Every input leaf's value on the new input.
                let mut env = std::collections::HashMap::new();
                for (l, node) in sym.ar.nodes.iter().enumerate() {
                    if let ptxv_sim::symb::Node::Input {
                        alloc,
                        off: lo,
                        bytes,
                    } = *node
                    {
                        let init = &m1.launch.allocs[alloc as usize].init;
                        let mut v = 0u64;
                        for b in 0..bytes as usize {
                            v |= (init[lo as usize + b] as u64) << (8 * b);
                        }
                        env.insert(l as u32, v);
                    }
                }
                for (i, &(off, e)) in es.iter().enumerate() {
                    let t = i / k;
                    // The expression holds where the thread's control atoms hold (SASS
                    // division sequences branch on FCHK); guards of register operations are
                    // data (`ite(guard, new, old)`, the guard-false side modelled), not skipped.
                    let path = sym.path.get(&(t as u32)).cloned().unwrap_or_default();
                    if path
                        .iter()
                        .filter(|a| a.kind == ptxv_sim::symb::AtomKind::Guard)
                        .any(|a| ev.eval(a.cond, &env).map(|v| v & 1 == 1) != Some(a.val))
                    {
                        skipped_path += 1;
                        continue;
                    }
                    let leaves = sym.ar.leaves(e);
                    if let Some(&l) = leaves.iter().find(|&&l| {
                        !matches!(
                            sym.ar.nodes[l as usize],
                            ptxv_sim::symb::Node::Input { .. } | ptxv_sim::symb::Node::Sreg { .. }
                        )
                    }) {
                        if let ptxv_sim::symb::Node::Opaque { id, .. } = sym.ar.nodes[l as usize] {
                            *opq.entry(sym.ar.opaque[id as usize].why.clone())
                                .or_insert(0usize) += 1;
                        }
                        skipped_opaque += 1;
                        continue;
                    }
                    let got = ev.eval(e, &env);
                    let want = u32::from_le_bytes(
                        out1[off as usize..off as usize + 4].try_into().unwrap(),
                    ) as u64;
                    if got != Some(want) {
                        for a in &path {
                            println!(
                                "  atom {:?} {} {} = {} now {:?}",
                                a.kind,
                                m0.k.describe(a.pc as usize),
                                sym.ar.show(a.cond, &|x| format!("a{x}"), 6),
                                a.val,
                                ev.eval(a.cond, &env)
                            );
                        }
                    }
                    assert_eq!(
                        got,
                        Some(want),
                        "seed {seed} {front} trial {trial}: word {} expr {}\n{ptx}",
                        off / 4,
                        sym.ar.show(e, &|a| format!("a{a}"), 12)
                    );
                    checked += 1;
                }
            }
        }
    }
    println!(
        "checked {checked} values, {skipped_opaque} with opaque leaves {opq:?}, {skipped_path} off the path"
    );
    assert!(checked > 0);
}

/// Every expression built for a destination register on these fixtures (integer, f64, half,
/// conversions, MUFU, shuffles and atomics, TMA, cp.async, local memory) agrees with the
/// simulator on the captured input, with no fallback (`sweep_fixtures_mismatches` runs all).
#[test]
fn expressions_match_simulator_on_fixtures() {
    let names = [
        "s_alu",
        "s_alu2",
        "s_cvt",
        "s_f64",
        "s_shf",
        "s_hfma2_bf16",
        "s_mufu",
        "s_ptx_packed",
        "s_selp64",
        "s_bmsk",
        "s_dp2a",
        "s_f2i",
        "s_gather4",
        "r0_tma_sum",
        "e_vec_regs",
        "c_online_ok",
        "c_fmax_atomic",
        "c_silu",
        "c_softcap",
        "e_sentinel_fill",
        "e_concolic",
        "c_triton_softmax_online",
        "s_routing_forms",
    ];
    let mut n = 0;
    for name in names {
        let dir = root().join("fixtures").join(name);
        let case = Case::load(&dir).unwrap();
        let v = case
            .m
            .variants
            .first()
            .map(|v| v.id.clone())
            .unwrap_or("good".into());
        for front in ["ptx", "sass"] {
            if case.m.cubin.is_some() && front == "ptx" {
                continue;
            }
            let Ok(m) = machine(&dir, &v, front) else {
                continue;
            };
            let (_, s) = ptxv_sim::coverage::analyze_with(&m, Some(Default::default()));
            let (sym, _) = s.unwrap();
            assert_eq!(sym.n_mismatch, 0, "{name}/{front}: {:#?}", sym.mismatches);
            n += 1;
        }
    }
    assert!(n >= 30, "{n}");
}
