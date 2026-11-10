/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `ptxv tpu ...`: the tpu7x TensorCore target (crates/tpu). A kernel directory is one of the
//! probe corpus layout (`<out>/k/<name>/meta.json` with inputs and hardware outputs, listings in
//! `<out>/llo/<name>/`, optional executable `<out>/rt/<name>.bin` for branch delays).

use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::Result;
use anyhow::bail;
use clap::Subcommand;
use ptxv_tpu::corpus::KernelDir;
use ptxv_tpu::corpus::run_kernel;
use ptxv_tpu::machine::DmaPolicy;
use ptxv_tpu::model::TpuModel;
use ptxv_tpu::num::Numerics;
use ptxv_tpu::num::PLACEHOLDER;
use ptxv_tpu_ir::Front;

use crate::jsonout;

#[derive(Subcommand)]
pub enum TpuCmd {
    /// Explore every DMA-completion and remote-delivery order: deadlock, race, exit-check,
    /// bounds-check findings.
    Check {
        /// A kernel directory, or a `*-final_bundles.txt` listing (zero inputs).
        dir: PathBuf,
        /// Listing mode: serialized executable for the branch delays (default: `exe.rt.bin`
        /// next to the listing directory).
        #[arg(long)]
        exe: Option<PathBuf>,
        /// Listing mode: devices for programs that signal other cores (default: `device_count`
        /// of a meta.json next to the listing directory, else 2).
        #[arg(long)]
        devices: Option<usize>,
        /// Wall-clock budget per front, seconds.
        #[arg(long)]
        time_limit: Option<f64>,
        /// final, original, binary (machine code of the executable), both (final + original)
        /// or all
        #[arg(long, default_value = "final")]
        front: String,
        #[arg(long, default_value_t = 1_000_000)]
        max_states: usize,
        #[arg(long, default_value_t = 8)]
        workers: usize,
        #[arg(long)]
        json: bool,
    },
    /// Run once (deterministic DMA policy) and compare the outputs with the hardware's.
    Sim {
        dir: PathBuf,
        #[arg(long, default_value = "both")]
        front: String,
        /// eager, lazy or random
        #[arg(long, default_value = "eager")]
        dma: String,
        #[arg(long, default_value_t = 0)]
        seed: u64,
        #[arg(long)]
        json: bool,
    },
    /// Disassemble the machine code of a serialized executable (either libtpu build).
    Dis {
        exe: PathBuf,
        #[arg(long, default_value_t = 0)]
        from: usize,
        #[arg(long)]
        count: Option<usize>,
        /// A kernel's `*-final_bundles.txt`: place it in the image, print its bundles with the
        /// listing ops the decode does not match.
        #[arg(long)]
        listing: Option<PathBuf>,
        /// With --listing: print the rendered listing-syntax text the binary front lifts.
        #[arg(long)]
        render: bool,
    },
    /// Lift listings (final_bundles or original) and report unlowered opcodes.
    Lift {
        listings: Vec<PathBuf>,
        #[arg(long)]
        json: bool,
    },
}

/// A front: one of the listing fronts or the machine code (binary front).
#[derive(Clone, Copy, Debug)]
enum Sel {
    Listing(Front),
    Binary,
}

impl Sel {
    fn label(self) -> String {
        match self {
            Sel::Listing(f) => format!("{f:?}").to_lowercase(),
            Sel::Binary => "binary".into(),
        }
    }

    fn lift(self, kd: &KernelDir) -> Result<ptxv_tpu_ir::Kernel> {
        Ok(match self {
            Sel::Listing(f) => kd.lift(f),
            Sel::Binary => ptxv_tpu::binlift::lift_kernel_dir(kd)?,
        })
    }
}

/// Numerics models: the calibrated tools/tpunum.py models (default) or, with
/// `PTXV_TPU_NUMERICS=placeholder`, the tools/llosim.py placeholders (differential oracle).
fn numerics() -> &'static dyn Numerics {
    match std::env::var("PTXV_TPU_NUMERICS").as_deref() {
        Ok("placeholder") => &PLACEHOLDER,
        _ => ptxv_tpu::num::calibrated(),
    }
}

fn fronts(s: &str) -> Result<Vec<Sel>> {
    use Sel::*;
    Ok(match s {
        "final" => vec![Listing(Front::Final)],
        "original" => vec![Listing(Front::Original)],
        "binary" => vec![Binary],
        "both" => vec![Listing(Front::Final), Listing(Front::Original)],
        "all" => vec![Listing(Front::Final), Listing(Front::Original), Binary],
        _ => bail!("--front {s}: final, original, binary, both or all"),
    })
}

pub fn run(cmd: TpuCmd) -> Result<()> {
    match cmd {
        TpuCmd::Check {
            dir,
            exe,
            devices,
            time_limit,
            front,
            max_states,
            workers,
            json,
        } => {
            let kd = if dir.is_file() {
                let up = dir.parent().and_then(Path::parent).map(Path::to_path_buf);
                let n = devices.or_else(|| {
                    let m: serde_json::Value = serde_json::from_str(
                        &std::fs::read_to_string(up.as_ref()?.join("meta.json")).ok()?,
                    )
                    .ok()?;
                    m["device_count"].as_u64().map(|x| x as usize)
                });
                let exe = exe.or_else(|| up.map(|p| p.join("exe.rt.bin")).filter(|p| p.exists()));
                KernelDir::from_listing(&dir, exe, n.unwrap_or(2))?
            } else {
                KernelDir::open(&dir)?
            };
            check(&kd, &front, max_states, workers, time_limit, json)
        }
        TpuCmd::Sim {
            dir,
            front,
            dma,
            seed,
            json,
        } => sim(&dir, &front, &dma, seed, json),
        TpuCmd::Dis {
            exe,
            from,
            count,
            listing,
            render,
        } => dis(&exe, from, count, listing.as_deref(), render),
        TpuCmd::Lift { listings, json } => lift(&listings, json),
    }
}

fn check(
    kd: &KernelDir,
    front: &str,
    max_states: usize,
    workers: usize,
    time_limit: Option<f64>,
    json: bool,
) -> Result<()> {
    let out = json.then(|| jsonout::begin("tpu check"));
    let mut docs = Vec::new();
    let mut dirty = false;
    for f in fronts(front)? {
        let k = Arc::new(f.lift(kd)?);
        println!(
            "== {} ({}): {} bundles, {} core(s), branch delays {}{}",
            kd.name,
            f.label(),
            k.bundles.len(),
            kd.ncores(),
            k.delay_source,
            if k.unlowered.is_empty() {
                String::new()
            } else {
                format!(", {} unlowered ops", k.unlowered.len())
            }
        );
        for n in &k.notes {
            println!("   note: {n}");
        }
        let m = TpuModel::new(kd.machine(k.clone(), numerics())?);
        let v = ptxv_sim::model::explore_until(
            m,
            max_states,
            workers,
            time_limit.map(std::time::Duration::from_secs_f64),
        );
        println!(
            "   states {} transitions {} terminal {} distinct outputs {} {} ({:.2}s)",
            v.states,
            v.transitions,
            v.terminal_states,
            v.distinct_outputs,
            if v.complete { "complete" } else { "TRUNCATED" },
            v.seconds
        );
        for x in &v.findings {
            println!("   {} {}: {}", x.kind.to_uppercase(), x.sig, x.msg);
            for t in &x.trace {
                println!("      {t}");
            }
        }
        println!("   assumptions: {}", v.assumptions.join(", "));
        let verdict = if !v.findings.is_empty() {
            "findings"
        } else if !v.complete {
            "incomplete"
        } else {
            "clean"
        };
        println!("   verdict: {verdict}");
        dirty |= verdict != "clean";
        docs.push(serde_json::json!({
            "front": f.label(),
            "verdict": verdict,
            "states": v.states,
            "transitions": v.transitions,
            "complete": v.complete,
            "terminal_states": v.terminal_states,
            "distinct_outputs": v.distinct_outputs,
            "findings": v.findings.iter().map(|x| serde_json::json!({"kind": x.kind, "sig": x.sig, "msg": x.msg, "trace": x.trace})).collect::<Vec<_>>(),
            "assumptions": v.assumptions,
            "delay_source": k.delay_source,
            "bundles": k.bundles.len(),
            "cores": kd.ncores(),
            "unlowered": ptxv_tpu::lift::lift_report(&k)
                .iter()
                .map(|(mn, c, why)| serde_json::json!({"mnemonic": mn, "count": c, "why": why}))
                .collect::<Vec<_>>(),
            "notes": k.notes,
            "seconds": v.seconds,
        }));
    }
    if let Some(o) = out {
        o.emit(
            "tpu check",
            serde_json::json!({"kernel": kd.name, "results": docs}),
        );
    }
    if dirty {
        std::process::exit(1);
    }
    Ok(())
}

fn sim(dir: &Path, front: &str, dma: &str, seed: u64, json: bool) -> Result<()> {
    let out = json.then(|| jsonout::begin("tpu sim"));
    let policy = match dma {
        "eager" => DmaPolicy::Eager,
        "lazy" => DmaPolicy::Lazy,
        "random" => DmaPolicy::Random(seed),
        _ => bail!("--dma {dma}: eager, lazy or random"),
    };
    let kd = KernelDir::open(dir)?;
    let mut docs = Vec::new();
    let mut all_exact = true;
    for f in fronts(front)? {
        let t0 = Instant::now();
        let r = match f {
            Sel::Listing(f) => run_kernel(&kd, f, policy, numerics())?,
            Sel::Binary => ptxv_tpu::binlift::run_binary(&kd, policy, numerics())?,
        };
        let mm: Vec<String> = r
            .outputs
            .iter()
            .map(|o| format!("{}{}", o.mismatch, if o.nondet { "?" } else { "" }))
            .collect();
        println!(
            "{} {:8} {:8} exact={} mismatches=[{}] steps={} {:.3}s {}",
            kd.name,
            r.front,
            r.status,
            r.exact.map_or("-".into(), |e| e.to_string()),
            mm.join(","),
            r.steps,
            t0.elapsed().as_secs_f64(),
            r.error
        );
        all_exact &= r.exact == Some(true);
        docs.push(serde_json::to_value(&r)?);
    }
    if let Some(o) = out {
        o.emit(
            "tpu sim",
            serde_json::json!({"kernel": kd.name, "results": docs}),
        );
    }
    if !all_exact {
        std::process::exit(1);
    }
    Ok(())
}

fn lift(listings: &[PathBuf], json: bool) -> Result<()> {
    let out = json.then(|| jsonout::begin("tpu lift"));
    let mut docs = Vec::new();
    for p in listings {
        let text = std::fs::read_to_string(p)?;
        let name = p.file_name().and_then(|f| f.to_str()).unwrap_or("?");
        let t0 = Instant::now();
        let k = if name.contains("original") {
            ptxv_tpu::lift::lift_original(name, &text)
        } else {
            // allocation header from the matching original listing, when present
            let orig = name
                .rfind("-final_bundles.txt")
                .and_then(|i| name.get(..i.saturating_sub(3)))
                .map(|stem| p.with_file_name(format!("{stem}-00-original.txt")));
            let otext = orig
                .and_then(|o| std::fs::read_to_string(o).ok())
                .unwrap_or_default();
            ptxv_tpu::lift::lift_final(name, &text, &otext)
        };
        let secs = t0.elapsed().as_secs_f64();
        let rep = ptxv_tpu::lift::lift_report(&k);
        let n = k.n_insts();
        let un: usize = rep.iter().map(|r| r.1).sum();
        println!(
            "{name}: {} bundles, {n} ops, {} unlowered ({:.1}% lowered), {:.3}s",
            k.bundles.len(),
            un,
            100.0 * (n - un.min(n)) as f64 / n.max(1) as f64,
            secs
        );
        for (mn, c, why) in &rep {
            println!("   {c:6} {mn}  ({why})");
        }
        docs.push(serde_json::json!({
            "listing": p.display().to_string(),
            "bundles": k.bundles.len(),
            "ops": n,
            "unlowered": rep.iter().map(|(mn, c, why)| serde_json::json!({"mnemonic": mn, "count": c, "why": why})).collect::<Vec<_>>(),
            "seconds": secs,
        }));
    }
    if let Some(o) = out {
        o.emit("tpu lift", serde_json::json!({"listings": docs}));
    }
    Ok(())
}

fn dis(
    exe: &Path,
    from: usize,
    count: Option<usize>,
    listing: Option<&Path>,
    render: bool,
) -> Result<()> {
    use ptxv_tpu::binlift;
    use ptxv_tpu::dis;
    let bytes = std::fs::read(exe)?;
    let Some(lp) = listing else {
        let img = ptxv_tpu::exe::code_image(&bytes)?;
        let n = count.unwrap_or(img.len().saturating_sub(from));
        for (k, raw) in img.iter().enumerate().skip(from).take(n) {
            let ops: Vec<String> = dis::decode(raw).iter().map(dis::fmt).collect();
            println!("{k:5}: {{ {} }}", ops.join("  ;;  "));
        }
        return Ok(());
    };
    let dir = lp.parent().ok_or_else(|| anyhow::anyhow!("listing dir"))?;
    let name = ptxv_tpu::exe::listing_name(lp)
        .ok_or_else(|| anyhow::anyhow!("not a final_bundles listing"))?;
    if render {
        let f = lp.to_string_lossy().to_string();
        let cut = f.rfind("-final_bundles.txt").expect("suffix");
        let otext = std::fs::read_to_string(format!("{}-00-original.txt", &f[..cut - 3]))
            .unwrap_or_default();
        let ftext = std::fs::read_to_string(lp)?;
        let ctx = binlift::ListingCtx {
            llo_dir: dir,
            kernel: &name,
            final_text: &ftext,
            original_text: &otext,
        };
        let (text, _, st) = binlift::binary_listing(&bytes, &ctx)?;
        print!("{text}");
        eprintln!("{st:?}");
        return Ok(());
    }
    let img = ptxv_tpu::exe::code_image(&bytes)?;
    let placed = binlift::place_kernel(&img, dir, &name)?;
    let (mut nops, mut nok) = (0, 0);
    let n = count.unwrap_or(placed.len().saturating_sub(from));
    for p in placed.iter().skip(from).take(n) {
        let ops: Vec<String> = dis::decode(&p.raw).iter().map(dis::fmt).collect();
        println!("{:#6x}: {{ {} }}", p.bundle.addr, ops.join("  ;;  "));
        let (t, ok, _, miss) = dis::score_bundle(&p.raw, &p.bundle.ops);
        nops += t;
        nok += ok;
        for m in miss {
            println!("        !! listing op not matched: {m}");
        }
    }
    println!("{name}: {nok}/{nops} listing ops matched by the decode");
    Ok(())
}
