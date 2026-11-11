/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Scoreboard mutation sweep. Every scoreboard bit ptxas set in the good binary is removed in
//! turn (one wait-mask bit, or one write/read scoreboard). Each one-bit mutant goes through the
//! static scoreboard check and through the GPU. This validates the control-bit model against
//! silicon: every mutant the GPU computes wrong must be flagged ("missed" must be empty).
//! Flagged mutants that still compute correctly are hazards the hardware happened to win.

use std::path::Path;
use std::path::PathBuf;
use std::sync::Mutex;

use anyhow::Result;
use ptxv_case::Case;
use ptxv_ir::CtrlPatch;
use ptxv_sass::scoreboard::Ctrl;
use ptxv_sass::scoreboard::patch_cubin;
use serde::Serialize;

use crate::HwRow;
use crate::cmd_num;
use crate::hw_run_cubin;
use crate::load;
use crate::out_dir;

#[derive(Serialize, Clone)]
struct Row {
    id: String,
    addr: u32,
    inst: String,
    /// "wait" (one wait bit cleared), "wbar" or "rbar" (scoreboard removed).
    field: String,
    sb: u8,
    flagged: bool,
    findings: Vec<String>,
    hw: Option<HwRow>,
    /// "caught" | "missed" | "hazard (hw won)" | "benign"
    class: String,
}

pub fn run(case: &Case, runs: usize, timeout: f64, out: &Option<PathBuf>) -> Result<()> {
    let dir = out_dir(case, out)?.join("sb_sweep");
    std::fs::create_dir_all(&dir)?;
    let (_, expected) = cmd_num(case, "good", out)?;
    let exp_path = dir.join("expected.bin");
    std::fs::write(&exp_path, &expected)?;
    let (_, good) = load(case, "good", "sass", &dir)?;
    let img = std::fs::read(&good)?;
    let dis = ptxv_sass::run_tool("nvdisasm", &["-c", "-hex", good.to_str().unwrap()])?;
    let (lines, _) = ptxv_sass::parse_disasm(&dis, &case.m.entry)?;

    let mut rows = Vec::new();
    for l in &lines {
        let c = Ctrl::decode(l.hi.unwrap());
        let mut muts: Vec<(String, u8, CtrlPatch)> = Vec::new();
        let base = CtrlPatch {
            at: format!("@{:#06x}", l.addr),
            wait: None,
            wbar: None,
            rbar: None,
            stall: None,
            ..Default::default()
        };
        for k in 0..6 {
            if c.wait >> k & 1 == 1 {
                muts.push((
                    "wait".into(),
                    k,
                    CtrlPatch {
                        wait: Some(c.wait & !(1 << k)),
                        ..base.clone()
                    },
                ));
            }
        }
        if let Some(k) = c.wbar {
            muts.push((
                "wbar".into(),
                k,
                CtrlPatch {
                    wbar: Some(7),
                    ..base.clone()
                },
            ));
        }
        if let Some(k) = c.rbar {
            muts.push((
                "rbar".into(),
                k,
                CtrlPatch {
                    rbar: Some(7),
                    ..base.clone()
                },
            ));
        }
        for (field, sb, p) in muts {
            let id = format!("{:04x}-{field}-SB{sb}", l.addr);
            let mut m = img.clone();
            patch_cubin(&mut m, &dis, &case.m.entry, &[p])?;
            let path = dir.join(format!("{id}.cubin"));
            std::fs::write(&path, &m)?;
            let t0 = std::time::Instant::now();
            let k = ptxv_sass::load_cubin(&path, &case.m.entry)?;
            if t0.elapsed().as_secs_f64() > 2.0 {
                println!(
                    "  {id}: static check took {:.1}s",
                    t0.elapsed().as_secs_f64()
                );
            }
            let findings: Vec<String> = k.static_findings.iter().map(|f| f.msg.clone()).collect();
            rows.push(Row {
                id,
                addr: l.addr,
                inst: l.text.clone(),
                field,
                sb,
                flagged: !findings.is_empty(),
                findings,
                hw: None,
                class: String::new(),
            });
        }
    }
    let flagged = rows.iter().filter(|r| r.flagged).count();
    println!(
        "sb-sweep {}: {} one-bit mutants, {flagged} flagged by the static check; running each {runs}x on GPU (smem scrubbed)",
        case.m.name,
        rows.len()
    );

    // One worker thread per free GPU. `--runs 0`: static check only.
    let gpus = if runs == 0 {
        vec![]
    } else {
        ptxv_hw::free_gpus()?
    };
    anyhow::ensure!(runs == 0 || !gpus.is_empty(), "no free GPU");
    let next = Mutex::new(0usize);
    let results: Mutex<Vec<(usize, HwRow)>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for &gpu in &gpus {
            let (rows, next, results, dir, exp_path) = (&rows, &next, &results, &dir, &exp_path);
            s.spawn(move || {
                loop {
                    let i = {
                        let mut n = next.lock().unwrap();
                        *n += 1;
                        *n - 1
                    };
                    if i >= rows.len() {
                        break;
                    }
                    let r = &rows[i];
                    let path: &Path = &dir.join(format!("{}.cubin", r.id));
                    match hw_run_cubin(case, &r.id, path, runs, timeout, exp_path, true, gpu) {
                        Ok(h) => results.lock().unwrap().push((i, h)),
                        Err(e) => eprintln!("{}: {e}", r.id),
                    }
                }
            });
        }
    });
    for (i, h) in results.into_inner().unwrap() {
        rows[i].hw = Some(h);
    }
    for r in rows.iter_mut() {
        let bad =
            r.hw.as_ref()
                .map(|h| h.mismatches > 0 || h.runs == 0)
                .unwrap_or(false);
        r.class = match (r.flagged, bad) {
            (true, true) => "caught",
            (false, true) => "missed",
            (true, false) => "hazard (hw won)",
            (false, false) => "benign",
        }
        .into();
    }
    let count = |c: &str| rows.iter().filter(|r| r.class == c).count();
    let mut md = format!(
        "# Scoreboard mutation sweep: {}\n\n{} one-bit mutants of the good binary; {runs} GPU runs each with shared memory scrubbed.\n\n\
         | class | count | meaning |\n|---|---|---|\n\
         | caught | {} | flagged, and the GPU computes wrong results / hangs |\n\
         | hazard (hw won) | {} | flagged; the GPU happened to be fast enough in every run |\n\
         | benign | {} | not flagged, GPU correct |\n\
         | **missed** | **{}** | not flagged, but the GPU is wrong (a hole in the model) |\n\n\
         | mutant | instruction | GPU | static check |\n|---|---|---|---|\n",
        case.m.name,
        rows.len(),
        count("caught"),
        count("hazard (hw won)"),
        count("benign"),
        count("missed")
    );
    for r in &rows {
        let hw =
            r.hw.as_ref()
                .map(|h| h.outcome.clone())
                .unwrap_or("not run".into());
        let f = r
            .findings
            .first()
            .map(|f| f.split(':').next().unwrap_or("").to_string())
            .unwrap_or("clean".into());
        let more = if r.findings.len() > 1 {
            format!(" (+{})", r.findings.len() - 1)
        } else {
            String::new()
        };
        md += &format!("| {} | `{}` | {hw} | {f}{more} |\n", r.id, r.inst);
    }
    std::fs::write(dir.join("sweep.md"), &md)?;
    std::fs::write(dir.join("sweep.json"), serde_json::to_string_pretty(&rows)?)?;
    println!(
        "caught {} | hazard (hw won) {} | benign {} | MISSED {}  -> {}",
        count("caught"),
        count("hazard (hw won)"),
        count("benign"),
        count("missed"),
        dir.join("sweep.md").display()
    );
    for r in rows
        .iter()
        .filter(|r| r.class == "missed" || r.class == "caught")
    {
        println!(
            "  {:<16} {:<8} {:<48} {}",
            r.id,
            r.class,
            r.inst,
            r.hw.as_ref().map(|h| h.outcome.as_str()).unwrap_or("")
        );
    }
    Ok(())
}
