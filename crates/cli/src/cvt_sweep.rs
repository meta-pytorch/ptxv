/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! `ptxv cvt-sweep probes/cvt_lp`: run the narrow-format cvt probe on a GB300 over the sweep
//! inputs (`lowp::sweep::sets`: every 16-bit source value and code pair; 2^20 edge and shaped
//! random f32 operands with random bits), compare every output with `lowp`, and with
//! `--record` store per-op hashes of the hardware outputs in sweep.json, which cargo test
//! replays against the model without a GPU (N-CVT-LP).

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use anyhow::bail;
use ptxv_sim::lowp;
use ptxv_sim::lowp::sweep;

pub fn run(dir: &Path, gpu: Option<u32>, record: bool, reuse: bool) -> Result<()> {
    let ptx = std::fs::read_to_string(dir.join("kernel.ptx"))?;
    let ops = lowp::probe_ops(&ptx);
    let root = std::env::var("PTXV_ROOT").unwrap_or_else(|_| ".".into());
    let gpu = match gpu {
        Some(g) => g,
        None if reuse => 0,
        None => *ptxv_hw::free_gpus()?
            .first()
            .ok_or_else(|| anyhow::anyhow!("no free GPU"))?,
    };
    let exe = std::env::current_exe()?;
    let mut rec: BTreeMap<String, BTreeMap<String, (usize, String)>> = BTreeMap::new();
    let mut total_bad = 0usize;
    for s in sweep::sets() {
        let n = s.v[0].len();
        let out = Path::new(&root).join("out/cvt_lp").join(s.name);
        std::fs::create_dir_all(&out)?;
        std::fs::write(out.join("kernel.ptx"), &ptx)?;
        let mut toml = format!(
            "name = \"cvt_lp_{}\"\ndescription = \"cvt-sweep set\"\nptx = \"kernel.ptx\"\nentry = \"cvt_lp\"\ngrid = [{}, 1, 1]\nblock = [128, 1, 1]\noutputs = [\"out\"]\n",
            s.name,
            n / 128
        );
        let names = ["x", "y", "z", "w", "r"];
        for b in names {
            toml += &format!(
                "\n[[buffers]]\nname = \"{b}\"\ndtype = \"u32\"\nshape = [{n}]\ninit = \"zeros\"\n"
            );
        }
        toml += &format!(
            "\n[[buffers]]\nname = \"out\"\ndtype = \"u32\"\nshape = [{}]\ninit = \"zeros\"\n",
            n * ops.len()
        );
        for b in names.iter().chain(&["out"]) {
            toml += &format!("\n[[params]]\nkind = \"ptr\"\nbuffer = \"{b}\"\n");
        }
        toml += "\n[[variants]]\nid = \"good\"\nexpect = [\"clean\"]\n";
        std::fs::write(out.join("case.toml"), toml)?;
        let cubin = out.join("kernel.cubin");
        ptxv_sass::ptxas(&out.join("kernel.ptx"), &cubin, "sm_103a")?;
        let mut args: Vec<String> = vec![
            "hw-worker".into(),
            out.to_str().unwrap().into(),
            cubin.to_str().unwrap().into(),
            "--runs".into(),
            "1".into(),
        ];
        for (i, b) in names.iter().enumerate() {
            let p = out.join(format!("{b}.bin"));
            std::fs::write(
                &p,
                s.v[i]
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<u8>>(),
            )?;
            args.push("--input".into());
            args.push(format!("{b}={}", p.display()));
        }
        let dump = out.join("out.bin");
        args.push("--dump".into());
        args.push(dump.to_str().unwrap().into());
        if !(reuse && dump.exists()) {
            let o = std::process::Command::new(&exe)
                .args(&args)
                .env("CUDA_VISIBLE_DEVICES", gpu.to_string())
                .output()?;
            if !o.status.success() {
                bail!("hw run failed: {}", String::from_utf8_lossy(&o.stderr));
            }
        }
        let hw: Vec<u32> = std::fs::read(&dump)?
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        if hw.len() != n * ops.len() {
            bail!("dump has {} words, expected {}", hw.len(), n * ops.len());
        }
        println!("set {} ({n} threads, GPU {gpu}):", s.name);
        for (j, name) in ops.iter().enumerate() {
            let col = |i: usize| hw[i * ops.len() + j];
            let m = sweep::model(&s, name);
            let bad: Vec<usize> = (0..n).filter(|&i| m[i] != col(i)).collect();
            total_bad += bad.len();
            print!("  {j:2} {name:<32} {:>8} mismatches", bad.len());
            if let Some(&i) = bad.first() {
                let v = &s.v;
                print!(
                    "  first: x={:#010x} y={:#010x} z={:#010x} w={:#010x} r={:#010x} hw={:#x} model={:#x}",
                    v[0][i],
                    v[1][i],
                    v[2][i],
                    v[3][i],
                    v[4][i],
                    col(i),
                    m[i]
                );
            }
            println!();
            rec.entry(s.name.into()).or_default().insert(
                name.clone(),
                (n, format!("{:016x}", sweep::hash((0..n).map(col)))),
            );
        }
    }
    println!("total mismatches: {total_bad}");
    if record {
        let p = dir.join("sweep.json");
        std::fs::write(&p, serde_json::to_string_pretty(&rec)? + "\n")?;
        println!("written {}", p.display());
    }
    Ok(())
}
