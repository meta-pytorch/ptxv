/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! ptxv simulator and model checker.

pub mod assumptions;
pub mod dist;
pub mod dpor;
pub mod elementwise;
pub mod equiv;
pub mod errdist;
pub mod exact;
pub mod exec;
pub mod explore;
pub mod fp;
pub mod groups;
pub mod halfin;
pub mod l1;
pub mod l1_cores;
pub mod lowp;
pub mod machine;
pub mod mma_lp;
pub mod mufu;
pub mod mufu_sample;
mod mufu_tables;
pub mod norm;
pub mod num;
pub mod residency;
pub mod seg;
pub mod slices;
pub mod smem;
pub mod softmax;
pub mod spin;
pub mod tenant;
pub mod tmap;
pub mod twolevel;
pub mod wellsync;

use anyhow::Result;
use anyhow::bail;
use machine::*;
use ptxv_case::Case;
use ptxv_ir::Kernel;

/// Build the launch (memory map, parameter block, tensor maps) for a case and kernel.
// ASSUME(S-GLOBAL): see ASSUMPTIONS.md
pub fn build_launch(case: &Case, k: &Kernel) -> Result<Launch> {
    let m = &case.m;
    let mut allocs = Vec::new();
    for (i, b) in m.buffers.iter().enumerate() {
        allocs.push(Alloc {
            name: b.name.clone(),
            base: GLOBAL_BASE + ((i as u64) << 32),
            init: b.init_bytes(),
        });
    }
    let base_of = |name: &str| -> Result<u64> {
        let i = m
            .buffers
            .iter()
            .position(|b| b.name == name)
            .ok_or_else(|| anyhow::anyhow!("no buffer {name}"))?;
        Ok(GLOBAL_BASE + ((i as u64) << 32))
    };
    // Pointer tables (`ptrs`): filled in now that every buffer has its address.
    for (i, b) in m.buffers.iter().enumerate() {
        if !b.ptrs.is_empty() {
            allocs[i].init = b.init_bytes_with(base_of)?;
        }
    }
    let launches = m.all_launches();
    let parts: Vec<(usize, &[ptxv_ir::Param], u32, u32)> = if k.parts.is_empty() {
        vec![(0, &k.params[..], k.param_cbank, k.smem_bytes)]
    } else {
        k.parts
            .iter()
            .map(|p| (p.entry, &p.params[..], p.param_cbank, p.smem_bytes))
            .collect()
    };
    if parts.len() != launches.len() {
        bail!(
            "kernel has {} parts, case launches {} kernels",
            parts.len(),
            launches.len()
        );
    }
    let mut tensormaps = Vec::new();
    let mut grids: Vec<Grid> = Vec::new();
    let mut cta0 = 0;
    for (gi, (l, &(entry, kparams, param_cbank, smem_bytes))) in
        launches.iter().zip(&parts).enumerate()
    {
        if kparams.len() != l.params.len() {
            bail!(
                "kernel {} has {} params, case declares {}",
                l.entry,
                kparams.len(),
                l.params.len()
            );
        }
        let total = kparams.iter().map(|p| p.offset + p.size).max().unwrap_or(0) as usize;
        let mut params = vec![0u8; total];
        for (p, spec) in kparams.iter().zip(l.params) {
            let o = p.offset as usize;
            match spec {
                ptxv_case::ParamSpec::Ptr { buffer, offset } => {
                    if p.size != 8 {
                        bail!("param {} is {} bytes, expected a pointer", p.name, p.size);
                    }
                    params[o..o + 8].copy_from_slice(&(base_of(buffer)? + offset).to_le_bytes());
                }
                ptxv_case::ParamSpec::U32 { value } => {
                    params[o..o + 4].copy_from_slice(&value.to_le_bytes())
                }
                ptxv_case::ParamSpec::U64 { value } => {
                    params[o..o + 8].copy_from_slice(&value.to_le_bytes())
                }
                ptxv_case::ParamSpec::Bytes { hex, fields } => {
                    let b = ptxv_case::param_bytes(hex)?;
                    if b.len() != p.size as usize {
                        bail!(
                            "param {} is {} bytes, case gives {}",
                            p.name,
                            p.size,
                            b.len()
                        );
                    }
                    params[o..o + b.len()].copy_from_slice(&b);
                    for f in fields {
                        let fo = o + f.offset as usize;
                        match &f.spec {
                            ptxv_case::ParamSpec::Ptr { buffer, offset } => params[fo..fo + 8]
                                .copy_from_slice(&(base_of(buffer)? + offset).to_le_bytes()),
                            ptxv_case::ParamSpec::Tensormap {
                                buffer,
                                r#box,
                                dims,
                                strides,
                                swizzle,
                                offset,
                                elem4,
                                ..
                            } => {
                                let b = case.buffer(buffer)?;
                                let (dims, strides) = ptxv_case::tmap_geometry(b, dims, strides);
                                tensormaps.push(TmapDesc {
                                    grid: gi,
                                    param_off: fo as u32,
                                    buffer: buffer.clone(),
                                    base: base_of(buffer)? + offset,
                                    esize: b.elem_bytes() as u32,
                                    dims,
                                    strides,
                                    boxd: r#box.clone(),
                                    swizzle: ptxv_case::swizzle_of(swizzle)?.0,
                                    elem4: *elem4,
                                });
                            }
                            _ => bail!("param {}: only ptr / tensormap fields", p.name),
                        }
                    }
                }
                ptxv_case::ParamSpec::Tensormap {
                    buffer,
                    r#box,
                    dims,
                    strides,
                    swizzle,
                    offset,
                    raw: _,
                    elem4,
                } => {
                    if p.size != 128 {
                        bail!(
                            "param {} is {} bytes, expected a 128-byte tensor map",
                            p.name,
                            p.size
                        );
                    }
                    let b = case.buffer(buffer)?;
                    let es = b.elem_bytes() as u64;
                    let (dims, strides) = ptxv_case::tmap_geometry(b, dims, strides);
                    tensormaps.push(TmapDesc {
                        grid: gi,
                        param_off: p.offset,
                        buffer: buffer.clone(),
                        base: base_of(buffer)? + offset,
                        esize: es as u32,
                        dims,
                        strides,
                        boxd: r#box.clone(),
                        swizzle: ptxv_case::swizzle_of(swizzle)?.0,
                        elem4: *elem4,
                    });
                }
            }
        }
        let cluster = l.cluster.unwrap_or([1, 1, 1]);
        if (0..3).any(|d| cluster[d] == 0 || l.grid[d] % cluster[d] != 0) {
            bail!(
                "grid {:?} is not a multiple of the cluster shape {:?}",
                l.grid,
                cluster
            );
        }
        // ASSUME(S-CGA-SHAPE): a preferred shape must tile the grid and be a multiple of the
        // regular one, so that every group of CTAs is one preferred cluster or a whole number
        // of regular ones.
        if let Some(p) = l.preferred_cluster {
            if l.cluster.is_none()
                || (0..3).any(|d| p[d] == 0 || p[d] % cluster[d] != 0 || l.grid[d] % p[d] != 0)
            {
                bail!(
                    "preferred cluster {p:?}: must be a multiple of the cluster shape {:?} and divide the grid {:?}",
                    l.cluster,
                    l.grid
                );
            }
        }
        let g = Grid {
            grid: l.grid,
            block: l.block,
            cluster,
            explicit_cluster: l.cluster.is_some(),
            preferred: l.preferred_cluster.filter(|&p| p != cluster),
            cga_shape: 0,
            params,
            entry: entry as u32,
            cta0,
            // The dynamic window follows the static shared memory (ptx: `.extern .shared`).
            smem_bytes: smem_bytes.div_ceil(16) * 16 + l.dynamic_smem,
            param_cbank,
            pdl: l.pdl,
            device: l.device,
            after: match l.after {
                None => None,
                Some(a) if a < 0 => Some(None),
                Some(a) => {
                    let a = a as usize;
                    if a >= grids.len() || grids[a].device != l.device {
                        bail!(
                            "launch {}: after = {a} is not an earlier launch on its GPU",
                            grids.len()
                        );
                    }
                    Some(Some(a))
                }
            },
        };
        cta0 += g.ctas();
        grids.push(g);
    }
    let outputs = m
        .outputs
        .iter()
        .map(|o| m.buffers.iter().position(|b| &b.name == o).unwrap())
        .collect();
    let mma_model = match &m.mma_model {
        Some(name) => {
            fp::MmaModel::parse(name).ok_or_else(|| anyhow::anyhow!("bad mma_model {name}"))?
        }
        None => fp::DEFAULT_MMA,
    };
    let mut replicas = Vec::new();
    for g in &m.replicas {
        let mut ids = Vec::new();
        for o in g {
            ids.push(
                m.buffers
                    .iter()
                    .position(|b| &b.name == o)
                    .ok_or_else(|| anyhow::anyhow!("replicas: no buffer {o}"))?,
            );
        }
        replicas.push(ids);
    }
    let fence = std::env::var("PTXV_EXIT_FENCE")
        .ok()
        .or_else(|| m.exit_fence.clone());
    let exit_fence = match fence.as_deref() {
        None | Some("none") => None,
        Some("sys") => Some(ptxv_ir::Scope::Sys),
        Some("gpu") => Some(ptxv_ir::Scope::Gpu),
        Some(x) => bail!("exit_fence: unknown scope {x} (none, gpu, sys)"),
    };
    let l1_cores = match std::env::var("PTXV_L1_CORES").as_deref() {
        Ok("1") => true,
        Ok("0") => false,
        _ => m.l1_coresident,
    };
    let l1_havoc = match std::env::var("PTXV_L1_HAVOC").as_deref() {
        Ok("1") => true,
        Ok("0") => false,
        _ => m.l1_havoc,
    };
    // A refuted assumption (ASSUMPTIONS.md) needs an explicit opt-in, named in the verdict.
    let opted = assume_refuted(case);
    for a in &opted {
        match assumptions::get(a) {
            Some(x) if x.status == "refuted" => {}
            Some(x) => bail!("assume_refuted: {a} is not refuted (status {})", x.status),
            None => bail!("assume_refuted: unknown assumption {a}"),
        }
    }
    let poll_acquire = match std::env::var("PTXV_POLL_ACQUIRE").as_deref() {
        Ok("1") => true,
        Ok("0") => false,
        _ => m.poll_acquire,
    };
    if poll_acquire && !opted.iter().any(|a| a == "E-POLL-ACQ") {
        bail!(
            "case {} sets poll_acquire (or PTXV_POLL_ACQUIRE=1), which relies on E-POLL-ACQ, refuted on GB300 (ASSUMPTIONS.md, docs/poll-acq-litmus.md). Remove it to check the strict PTX model, or opt in explicitly: assume_refuted = [\"E-POLL-ACQ\"] in case.toml, --assume-refuted E-POLL-ACQ, or PTXV_ASSUME_REFUTED=E-POLL-ACQ",
            m.name
        );
    }
    let refuted_opt_in: Vec<String> = if poll_acquire {
        vec!["E-POLL-ACQ".into()]
    } else {
        vec![]
    };
    Ok((Launch {
        grids,
        replicas,
        allocs,
        tensormaps,
        outputs,
        mma_model,
        exit_fence,
        poll_acquire,
        smem_cumulative: match std::env::var("PTXV_SMEM_CUMULATIVE").as_deref() {
            Ok("1") => true,
            Ok("0") => false,
            _ => m.smem_cumulative,
        },
        hw_l1: match std::env::var("PTXV_HW_L1").as_deref() {
            Ok("1") => true,
            Ok("0") => false,
            _ => m.hw_l1,
        } || l1_cores
            || l1_havoc,
        l1_cores,
        l1_havoc,
        env: residency::Env::from_case(m)?,
        read_done: match std::env::var("PTXV_READ_DONE").as_deref() {
            Ok("1") => true,
            Ok("0") => false,
            _ => m.read_done,
        },
        hw_order: false,
        refuted_opt_in,
    })
    .with_hw_model_env())
}

/// Refuted assumptions the case may use: its `assume_refuted` plus `PTXV_ASSUME_REFUTED`
/// (comma-separated; `ptxv check --assume-refuted` sets it).
pub fn assume_refuted(case: &Case) -> Vec<String> {
    let mut v = case.m.assume_refuted.clone();
    if let Ok(s) = std::env::var("PTXV_ASSUME_REFUTED") {
        v.extend(
            s.split(',')
                .map(|x| x.trim().to_string())
                .filter(|x| !x.is_empty()),
        );
    }
    v.sort();
    v.dedup();
    v
}

/// Exact reference outputs for the case's math spec (independent of the kernel).
pub fn spec_exact(case: &Case) -> Result<Option<Vec<exact::Expansion>>> {
    let Some(spec) = &case.m.spec else {
        return Ok(None);
    };
    match spec.as_str() {
        "half_row_sum" => {
            let a = case.buffer("A")?;
            let bytes = a.init_bytes();
            let cols = a.shape[1] as usize;
            let val = |r: usize, c: usize| {
                let i = (r * cols + c) * 2;
                ptxv_case::bf16_to_f32(u16::from_le_bytes([bytes[i], bytes[i + 1]])) as f64
            };
            let n = case.buffer("out")?.elems();
            Ok(Some(
                (0..n)
                    .map(|t| {
                        let mut e = vec![];
                        for c in 0..32 {
                            e = exact::add(&e, &exact::from(val(t / 2, (t % 2) * 32 + c)));
                        }
                        e
                    })
                    .collect(),
            ))
        }
        "gemm_tn" => {
            // D[m][n] = sum_k A[m][k] * B[n][k]  (bf16 inputs, exact)
            let (a, b) = (case.buffer("A")?, case.buffer("B")?);
            let (ab, bb) = (a.init_bytes(), b.init_bytes());
            let k = a.shape[1] as usize;
            let (m, n) = (a.shape[0] as usize, b.shape[0] as usize);
            let v = |bytes: &[u8], i: usize| {
                ptxv_case::bf16_to_f32(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]])) as f64
            };
            let mut out = Vec::with_capacity(m * n);
            for mi in 0..m {
                for ni in 0..n {
                    let mut e = vec![];
                    for ki in 0..k {
                        e = exact::add(&e, &exact::from(v(&ab, mi * k + ki) * v(&bb, ni * k + ki)));
                    }
                    out.push(e);
                }
            }
            Ok(Some(out))
        }
        "gemm_tn_mx" => {
            // MXF8: D[m][n] = sum_k A[m][k] B[n][k] 2^(SFA[m][k/32] - 127) 2^(SFB[n][k/32] - 127),
            // A / B e4m3 codes, ue8m0 scales in CUTLASS's 512-byte scale-factor atoms (128
            // rows x 4 K-blocks): byte ((mn/128 * rk + kb/4) * 32 + mn%32) * 16 + (mn/32)%4 * 4
            // + kb%4, rk = K/128 (exact).
            let (a, b) = (case.buffer("A")?, case.buffer("B")?);
            let (ab, bb) = (a.init_bytes(), b.init_bytes());
            let (sa, sb) = (
                case.buffer("SFA")?.init_bytes(),
                case.buffer("SFB")?.init_bytes(),
            );
            let k = a.shape[1] as usize;
            let (m, n) = (a.shape[0] as usize, b.shape[0] as usize);
            let rk = k / 128;
            let sf = |bytes: &[u8], r: usize, kb: usize| {
                let c =
                    bytes[((r / 128 * rk + kb / 4) * 32 + r % 32) * 16 + (r / 32) % 4 * 4 + kb % 4];
                lowp::value(c as u32, ptxv_ir::Lp::Ue8m0)
            };
            let e4 = |c: u8| lowp::value(c as u32, ptxv_ir::Lp::E4m3);
            let mut out = Vec::with_capacity(m * n);
            for mi in 0..m {
                for ni in 0..n {
                    let mut e = vec![];
                    for ki in 0..k {
                        let p = e4(ab[mi * k + ki])
                            * e4(bb[ni * k + ki])
                            * sf(&sa, mi, ki / 32)
                            * sf(&sb, ni, ki / 32);
                        e = exact::add(&e, &exact::from(p));
                    }
                    out.push(e);
                }
            }
            Ok(Some(out))
        }
        s => bail!("unknown spec {s}"),
    }
}
