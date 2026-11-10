/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Host arrays: a minimal `.npy` reader/writer and the tiled memory layouts of tpu7x operands.

use std::path::Path;

use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;

/// A host array as 32-bit or 16-bit bit patterns (bf16 arrays are stored as uint16).
#[derive(Clone, Debug, PartialEq)]
pub struct Array {
    pub shape: Vec<u64>,
    /// Element size in bytes (2 or 4).
    pub itemsize: usize,
    /// numpy descr (`<f4`, `<i4`, `<u4`, `<u2`).
    pub descr: String,
    /// Elements, zero-extended to u32.
    pub data: Vec<u32>,
}

impl Array {
    pub fn zeros(shape: &[u64], itemsize: usize, descr: &str) -> Self {
        let n = shape.iter().product::<u64>() as usize;
        Array {
            shape: shape.to_vec(),
            itemsize,
            descr: descr.to_string(),
            data: vec![0; n],
        }
    }
}

pub fn read_npy(path: &Path) -> Result<Array> {
    let b = std::fs::read(path).map_err(|e| anyhow!("{}: {e}", path.display()))?;
    if b.len() < 10 || &b[..6] != b"\x93NUMPY" {
        bail!("{}: not an npy file", path.display());
    }
    let (hlen, off) = if b[6] == 1 {
        (u16::from_le_bytes([b[8], b[9]]) as usize, 10)
    } else {
        (u32::from_le_bytes([b[8], b[9], b[10], b[11]]) as usize, 12)
    };
    let header = std::str::from_utf8(&b[off..off + hlen])?;
    let field = |k: &str| -> Option<&str> {
        let i = header.find(k)?;
        Some(&header[i + k.len()..])
    };
    let descr = field("'descr': '")
        .and_then(|r| r.split('\'').next())
        .ok_or_else(|| anyhow!("npy descr"))?
        .to_string();
    let fortran = field("'fortran_order': True").is_some();
    let shp = field("'shape': (")
        .and_then(|r| r.split(')').next())
        .ok_or_else(|| anyhow!("npy shape"))?;
    let shape: Vec<u64> = shp
        .split(',')
        .filter_map(|x| x.trim().parse().ok())
        .collect();
    let itemsize: usize = descr[2..].parse()?;
    let body = &b[off + hlen..];
    let n = shape.iter().product::<u64>() as usize;
    let data = match itemsize {
        4 => body
            .chunks_exact(4)
            .take(n)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        2 => body
            .chunks_exact(2)
            .take(n)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as u32)
            .collect(),
        1 => body.iter().take(n).map(|&c| c as u32).collect(),
        _ => bail!("npy itemsize {itemsize}"),
    };
    let data: Vec<u32> = if fortran && shape.len() > 1 {
        // column-major file order -> row-major elements
        let mut out = vec![0u32; n];
        let mut idx = vec![0u64; shape.len()];
        for &v in &data {
            let flat = idx.iter().zip(&shape).fold(0u64, |f, (&i, &d)| f * d + i);
            out[flat as usize] = v;
            for d in 0..shape.len() {
                idx[d] += 1;
                if idx[d] < shape[d] {
                    break;
                }
                idx[d] = 0;
            }
        }
        out
    } else {
        data
    };
    Ok(Array {
        shape,
        itemsize,
        descr,
        data,
    })
}

pub fn write_npy(path: &Path, a: &Array) -> Result<()> {
    let shape = match a.shape.len() {
        1 => format!("({},)", a.shape[0]),
        _ => format!(
            "({})",
            a.shape
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ),
    };
    let mut h = format!(
        "{{'descr': '{}', 'fortran_order': False, 'shape': {shape}, }}",
        a.descr
    );
    while (10 + h.len() + 1) % 64 != 0 {
        h.push(' ');
    }
    h.push('\n');
    let mut out = b"\x93NUMPY\x01\x00".to_vec();
    out.extend_from_slice(&(h.len() as u16).to_le_bytes());
    out.extend_from_slice(h.as_bytes());
    for &v in &a.data {
        match a.itemsize {
            4 => out.extend_from_slice(&v.to_le_bytes()),
            1 => out.push(v as u8),
            _ => out.extend_from_slice(&(v as u16).to_le_bytes()),
        }
    }
    std::fs::write(path, out)?;
    Ok(())
}

/// Element size and numpy descr of an LLO dtype (`f32`, `bf16`, `s32`, `u32`).
pub fn dtype(s: &str) -> Result<(usize, &'static str)> {
    Ok(match s {
        "f32" => (4, "<f4"),
        "s32" => (4, "<i4"),
        "u32" => (4, "<u4"),
        "bf16" => (2, "<u2"),
        "s8" => (1, "|i1"),
        _ => bail!("dtype {s}"),
    })
}

fn as_2d(shape: &[u64]) -> (usize, usize) {
    match shape.len() {
        0 => (1, 1),
        1 => (1, shape[0] as usize),
        _ => (
            shape[..shape.len() - 1].iter().product::<u64>() as usize,
            shape[shape.len() - 1] as usize,
        ),
    }
}

/// Is the array stored as plain words (SMEM, or 1-D outside VMEM)?
fn flat(shape: &[u64], space: &str) -> bool {
    space == "smem" || (shape.len() == 1 && space != "vmem")
}

/// 32-bit (1, N) arrays in VMEM: (1, 128) tiles, column tile j at unit offset j (Mosaic layout;
/// review_mxu_ops gridepi reads columns 128..255 of an f32[1,256] bias with
/// `vld [bias + $0x1] ss:$0`). ASSUME(TPU-L-ROW-TILE)
fn row_tiled(shape: &[u64], itemsize: usize, space: &str) -> bool {
    space == "vmem" && shape.len() == 2 && shape[0] == 1 && itemsize == 4
}

/// Array -> u32 words in memory order. 32-bit: tiles T(8,128), row-major over tiles; 16-bit:
/// T(8,128)(2,1), rows 2k (low half) and 2k+1 (high half) of a tile in one word
/// (HLO layouts of the corpus). ASSUME(TPU-L-BF16-PACK)
pub fn to_words(a: &Array, space: &str) -> Result<Vec<u32>> {
    if flat(&a.shape, space) {
        if a.itemsize != 4 {
            bail!("flat 16-bit array");
        }
        return Ok(a.data.clone());
    }
    let (r, c) = as_2d(&a.shape);
    if a.itemsize == 1 {
        return s8_words(a, space, r, c);
    }
    if row_tiled(&a.shape, a.itemsize, space) {
        let mut out = a.data.clone();
        out.resize(c.div_ceil(128) * 128, 0);
        return Ok(out);
    }
    let (rp, cp) = (r.div_ceil(8) * 8, c.div_ceil(128) * 128);
    let get = |i: usize, j: usize| if i < r && j < c { a.data[i * c + j] } else { 0 };
    let mut out = Vec::with_capacity(rp * cp);
    match a.itemsize {
        4 => {
            for ti in 0..rp / 8 {
                for tj in 0..cp / 128 {
                    for s in 0..8 {
                        for l in 0..128 {
                            out.push(get(ti * 8 + s, tj * 128 + l));
                        }
                    }
                }
            }
        }
        2 => {
            for ti in 0..rp / 8 {
                for tj in 0..cp / 128 {
                    for s in 0..4 {
                        for l in 0..128 {
                            let lo = get(ti * 8 + 2 * s, tj * 128 + l) & 0xFFFF;
                            let hi = get(ti * 8 + 2 * s + 1, tj * 128 + l) & 0xFFFF;
                            out.push(lo | (hi << 16));
                        }
                    }
                }
            }
        }
        _ => bail!("itemsize {}", a.itemsize),
    }
    Ok(out)
}

/// int8 operands in VMEM: (32, 128) tiles, row 4s+i of a tile in byte i of sublane s.
/// ASSUME(TPU-L-S8-PACK) (by analogy with the bf16 (2,1) packing; validated through the int8
/// matmul replays, review_sched int8).
fn s8_words(a: &Array, space: &str, r: usize, c: usize) -> Result<Vec<u32>> {
    if space != "vmem" {
        bail!("int8 array outside VMEM");
    }
    let (rp, cp) = (r.div_ceil(32) * 32, c.div_ceil(128) * 128);
    let get = |i: usize, j: usize| {
        if i < r && j < c {
            a.data[i * c + j] & 0xFF
        } else {
            0
        }
    };
    let mut out = Vec::with_capacity(rp * cp / 4);
    for ti in 0..rp / 32 {
        for tj in 0..cp / 128 {
            for s in 0..8 {
                for l in 0..128 {
                    let w = (0..4).fold(0u32, |w, i| {
                        w | get(ti * 32 + 4 * s + i, tj * 128 + l) << (8 * i)
                    });
                    out.push(w);
                }
            }
        }
    }
    Ok(out)
}

/// Number of words `to_words` produces.
pub fn n_words(shape: &[u64], itemsize: usize, space: &str) -> usize {
    if flat(shape, space) {
        return shape.iter().product::<u64>() as usize;
    }
    let (r, c) = as_2d(shape);
    if row_tiled(shape, itemsize, space) {
        return c.div_ceil(128) * 128;
    }
    if itemsize == 1 {
        return r.div_ceil(32) * 32 * c.div_ceil(128) * 128 / 4;
    }
    r.div_ceil(8) * 8 * c.div_ceil(128) * 128 / (4 / itemsize)
}

pub fn from_words(
    words: &[u32],
    shape: &[u64],
    itemsize: usize,
    descr: &str,
    space: &str,
) -> Array {
    let mut a = Array::zeros(shape, itemsize, descr);
    if flat(shape, space) || row_tiled(shape, itemsize, space) {
        let n = a.data.len();
        a.data.copy_from_slice(&words[..n]);
        return a;
    }
    let (r, c) = as_2d(shape);
    let cp = c.div_ceil(128) * 128;
    let tiles_c = cp / 128;
    if itemsize == 1 {
        for i in 0..r {
            for j in 0..c {
                let w =
                    words[((i / 32) * tiles_c + j / 128) * 1024 + ((i % 32) / 4) * 128 + j % 128];
                a.data[i * c + j] = (w >> (8 * (i % 4))) & 0xFF;
            }
        }
        return a;
    }
    for i in 0..r {
        for j in 0..c {
            let (ti, tj) = (i / 8, j / 128);
            let tile = ti * tiles_c + tj;
            a.data[i * c + j] = match itemsize {
                4 => words[tile * 1024 + (i % 8) * 128 + j % 128],
                _ => {
                    let w = words[tile * 512 + ((i % 8) / 2) * 128 + j % 128];
                    if i % 2 == 0 { w & 0xFFFF } else { w >> 16 }
                }
            };
        }
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layouts_round_trip() {
        for (shape, isz) in [
            (vec![16u64, 256], 4usize),
            (vec![24, 130], 2),
            (vec![64], 4),
            (vec![1, 256], 4),
            (vec![1, 200], 2),
            (vec![40, 130], 1),
        ] {
            let mut a = Array::zeros(&shape, isz, if isz == 4 { "<f4" } else { "<u2" });
            for (i, v) in a.data.iter_mut().enumerate() {
                *v = (i as u32 * 7919) & (u32::MAX >> (32 - 8 * isz));
            }
            for space in ["vmem", "hbm"]
                .into_iter()
                .filter(|s| isz != 1 || *s == "vmem")
            {
                let w = to_words(&a, space).unwrap();
                assert_eq!(w.len(), n_words(&shape, isz, space));
                assert_eq!(from_words(&w, &shape, isz, &a.descr, space), a);
            }
        }
    }

    /// gridepi regression: an f32[1,256] VMEM operand occupies two (1,128) tiles, so columns
    /// 128..255 start at word 128 (unit offset 1), not at word 1024 as with (8,128) padding.
    #[test]
    fn row_vector_uses_1x128_tiles() {
        let mut a = Array::zeros(&[1, 256], 4, "<f4");
        for (i, v) in a.data.iter_mut().enumerate() {
            *v = i as u32 + 1;
        }
        let w = to_words(&a, "vmem").unwrap();
        assert_eq!(w.len(), 256);
        assert_eq!(w[128], 129);
        assert_eq!(n_words(&[1, 256], 4, "vmem"), 256);
        assert_eq!(to_words(&a, "hbm").unwrap().len(), 8 * 256);
    }
}
