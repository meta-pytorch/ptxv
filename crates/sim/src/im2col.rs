/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! im2col-mode TMA: `cp.async.bulk.tensor.<n>d ... .im2col` loads (SASS
//! `UTMALDG.<n>D.IM2COL`) and `.im2col_no_offs` stores (`UTMASTG.<n>D.IM2COL`) through a
//! `cuTensorMapEncodeIm2col` map (tmap.rs `Im2col`). Measured on GB300 (probes/tma_im2col):
//!
//! - The box is `pixels` rows of `channels` elements (shared memory as a 2-D box [pixels]
//!   [channels], padded / swizzled as a tiled box).
//! - Pixel q is the start pixel (the instruction's spatial coordinates and n) advanced q
//!   steps through the pixel bounding box in row-major order, innermost spatial dimension
//!   first: per spatial dimension i the box spans [lower_i, dim_i - 1 + upper_i]; past its
//!   end the coordinate restarts at lower_i and the next dimension advances; past the last
//!   spatial dimension, n advances (unbounded).
//! - Element (q, c) reads (or writes) the tensor at channel c0 + c, spatial coordinates
//!   pixel + offset_i, and n. Elements outside the tensor are zero (loads) or not written
//!   (stores). Offsets are unsigned, packed in 16 bits as SASS passes them: rank 5 three
//!   5-bit fields, rank 4 two 8-bit fields, rank 3 one 16-bit field (offset 0 first).
//! - A channel start whose byte offset is not a multiple of 16 is an illegal instruction, as
//!   is a store whose pixel box reaches outside the tensor (a lower corner below 0 or an
//!   upper corner above 0).
//!
//! A start pixel outside the bounding box or a start n outside the tensor is not modeled.
// ASSUME(S-TMA): see ASSUMPTIONS.md

use crate::tmap::Im2col;

/// Why a box cannot be walked.
#[derive(Debug, PartialEq, Eq)]
pub enum WalkErr {
    /// GB300 raises an illegal instruction.
    Fault(String),
    /// Outside the measured behaviour.
    Unmodeled(String),
}

/// Bits per im2col offset for a map of `rank` (SASS packs them in one 16-bit field).
pub fn offset_bits(rank: usize) -> u32 {
    match rank {
        5 => 5,
        4 => 8,
        _ => 16,
    }
}

/// The box elements of one im2col access, pixel-major (channel innermost): each element's
/// byte offset from the tensor base, or None outside the tensor. `coords` are the rank
/// coordinates (c, spatial..., n) and then the packed offsets (0 for a store).
pub fn walk(
    m: &Im2col,
    channels: u32,
    dims: &[u64],
    strides: &[u64],
    esize: u32,
    coords: &[i64],
    store: bool,
) -> Result<Vec<Option<u64>>, WalkErr> {
    let rank = dims.len();
    if !(3..=5).contains(&rank) || coords.len() != rank + 1 || m.lower.len() != rank - 2 {
        return Err(WalkErr::Unmodeled(format!(
            "im2col access of rank {rank} with {} coordinates",
            coords.len()
        )));
    }
    let k = rank - 2;
    let c0 = coords[0];
    if (c0 * esize as i64).rem_euclid(16) != 0 {
        return Err(WalkErr::Fault(format!(
            "im2col channel start {c0} ({esize}-byte elements) is not 16-byte aligned: illegal instruction on GB300"
        )));
    }
    if store && (m.lower.iter().any(|&l| l < 0) || m.upper.iter().any(|&u| u > 0)) {
        return Err(WalkErr::Fault(format!(
            "im2col store with a pixel box reaching outside the tensor (lower corner {:?}, upper corner {:?}): illegal instruction on GB300",
            m.lower, m.upper
        )));
    }
    let bits = offset_bits(rank);
    let packed = coords[rank] as u64 & 0xffff;
    let offs: Vec<i64> = (0..k)
        .map(|i| ((packed >> (bits as usize * i)) & ((1 << bits) - 1)) as i64)
        .collect();
    let hi: Vec<i64> = (0..k)
        .map(|i| dims[1 + i] as i64 - 1 + m.upper[i] as i64)
        .collect();
    let mut pix: Vec<i64> = coords[1..1 + k].to_vec();
    let mut n = coords[rank - 1];
    for i in 0..k {
        if pix[i] < m.lower[i] as i64 || pix[i] > hi[i] {
            return Err(WalkErr::Unmodeled(format!(
                "im2col start pixel {:?} outside the pixel bounding box (lower {:?}, upper corner {:?}, dims {:?})",
                &coords[1..1 + k],
                m.lower,
                m.upper,
                dims
            )));
        }
    }
    if n < 0 || n as u64 >= dims[rank - 1] {
        return Err(WalkErr::Unmodeled(format!(
            "im2col start n {n} outside the tensor (N = {})",
            dims[rank - 1]
        )));
    }
    let stride = |d: usize| if d == 0 { esize as u64 } else { strides[d - 1] };
    let mut out = Vec::with_capacity((m.pixels * channels) as usize);
    for _ in 0..m.pixels {
        let mut base: Option<u64> = Some(0);
        for i in 0..k {
            let g = pix[i] + offs[i];
            if g < 0 || g as u64 >= dims[1 + i] {
                base = None;
            }
            base = base.map(|b| b + g as u64 * stride(1 + i));
        }
        if n as u64 >= dims[rank - 1] {
            base = None;
        }
        let base = base.map(|b| b + n as u64 * stride(rank - 1));
        for c in 0..channels as i64 {
            let g = c0 + c;
            out.push(match base {
                Some(b) if g >= 0 && (g as u64) < dims[0] => Some(b + g as u64 * esize as u64),
                _ => None,
            });
        }
        // Advance one pixel through the bounding box.
        let mut i = 0;
        loop {
            if i == k {
                n += 1;
                break;
            }
            pix[i] += 1;
            if pix[i] > hi[i] {
                pix[i] = m.lower[i] as i64;
                i += 1;
                continue;
            }
            break;
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn walk_wraps_rows_and_n() {
        // rank 4 (C, W, H, N) = (4, 3, 2, 2), W box [-1, 2], H box [0, 1], 2 channels, 10
        // pixels from (W, H, n) = (1, 1, 0) with offsets (1, 0).
        let m = Im2col {
            lower: vec![-1, 0],
            upper: vec![0, 0],
            pixels: 10,
        };
        let dims = [4, 3, 2, 2];
        let strides = [16, 48, 96];
        let w = walk(&m, 2, &dims, &strides, 4, &[0, 1, 1, 0, 1], false).unwrap();
        assert_eq!(w.len(), 20);
        // pixel 0: (w, h, n) = (1, 1, 0), + offset w 1 -> w = 2: in bounds.
        assert_eq!(w[0], Some(2 * 16 + 48));
        // pixel 1: w = 2 + 1 = 3: outside.
        assert_eq!(w[2], None);
        // pixel 2: wraps to w = -1 (+1 = 0), h advances past 1 -> 0, n = 1.
        assert_eq!(w[4], Some(96));
        // Unaligned channel start, store outside the tensor, start outside the box.
        assert!(matches!(
            walk(&m, 2, &dims, &strides, 4, &[1, 1, 1, 0, 0], false),
            Err(WalkErr::Fault(_))
        ));
        assert!(matches!(
            walk(&m, 2, &dims, &strides, 4, &[0, 1, 1, 0, 0], true),
            Err(WalkErr::Fault(_))
        ));
        assert!(matches!(
            walk(&m, 2, &dims, &strides, 4, &[0, 3, 1, 0, 0], false),
            Err(WalkErr::Unmodeled(_))
        ));
    }
}
