/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Decoder for the opaque CUtensorMap (TMA descriptor) binary format, as read by the
//! hardware from memory. The layout was reverse-engineered from `cuTensorMapEncodeTiled`
//! outputs (probes/tmap_encode, driver 580 / CUDA 13.0, GB300); see docs/tensormap.md for
//! the field table and the evidence behind each entry.
//!
//! The decoder is strict: any bit it does not understand, any mode it does not model
//! (im2col, interleaved layouts, sub-byte element types) and any inconsistency in the
//! derived fields is an error. It never guesses.

/// Element type (hardware code in bits 7..10 of the 32-bit word at byte 8, plus the TF32
/// flag bit 16). The hardware codes differ from the `CUtensorMapDataType` enum order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    U8,
    U16,
    U32,
    S32,
    U64,
    S64,
    F16,
    F32,
    F32Ftz,
    F64,
    Bf16,
    Tf32,
    Tf32Ftz,
}

impl Dtype {
    pub fn size(self) -> u32 {
        match self {
            Dtype::U8 => 1,
            Dtype::U16 | Dtype::F16 | Dtype::Bf16 => 2,
            Dtype::U32 | Dtype::S32 | Dtype::F32 | Dtype::F32Ftz | Dtype::Tf32 | Dtype::Tf32Ftz => {
                4
            }
            Dtype::U64 | Dtype::S64 | Dtype::F64 => 8,
        }
    }
}

/// `CU_TENSOR_MAP_SWIZZLE_128B_ATOM_*`: granule of the 128-byte swizzle (bits 19..20).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SwizzleAtom {
    /// The classic swizzle: 16-byte chunks (`SWIZZLE_32B/64B/128B`, and `NONE`).
    B16,
    /// `SWIZZLE_128B_ATOM_32B`.
    B32,
    /// `SWIZZLE_128B_ATOM_32B_FLIP_8B`.
    B32Flip8,
    /// `SWIZZLE_128B_ATOM_64B`.
    B64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Decoded {
    /// Global base address (bits 0..56; 16-byte aligned).
    pub base: u64,
    pub dtype: Dtype,
    /// Element size in bytes (`dtype.size()`).
    pub esize: u32,
    /// Extents, innermost first (`rank` entries).
    pub dims: Vec<u64>,
    /// Byte strides of dims 1.. (`rank - 1` entries; dim 0 is contiguous).
    pub strides: Vec<u64>,
    /// Box (tile) extents in elements, innermost first.
    pub boxd: Vec<u32>,
    /// Element (traversal) strides, innermost first, each 1..=8.
    pub estrides: Vec<u32>,
    /// Shared-memory swizzle span in bytes: 0 (none), 32, 64 or 128.
    pub swizzle: u32,
    /// Swizzle granule; anything but `B16` only occurs with `swizzle == 128`.
    pub swizzle_atom: SwizzleAtom,
    /// Interleave in bytes. Always 0: interleaved layouts are rejected.
    pub interleave: u32,
    /// `CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA` (else out-of-bounds fill is zero).
    pub oob_nan: bool,
    /// L2 promotion in bytes: 0, 64, 128 or 256.
    pub l2_promotion: u32,
    /// Bit 21 of the word at byte 8. The driver sets it exactly when
    /// `prod(dims) * esize >= 128 KiB` (wrapping u64 product; 3097/3097 samples). It is a
    /// hint of unknown meaning: CUTLASS's device-side updates never refresh it.
    pub large_hint: bool,
    /// Bit 1 of the word at byte 8. Never produced by the driver; set by library-built
    /// descriptors (cuBLAS nvjet, CuTeDSL). Meaning unknown; when set, only bytes 0..63 are
    /// interpreted (cuBLAS packs such descriptors 64 bytes apart).
    pub lib_flag: bool,
}

fn u32_at(b: &[u8], off: usize) -> u32 {
    u32::from_le_bytes(b[off..off + 4].try_into().unwrap())
}
fn u64_at(b: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(b[off..off + 8].try_into().unwrap())
}

/// Decode a tensor map. `b` must hold at least 64 bytes; at least 128 unless the descriptor
/// has the library flag (bit 1 of the word at byte 8), in which case bytes 64.. are ignored.
pub fn decode(b: &[u8]) -> Result<Decoded, String> {
    if b.len() < 64 {
        return Err(format!(
            "tensor map: need at least 64 bytes, got {}",
            b.len()
        ));
    }
    let base = u64_at(b, 0);
    if base >> 57 != 0 {
        return Err(format!(
            "tensor map: address bits 57..63 set ({base:#x}); never observed"
        ));
    }
    if base & 0xf != 0 {
        return Err(format!(
            "tensor map: address {base:#x} not 16-byte aligned; never observed"
        ));
    }
    let w = u32_at(b, 8);
    if w & 1 != 0 {
        return Err("tensor map: im2col mode (bit 0 of word 8) is not modeled".into());
    }
    let lib_flag = w & 2 != 0;
    // Bits of word 8 this decoder understands: 1, 4..21.
    let known = 0x2 | 0x003f_fff0;
    if w & !known != 0 {
        return Err(format!(
            "tensor map: unknown bits {:#x} set in word 8 ({w:#010x})",
            w & !known
        ));
    }
    let rank = ((w >> 4) & 7) as usize + 1;
    if rank > 5 {
        return Err(format!("tensor map: rank {rank} > 5"));
    }
    let tf32 = (w >> 16) & 1 != 0;
    let code = (w >> 7) & 0xf;
    let dtype = match (code, tf32) {
        (0, false) => Dtype::U8,
        (1, false) => Dtype::U16,
        (2, false) => Dtype::U32,
        (3, false) => Dtype::S32,
        (4, false) => Dtype::U64,
        (5, false) => Dtype::S64,
        (6, false) => Dtype::F16,
        (7, false) => Dtype::F32,
        (8, false) => Dtype::F32Ftz,
        (9, false) => Dtype::F64,
        (10, false) => Dtype::Bf16,
        (7, true) => Dtype::Tf32,
        (8, true) => Dtype::Tf32Ftz,
        (11..=13, false) => {
            return Err(format!(
                "tensor map: sub-byte element type (hardware code {code}: 16U4_ALIGN8B / \
                 16U4_ALIGN16B / 16U6_ALIGN16B) is not modeled"
            ));
        }
        _ => {
            return Err(format!(
                "tensor map: unknown element type code {code} (tf32 flag {tf32})"
            ));
        }
    };
    let esize = dtype.size();
    let il = (w >> 11) & 3;
    if il != 0 {
        return Err(format!(
            "tensor map: interleaved layout (code {il}: {}) is not modeled",
            ["none", "16B", "32B", "?"][il as usize]
        ));
    }
    let swizzle = match (w >> 13) & 3 {
        0 => 0,
        1 => 32,
        2 => 64,
        _ => 128,
    };
    let oob_nan = (w >> 15) & 1 != 0;
    let l2_promotion = [0, 64, 128, 256][((w >> 17) & 3) as usize];
    let swizzle_atom = match (w >> 19) & 3 {
        0 => SwizzleAtom::B16,
        1 => SwizzleAtom::B32,
        2 => SwizzleAtom::B32Flip8,
        _ => SwizzleAtom::B64,
    };
    if swizzle_atom != SwizzleAtom::B16 && swizzle != 128 {
        return Err(format!(
            "tensor map: swizzle atom {swizzle_atom:?} with a {swizzle}B span; never observed"
        ));
    }
    let large_hint = (w >> 21) & 1 != 0;

    // Strides of dims 1..4: low 32 bits of (stride >> 4) at 0x0c + 4(i-1), bits 36..39 in
    // nibble i-1 of the word at 0x1c. Slots at or beyond `rank` are ignored (CUTLASS's
    // device-side updates write junk there).
    let hi = u32_at(b, 0x1c);
    if hi >> 16 != 0 {
        return Err(format!(
            "tensor map: unknown bits set in word 0x1c ({hi:#010x})"
        ));
    }
    let strides: Vec<u64> = (1..rank)
        .map(|i| {
            let lo = u32_at(b, 0x0c + 4 * (i - 1)) as u64;
            let h = ((hi >> (4 * (i - 1))) & 0xf) as u64;
            ((h << 32) | lo) << 4
        })
        .collect();
    let dims: Vec<u64> = (0..rank)
        .map(|i| u32_at(b, 0x20 + 4 * i) as u64 + 1)
        .collect();

    // Word 0x34: element strides (3 bits each, minus 1) in bits 0..14, box dim 0 minus 1 in
    // byte 0x37. Bytes 0x38..0x3b: box dims 1..4 minus 1. Bytes 0x3c..0x3f: im2col only.
    let w34 = u32_at(b, 0x34);
    if w34 & 0x00ff_8000 != 0 {
        return Err(format!(
            "tensor map: unknown bits set in word 0x34 ({w34:#010x})"
        ));
    }
    let estrides: Vec<u32> = (0..rank).map(|i| ((w34 >> (3 * i)) & 7) + 1).collect();
    let boxd: Vec<u32> = (0..rank).map(|i| b[0x37 + i] as u32 + 1).collect();
    if u32_at(b, 0x3c) != 0 {
        return Err(format!(
            "tensor map: bytes 0x3c..0x3f nonzero ({:#010x}); only im2col maps use them",
            u32_at(b, 0x3c)
        ));
    }

    if !lib_flag {
        // Driver-encoded maps carry two derived quantities and zeros after them.
        if b.len() < 128 {
            return Err(format!(
                "tensor map: need 128 bytes for a driver-encoded map, got {}",
                b.len()
            ));
        }
        // Integer division per dim, as the driver computes it (even when it truncates).
        let box_bytes = boxd
            .iter()
            .zip(&estrides)
            .fold(esize as u64, |p, (&bx, &e)| p * (bx / e) as u64);
        // Maps built on the device (tensormap.replace into zeroed memory: Triton's device-side
        // descriptors, CuTeDSL grouped GEMMs) leave both derived words 0, and GB300 runs them
        // (probes/tmap_device): 0 means "not provided", any other value must match.
        let q8 = u64_at(b, 0x40);
        if q8 != box_bytes && q8 != 0 {
            return Err(format!(
                "tensor map: word 0x40 = {q8:#x}, expected box bytes {box_bytes:#x} \
                 (esize * prod(box / estride)); stale or unknown encoding"
            ));
        }
        let rep = if swizzle == 0 { 16 } else { 8 * swizzle as u64 };
        let q9 = u64_at(b, 0x48);
        if q9 != rep && q9 != 0 {
            return Err(format!(
                "tensor map: word 0x48 = {q9:#x}, expected swizzle repeat {rep:#x}"
            ));
        }
        if b[0x50..0x80].iter().any(|&x| x != 0) {
            return Err("tensor map: bytes 0x50..0x7f nonzero; never observed".into());
        }
    }

    Ok(Decoded {
        base,
        dtype,
        esize,
        dims,
        strides,
        boxd,
        estrides,
        swizzle,
        swizzle_atom,
        interleave: 0,
        oob_nan,
        l2_promotion,
        large_hint,
        lib_flag,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hex(s: &str) -> Vec<u8> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
            .collect()
    }

    // Samples from probes/tmap_encode (cuTensorMapEncodeTiled, driver 580, GB300). Each
    // comment is the probe input: dtype rank interleave swizzle l2 oob addr dims strides box
    // estrides (CUtensorMapDataType / enum values).

    #[test]
    fn bf16_2d_swizzle128() {
        // 9 2 0 3 2 0 0x7f0000001000 128 512 256 64 128 1 1
        let d = decode(&hex("00100000007f00001065240010000000000000000000000000000000000000007f000000ff0100000000000000000000000000000000003f7f0000000000000000400000000000000004000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")).unwrap();
        assert_eq!(d.base, 0x7f0000001000);
        assert_eq!(d.dtype, Dtype::Bf16);
        assert_eq!(d.esize, 2);
        assert_eq!(d.dims, vec![128, 512]);
        assert_eq!(d.strides, vec![256]);
        assert_eq!(d.boxd, vec![64, 128]);
        assert_eq!(d.estrides, vec![1, 1]);
        assert_eq!(d.swizzle, 128);
        assert_eq!(d.swizzle_atom, SwizzleAtom::B16);
        assert_eq!(d.l2_promotion, 128);
        assert!(!d.oob_nan);
        assert!(d.large_hint); // 128 * 512 * 2 = 128 KiB
        assert!(!d.lib_flag);
    }

    #[test]
    fn bf16_5d_strides_estrides() {
        // 9 5 0 0 0 0 0x7f0000001000 128 5 6 7 8 0x110 0x1220 0x13330 0x144440 64 3 4 5 6 1 2 3 4 5
        let d = decode(&hex("00100000007f00004005200011000000220100003313000044440100000000007f000000040000000500000006000000070000008846003f020304050000000080000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")).unwrap();
        assert_eq!(d.dims, vec![128, 5, 6, 7, 8]);
        assert_eq!(d.strides, vec![0x110, 0x1220, 0x13330, 0x144440]);
        assert_eq!(d.boxd, vec![64, 3, 4, 5, 6]);
        assert_eq!(d.estrides, vec![1, 2, 3, 4, 5]);
        assert_eq!(d.swizzle, 0);
        assert_eq!(d.l2_promotion, 0);
    }

    #[test]
    fn f32_large_strides_oob_l2() {
        // 7 3 0 1 1 1 0x1fffffffffff0 1000 3 4096 0xfffffffff0 0x1fffffffe0 8 2 4 1 1 1
        let d = decode(&hex("f0ffffffffff0100a0a32200fffffffffeffffff00000000000000001f000000e703000002000000ff0f0000000000000000000000000007010300000000000000010000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")).unwrap();
        assert_eq!(d.base, 0x1fffffffffff0);
        assert_eq!(d.dtype, Dtype::F32);
        assert_eq!(d.dims, vec![1000, 3, 4096]);
        assert_eq!(d.strides, vec![0xfffffffff0, 0x1fffffffe0]);
        assert_eq!(d.boxd, vec![8, 2, 4]);
        assert_eq!(d.swizzle, 32);
        assert!(d.oob_nan);
        assert_eq!(d.l2_promotion, 64);
    }

    #[test]
    fn tf32_atom32_estride() {
        // 11 2 0 5 3 0 0x7f0000002000 256 300 1024 32 16 1 8
        let d = decode(&hex("00200000007f0000906337004000000000000000000000000000000000000000ff0000002b0100000000000000000000000000003800001f0f0000000000000000010000000000000004000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")).unwrap();
        assert_eq!(d.dtype, Dtype::Tf32);
        assert_eq!(d.dims, vec![256, 300]);
        assert_eq!(d.strides, vec![1024]);
        assert_eq!(d.boxd, vec![32, 16]);
        assert_eq!(d.estrides, vec![1, 8]);
        assert_eq!(d.swizzle, 128);
        assert_eq!(d.swizzle_atom, SwizzleAtom::B32Flip8);
        assert_eq!(d.l2_promotion, 256);
        assert!(d.large_hint); // 256 * 300 * 4 = 300 KiB
    }

    #[test]
    fn u8_rank1_and_u64_atom64() {
        // 0 1 0 0 0 0 0x7f0000001000 1024 256 1
        let d = decode(&hex("00100000007f0000000000000000000000000000000000000000000000000000ff03000000000000000000000000000000000000000000ff000000000000000000010000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")).unwrap();
        assert_eq!(
            (d.dtype, d.dims.clone(), d.strides.len()),
            (Dtype::U8, vec![1024], 0)
        );
        assert_eq!(d.boxd, vec![256]);
        // 4 2 0 6 0 0 0x7f0000001000 64 64 512 16 4 1 1
        let d = decode(&hex("00100000007f00001062180020000000000000000000000000000000000000003f0000003f0000000000000000000000000000000000000f030000000000000000020000000000000004000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")).unwrap();
        assert_eq!(d.dtype, Dtype::U64);
        assert_eq!(d.swizzle_atom, SwizzleAtom::B64);
        assert_eq!(d.strides, vec![512]);
    }

    #[test]
    fn rejects_unsupported() {
        // 9 3 2 0 0 0 0x7f0000001000 16 64 4 256 16384 16 16 2 1 1 1   (interleave 32B)
        let e = decode(&hex("00100000007f00002015200010000000000400000000000000000000000000000f0000003f0000000300000000000000000000000000000f0f0100000000000000400000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")).unwrap_err();
        assert!(e.contains("interleaved"), "{e}");
        // 13 2 0 0 0 0 0x7f0000001000 128 4 64 128 1 1 1   (16U4_ALIGN8B)
        let e = decode(&hex("00100000007f00009005000004000000000000000000000000000000000000007f000000030000000000000000000000000000000000007f000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")).unwrap_err();
        assert!(e.contains("sub-byte"), "{e}");
        // im2col 9 3 0 0 0 0 0x7f0000001000 64 32 32 128 4096 0 0 64 32 1 1 1
        let e = decode(&hex("00100000007f00002105200008000000000100000000000000000000000000003f0000001f0000001f00000000000000000000000000003f000000001f00000000100000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")).unwrap_err();
        assert!(e.contains("im2col"), "{e}");
        // An unobserved bit (word 8 bit 22) and a stale box-bytes word.
        let mut b = hex(
            "00100000007f00001065240010000000000000000000000000000000000000007f000000ff0100000000000000000000000000000000003f7f0000000000000000400000000000000004000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
        );
        b[10] |= 0x40;
        assert!(decode(&b).unwrap_err().contains("unknown bits"));
        b[10] &= !0x40;
        b[0x37] = 31; // box 0 = 32 without updating word 0x40
        assert!(decode(&b).unwrap_err().contains("word 0x40"));
        // Device-built maps leave the derived words 0 (probes/tmap_device).
        b[0x40..0x50].fill(0);
        assert!(decode(&b).is_ok());
        assert!(decode(&b[..64]).is_err());
    }

    /// The four mainloop templates and the epilogue template in rung 3a's 2304-byte CUTLASS
    /// Params block (fixtures/r3_grouped_mm, offsets 128 / 384 / 640 / 896 and 1664) are
    /// byte-identical to the driver's encoding with address 0, stride 0, and these dims.
    #[test]
    fn r3_grouped_mm_templates() {
        // 9 2 0 3 2 0 0 1 1 0 64 128 1 1
        let a = decode(&hex("000000000000000010650400000000000000000000000000000000000000000000000000000000000000000000000000000000000000003f7f0000000000000000400000000000000004000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")).unwrap();
        assert_eq!(
            (a.base, a.dims.clone(), a.strides.clone()),
            (0, vec![1, 1], vec![0])
        );
        assert_eq!(
            (a.boxd.clone(), a.swizzle, a.l2_promotion),
            (vec![64, 128], 128, 128)
        );
        // 9 2 0 2 2 0 0 256 128 0 32 128 1 1
        let d = decode(&hex("0000000000000000104504000000000000000000000000000000000000000000ff0000007f0000000000000000000000000000000000001f7f0000000000000000200000000000000002000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000")).unwrap();
        assert_eq!(
            (d.dims.clone(), d.strides.clone()),
            (vec![256, 128], vec![0])
        );
        assert_eq!((d.boxd.clone(), d.swizzle), (vec![32, 128], 64));
    }

    /// cuBLAS nvjet descriptors (fixtures/r2_cublas_gemm, params offset 256 = A): the
    /// library flag is set and the next descriptor follows 64 bytes later.
    #[test]
    fn r2_cublas_lib_flag() {
        let d = decode(&hex("0000c0fdf9ff00002265040010000000000000000000001000000000000000007f000000ff0100000000000000000000000000000000003f0700000000000000")).unwrap();
        assert!(d.lib_flag);
        assert_eq!(d.base, 0xfff9fdc00000);
        assert_eq!(d.dtype, Dtype::Bf16);
        assert_eq!(d.dims, vec![128, 512, 1]);
        // Stride of the extent-1 dim 2 is 0 in the descriptor (never stepped).
        assert_eq!(d.strides, vec![256, 0]);
        assert_eq!(d.boxd, vec![64, 8, 1]);
        assert_eq!(d.swizzle, 128);
    }
}

#[cfg(test)]
mod sweep {
    use super::*;

    /// Random cuTensorMapEncodeTiled sweep (probes/tmap_encode/sweep.py): every sample the
    /// decoder accepts must reproduce the encoder's arguments; the rest must be rejected for
    /// a modeled reason (interleave).
    #[test]
    fn encoder_sweep() {
        let text = include_str!("../../../probes/tmap_encode/samples.txt");
        let (mut ok, mut rejected) = (0, 0);
        for line in text.lines().filter(|l| !l.trim().is_empty()) {
            let (args, hexs) = line.split_once(" => ").unwrap();
            let v: Vec<u64> = args
                .split_whitespace()
                .map(|x| x.parse().unwrap())
                .collect();
            let b: Vec<u8> = (0..hexs.trim().len())
                .step_by(2)
                .map(|i| u8::from_str_radix(&hexs[i..i + 2], 16).unwrap())
                .collect();
            let (dt, rank, il, sw, l2, oob, addr) =
                (v[0], v[1] as usize, v[2], v[3], v[4], v[5], v[6]);
            let mut k = 7;
            let dims = v[k..k + rank].to_vec();
            k += rank;
            let strides = v[k..k + rank - 1].to_vec();
            k += rank - 1;
            let boxd: Vec<u32> = v[k..k + rank].iter().map(|&x| x as u32).collect();
            k += rank;
            let es: Vec<u32> = v[k..k + rank].iter().map(|&x| x as u32).collect();
            let r = decode(&b);
            if il != 0 {
                assert!(r.unwrap_err().contains("interleaved"), "{line}");
                rejected += 1;
                continue;
            }
            let d = r.unwrap_or_else(|e| panic!("{line}: {e}"));
            let want = [
                Dtype::U8,
                Dtype::U16,
                Dtype::U32,
                Dtype::S32,
                Dtype::U64,
                Dtype::S64,
                Dtype::F16,
                Dtype::F32,
                Dtype::F64,
                Dtype::Bf16,
                Dtype::F32Ftz,
                Dtype::Tf32,
                Dtype::Tf32Ftz,
            ][dt as usize];
            assert_eq!(d.dtype, want, "{line}");
            assert_eq!(d.base, addr, "{line}");
            assert_eq!(d.dims, dims, "{line}");
            assert_eq!(d.strides, strides, "{line}");
            assert_eq!(d.boxd, boxd, "{line}");
            assert_eq!(d.estrides, es, "{line}");
            assert_eq!(
                d.swizzle,
                [0, 32, 64, 128, 128, 128, 128][sw as usize],
                "{line}"
            );
            let atom = [
                SwizzleAtom::B16,
                SwizzleAtom::B16,
                SwizzleAtom::B16,
                SwizzleAtom::B16,
                SwizzleAtom::B32,
                SwizzleAtom::B32Flip8,
                SwizzleAtom::B64,
            ][sw as usize];
            assert_eq!(d.swizzle_atom, atom, "{line}");
            assert_eq!(d.l2_promotion, [0, 64, 128, 256][l2 as usize], "{line}");
            assert_eq!(d.oob_nan, oob == 1, "{line}");
            let size = dims.iter().fold(d.esize as u64, |p, &x| p.wrapping_mul(x));
            assert_eq!(d.large_hint, size >= 1 << 17, "{line}");
            ok += 1;
        }
        assert!(ok > 0 && rejected > 0, "{ok} {rejected}");
    }
}

#[cfg(test)]
mod fixtures {
    use super::*;

    /// The parameter block (hex) of the given byte length in a case file.
    fn block(case: &str, len: usize) -> Vec<u8> {
        let h = case
            .split("hex = \"")
            .skip(1)
            .map(|s| &s[..s.find('"').unwrap()])
            .find(|h| h.len() == 2 * len)
            .expect("no parameter block of that size");
        (0..h.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&h[i..i + 2], 16).unwrap())
            .collect()
    }

    /// Rung 3a: CUTLASS's host-built templates (address 0, extents 1 / stride 0) that the
    /// kernel copies to shared memory, patches (address, dims, stride) and publishes to the
    /// global workspace. No other 64-byte-aligned offset holds a decodable map.
    #[test]
    fn r3_grouped_mm_params() {
        let b = block(
            include_str!("../../../fixtures/r3_grouped_mm/case.toml"),
            2304,
        );
        let mut found = vec![];
        for off in (0..=b.len() - 128).step_by(64) {
            if let Ok(d) = decode(&b[off..off + 128]) {
                if !d.lib_flag
                    && u64::from_le_bytes(b[off + 0x48..off + 0x50].try_into().unwrap()) != 0
                {
                    found.push((off, d));
                }
            }
        }
        let offs: Vec<usize> = found.iter().map(|f| f.0).collect();
        assert_eq!(offs, vec![128, 384, 640, 896, 1664]);
        for (off, d) in &found {
            assert_eq!(
                (d.base, d.dtype, d.strides.clone()),
                (0, Dtype::Bf16, vec![0]),
                "{off}"
            );
            assert_eq!(d.l2_promotion, 128);
            if *off == 1664 {
                assert_eq!(
                    (d.dims.clone(), d.boxd.clone(), d.swizzle),
                    (vec![256, 128], vec![32, 128], 64)
                );
            } else {
                assert_eq!(
                    (d.dims.clone(), d.boxd.clone(), d.swizzle),
                    (vec![1, 1], vec![64, 128], 128)
                );
            }
        }
    }

    /// Rung 2e: the cuBLAS maps agree with the case's declared `tensormap` fields, except
    /// the stride of the extent-1 dim 2, which the descriptor stores as 0 and capture.py
    /// synthesized.
    #[test]
    fn r2_cublas_params() {
        let b = block(
            include_str!("../../../fixtures/r2_cublas_gemm/case.toml"),
            2560,
        );
        // (offset, dims, strides of dims 1.. as declared, box, base as captured)
        type Want = (usize, [u64; 3], [u64; 2], [u32; 3]);
        let want: [Want; 5] = [
            (192, [128, 128, 1], [256, 32768], [64, 16, 1]),
            (256, [128, 512, 1], [256, 131072], [64, 8, 1]),
            (832, [128, 512, 1], [512, 262144], [32, 8, 1]),
            (1216, [128, 128, 1], [256, 32768], [64, 16, 1]),
            (1280, [128, 512, 1], [256, 131072], [64, 8, 1]),
        ];
        for (off, dims, strides, boxd) in want {
            let d = decode(&b[off..off + 64]).unwrap();
            assert!(d.lib_flag);
            assert_eq!(d.dims, dims.to_vec(), "{off}");
            assert_eq!(d.strides, vec![strides[0], 0], "{off}");
            assert_eq!(d.boxd, boxd.to_vec(), "{off}");
            assert_eq!(d.swizzle, 128);
            assert_eq!(d.dtype, if off == 832 { Dtype::F32 } else { Dtype::Bf16 });
            assert_eq!(d.l2_promotion, 128);
        }
    }
}
