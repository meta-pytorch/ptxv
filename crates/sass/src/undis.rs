/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Encodings nvdisasm 13.0 prints nothing for (cubins from a newer ptxas, e.g. CuTe DSL's
//! bundled one): its listing skips the 16-byte slot. The few encodings decoded here are read
//! from the kernel's `.text` bytes and printed as a newer nvdisasm (13.4) prints them; every
//! fixed bit must match, so any other encoding stays a hole and the kernel is refused
//! ("instruction addresses are not contiguous").
//!
//! - `F2FP.SATFINITE.BF16.S2_6.UNPACK_B Rd, Ra[.reuse][.H1], Rb[.reuse].H0 | imm.H0`
//!   (`cvt.rn.satfinite.scaled::n2::ue8m0.bf16x2.s2f6x2`): fields read off the 1088 instances
//!   of the kernel DB plus probes/s2f6_cvt, each printed identically by nvdisasm 13.4
//!   (test `s2f6_decode_matches_nvdisasm_13_4`). The immediate is printed in hex (13.4 prints
//!   its bits as an f32 literal).
// ASSUME(L-NVDISASM): see ASSUMPTIONS.md

/// The disassembly of one 128-bit encoding (`lo` = bits 0..64, `hi` = bits 64..128), without
/// the trailing `;`, or None for an encoding not decoded here.
pub fn decode(lo: u64, hi: u64) -> Option<String> {
    let op = lo & 0xfff;
    if op != 0x63e && op != 0xe3e {
        return None;
    }
    // hi bits 8..24 = 0x0834, bit 25 set, bits 26..41 clear (bits 41.. are control bits and
    // reuse flags); lo bits 24..32 clear.
    if (hi >> 8) & 0xffff != 0x0834 || (hi >> 25) & 0xffff != 1 || (lo >> 24) & 0xff != 0 {
        return None;
    }
    let reg = |n: u64| {
        if n == 255 {
            "RZ".to_string()
        } else {
            format!("R{n}")
        }
    };
    let reuse = |bit: u32| if (hi >> bit) & 1 == 1 { ".reuse" } else { "" };
    let (a, s) = if op == 0x63e {
        // Register scale: data in lo bits 32..40, scale in hi bits 0..8.
        if lo >> 40 != 0 {
            return None;
        }
        let s = format!("{}{}.H0", reg(hi & 0xff), reuse(60));
        (reg((lo >> 32) & 0xff), s)
    } else {
        // Immediate scale in lo bits 32..64, data in hi bits 0..8.
        (reg(hi & 0xff), format!("{:#x}.H0", lo >> 32))
    };
    let a = format!(
        "{a}{}{}",
        reuse(59),
        if (hi >> 24) & 1 == 1 { ".H1" } else { "" }
    );
    let g = (lo >> 12) & 0xf;
    let guard = match (g & 7, g & 8 != 0) {
        (7, false) => String::new(),
        (7, true) => "@!PT ".to_string(),
        (p, neg) => format!("@{}P{p} ", if neg { "!" } else { "" }),
    };
    Some(format!(
        "{guard}F2FP.SATFINITE.BF16.S2_6.UNPACK_B {}, {a}, {s}",
        reg((lo >> 16) & 0xff)
    ))
}

#[cfg(test)]
mod tests {
    use super::decode;

    /// Encodings and nvdisasm 13.4's text (immediate rewritten to hex), from the kernel DB
    /// (FA4 int4 / fp8 KV, token mean pooling) and probes/s2f6_cvt.
    #[test]
    fn s2f6_decode_matches_nvdisasm_13_4() {
        for (lo, hi, text) in [
            (
                0x0000002c0002763e,
                0x190fe40002083417,
                "F2FP.SATFINITE.BF16.S2_6.UNPACK_B R2, R44.reuse, R23.reuse.H0",
            ),
            (
                0x0000002c0004763e,
                0x100fe40003083417,
                "F2FP.SATFINITE.BF16.S2_6.UNPACK_B R4, R44.H1, R23.reuse.H0",
            ),
            (
                0x0000002f0017763e,
                0x000fe20003083417,
                "F2FP.SATFINITE.BF16.S2_6.UNPACK_B R23, R47.H1, R23.H0",
            ),
            (
                0x000000280018863e,
                0x120fe20002083423,
                "@!P0 F2FP.SATFINITE.BF16.S2_6.UNPACK_B R24, R40, R35.reuse.H0",
            ),
            (
                0x0000003c0017b63e,
                0x080fe20002083424,
                "@!P3 F2FP.SATFINITE.BF16.S2_6.UNPACK_B R23, R60.reuse, R36.H0",
            ),
            (
                0x000000360022863e,
                0x080fe20003083424,
                "@!P0 F2FP.SATFINITE.BF16.S2_6.UNPACK_B R34, R54.reuse.H1, R36.H0",
            ),
            (
                0x00008181001c7e3e,
                0x000fe40002083416,
                "F2FP.SATFINITE.BF16.S2_6.UNPACK_B R28, R22, 0x8181.H0",
            ),
            (
                0x00008181001f7e3e,
                0x080fe4000308340f,
                "F2FP.SATFINITE.BF16.S2_6.UNPACK_B R31, R15.reuse.H1, 0x8181.H0",
            ),
            (
                0x000000ff00047e3e,
                0x000fc40002083405,
                "F2FP.SATFINITE.BF16.S2_6.UNPACK_B R4, R5, 0xff.H0",
            ),
            (
                0x0000fe0100057e3e,
                0x000fe40002083405,
                "F2FP.SATFINITE.BF16.S2_6.UNPACK_B R5, R5, 0xfe01.H0",
            ),
        ] {
            assert_eq!(decode(lo, hi).as_deref(), Some(text));
        }
        // Any other value of a fixed field is not decoded.
        assert_eq!(
            decode(0x0000002c0002763e | 1 << 24, 0x190fe40002083417),
            None
        );
        assert_eq!(
            decode(0x0000002c0002763e, 0x190fe40002083417 | 1 << 26),
            None
        );
        assert_eq!(decode(0x0000012c0002763e, 0x190fe40002083417), None);
        assert_eq!(decode(0x0000002c0002763f, 0x190fe40002083417), None);
    }
}
