/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::sync::Arc;

use ptxv_case::Case;
use ptxv_sim::build_launch;
use ptxv_sim::explore::run_once;
use ptxv_sim::machine::Machine;

fn machine(body: &str) -> Machine {
    let ptx = format!(
        ".version 8.8\n.target sm_103a\n.address_size 64\n\
         .visible .entry k(.param .u64 input, .param .u64 output) {{\n\
         .reg .b64 %p<2>, %x, %d; .reg .b32 %a, %b; .reg .b16 %h;\n\
         ld.param.u64 %p0, [input]; ld.param.u64 %p1, [output];\n\
         ld.global.u64 %x, [%p0]; ld.global.u32 %a, [%p0+8];\n\
         ld.global.u32 %b, [%p0+12]; {body}\n\
         st.global.u64 [%p1], %d; ret; }}"
    );
    let k = ptxv_ptx::parse_kernel(&ptx, Some("k")).unwrap();
    let case = Case {
        dir: Default::default(),
        m: serde_json::from_value(serde_json::json!({
            "name":"scalar", "description":"PTX scalar boundary semantics",
            "entry":"k", "grid":[1,1,1], "block":[1,1,1], "outputs":["out"],
            "buffers":[
                {"name":"x", "dtype":"u32", "shape":[4], "init":"zeros"},
                {"name":"out", "dtype":"u32", "shape":[2], "init":"zeros"}
            ],
            "params":[{"kind":"ptr", "buffer":"x"}, {"kind":"ptr", "buffer":"out"}],
            "variants":[]
        }))
        .unwrap(),
    };
    let launch = build_launch(&case, &k).unwrap();
    Machine::new(Arc::new(k), Arc::new(launch))
}

fn output(base: &Machine, x: u64, a: u32, b: u32) -> u64 {
    let mut m = base.clone();
    let bytes: Vec<u8> = x
        .to_le_bytes()
        .into_iter()
        .chain(a.to_le_bytes())
        .chain(b.to_le_bytes())
        .collect();
    for (i, byte) in bytes.into_iter().enumerate() {
        *m.gmem[0].bytes.get_mut(i) = byte;
    }
    let (m, findings, done) = run_once(m);
    // Notes (bfe64-isa-vs-gb300: the hardware rule differs) are not findings.
    assert!(
        done && findings.iter().all(|f| f.kind == "note"),
        "{findings:?}"
    );
    u64::from_le_bytes(std::array::from_fn(|i| *m.gmem[1].bytes.get(i)))
}

#[test]
fn bitfield_extract_obeys_source_width_and_sign_extension() {
    let edges = [0, 1, 15, 31, 32, 33, 63, 64, 65, 127, 255, 256, 257];
    for bits in [32u32, 64] {
        for signed in [false, true] {
            let base = machine(&format!(
                "bfe.{}{bits} %d, %x, %a, %b;",
                if signed { "s" } else { "u" }
            ));
            for x in [
                0,
                1,
                u64::MAX,
                0x8000000080000000,
                0xc0000000c0000000,
                0x123456789abcdef0,
            ] {
                for pos in edges {
                    for len in edges {
                        // pos and len mod 256 (PTX ISA; the PTX front's rule for both widths,
                        // GB300's bfe.64 differs: probes/bfe64).
                        let (p, l) = (pos & 255, len & 255);
                        let mut want = 0;
                        for bit in 0..bits {
                            let source_bit = if bit < l {
                                p + bit
                            } else {
                                p + l.saturating_sub(1)
                            };
                            let value = if l == 0 || (!signed && (bit >= l || source_bit >= bits)) {
                                0
                            } else {
                                (x >> source_bit.min(bits - 1)) & 1
                            };
                            want |= value << bit;
                        }
                        assert_eq!(
                            output(&base, x, pos, len),
                            want,
                            "bfe {}{bits} x={x:x} pos={pos} len={len}",
                            if signed { "s" } else { "u" }
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn narrow_shifts_keep_the_whole_32_bit_count() {
    for op in ["shl.b16", "shr.u16", "shr.s16"] {
        let base = machine(&format!("{op} %h, %x, %a; cvt.u64.u16 %d, %h;"));
        for x in [0u16, 1, 0x8000, 0xffff] {
            for count in [0u32, 1, 15, 16, 31, 65535, 65536, 65537, u32::MAX] {
                let want = match op {
                    "shl.b16" => x.checked_shl(count).unwrap_or(0),
                    "shr.u16" => x.checked_shr(count).unwrap_or(0),
                    _ => ((x as i16) >> count.min(15)) as u16,
                };
                assert_eq!(
                    output(&base, x as u64, count, 0),
                    want as u64,
                    "{op} {x:x} {count}"
                );
            }
        }
    }
}

#[test]
fn wide_multiply_extends_each_16_bit_source() {
    for signed in [false, true] {
        let base = machine(&format!(
            "mul.wide.{}16 %d, %x, %a;",
            if signed { "s" } else { "u" }
        ));
        for x in [0u16, 1, 2, 0x7fff, 0x8000, 0x8001, 0xffff] {
            for y in [0u16, 1, 2, 0x7fff, 0x8000, 0xffff] {
                let want = if signed {
                    (x as i16 as i32 * y as i16 as i32) as u32
                } else {
                    x as u32 * y as u32
                };
                assert_eq!(
                    output(&base, x as u64, y as u32, 0),
                    want as u64,
                    "{signed} {x:x} {y:x}"
                );
            }
        }
    }
}

#[test]
fn signed_saturating_arithmetic_clamps_overflow() {
    for sub in [false, true] {
        let base = machine(&format!(
            "{}.sat.s32 %d, %x, %a;",
            if sub { "sub" } else { "add" }
        ));
        for x in [i32::MIN, i32::MIN + 1, -1, 0, 1, i32::MAX - 1, i32::MAX] {
            for y in [i32::MIN, -1, 0, 1, i32::MAX] {
                let exact = if sub {
                    x as i64 - y as i64
                } else {
                    x as i64 + y as i64
                };
                let want = exact.clamp(i32::MIN as i64, i32::MAX as i64) as i32 as u32;
                assert_eq!(
                    output(&base, x as u32 as u64, y as u32, 0),
                    want as u64,
                    "{sub} {x} {y}"
                );
            }
        }
    }
}

#[test]
fn saturating_half_conversion_clamps_before_rounding() {
    let base = machine("cvt.rn.sat.f16.f32 %h, %x; cvt.u64.u16 %d, %h;");
    for (bits, want) in [
        (0x40000000, 0x3c00),
        (0xbf800000, 0),
        (0x80000000, 0),
        (0x3f000000, 0x3800),
        (0x7f800000, 0x3c00),
        (0xff800000, 0),
        (0x7fc00000, 0),
        (0x7f800001, 0),
    ] {
        assert_eq!(output(&base, bits, 0, 0), want, "{bits:x}");
    }
}

#[test]
fn nested_scopes_preserve_shadowed_and_inherited_bindings() {
    let base = machine(
        "{ .reg .b64 %local; mov.b64 %local, %x; \
           { .reg .b64 %local; mov.b64 %local, 0; } \
           { .reg .b64 %other; mov.b64 %other, %local; \
             { .reg .b64 %local; mov.b64 %local, 17; } \
             mov.b64 %d, %other; } }",
    );
    for x in [0, 1, u64::MAX, 0x123456789abcdef0] {
        assert_eq!(output(&base, x, 0, 0), x);
    }
}
