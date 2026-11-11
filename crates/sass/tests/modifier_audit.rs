/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! The SASS front-end's modifier gate (`modset.rs`, L-ROSETTA): sorted, and every modifier
//! outside the audited set of an opcode is refused. With `PTXV_ISA_TABLE` pointing at a
//! third-party sm_103a encoding table (JSON, `instructions.<form>.mod_groups`; not part of
//! this repository), every modifier group the table lists for a lifted opcode is tried: a group
//! with a token outside the audited set must be refused, and the tokens the table does not know
//! are printed (table gaps, e.g. `FMNMX.XORSIGN`).

#[test]
fn audited_set_is_sorted_and_refuses_the_rest() {
    let a = ptxv_sass::audited_mods();
    assert!(
        a.windows(2).all(|w| w[0].0 < w[1].0),
        "modset.rs must be sorted by opcode"
    );
    // Forms whose modifier was once silently ignored.
    for l in [
        "VIADD.16x2 R1, R2, -R3",
        "VIADD.32x1 R1, R2, R3",
        "STG.E.256 desc[UR46][R22.64], R24, R28",
        "LDG.E.256 R12, R4, desc[UR4][R2.64]",
        "STG.E.ENL2.256 desc[UR46][R22.64], RZ, R28",
        "ISETP.NE.AND.S64 P0, PT, R0, RZ, PT",
        "IMAD.B90.WIDE.U32.X R32, R32, R7, R40, P0",
    ] {
        assert!(
            ptxv_sass::lower_text(l, None).is_err(),
            "{l} must be refused"
        );
    }
    // Modeled since (probes/viadd16x2): two 16-bit lanes.
    assert!(ptxv_sass::lower_text("VIADD.16x2 R9, R2, 0x8001ffff", None).is_ok());
    // 32-byte accesses (probes/ldst256).
    assert!(ptxv_sass::lower_text("STG.E.ENL2.256 desc[UR46][R22.64], R24, R28", None).is_ok());
    assert!(ptxv_sass::lower_text("LDG.E.ENL2.256 R12, R4, desc[UR4][R2.64]", None).is_ok());
    // f64 "less or equal" / "less than or unordered" with each combiner (probes/dsetp_ltu_le)
    // and the unordered f32 compare as 1.0f / 0 (probes/fset_nan).
    for l in [
        "DSETP.LTU.OR P0, PT, R44, UR4, P0",
        "DSETP.LE.AND P0, PT, R26, 1, PT",
        "DSETP.LE.XOR P6, PT, -|R4|, UR6, !P1",
        "DSETP.LTU.AND P3, PT, R4, +INF , P1",
        "FSET.BF.NAN.AND R51, R26, R26, PT",
        "FSET.BF.NAN.AND R7, |R0|, -R7, !P0",
    ] {
        assert!(ptxv_sass::lower_text(l, None).is_ok(), "{l} must lift");
    }
    // The acceptance is narrow: the unprobed comparisons, a second predicate output, a
    // missing, doubled or foreign combiner, and FSET without .BF or with .OR / .XOR / .FTZ
    // are refused.
    for l in [
        "DSETP.EQU.AND P0, PT, R2, R4, PT",
        "DSETP.NUM.OR P0, PT, R2, R4, P1",
        "DSETP.LE.AND P0, P1, R2, R4, PT",
        "DSETP.LTU P0, PT, R2, R4, PT",
        "DSETP.LTU.OR.AND P0, PT, R2, R4, P1",
        "DSETP.LE.AND.FTZ P0, PT, R2, R4, PT",
        "DSETP.LE.MIN P0, PT, R2, R4, PT",
        "FSET.NAN.AND R1, R2, R3, PT",
        "FSET.BF.NAN R1, R2, R3, PT",
        "FSET.BF.NAN.OR R1, R2, R3, P0",
        "FSET.BF.NAN.XOR R1, R2, R3, P0",
        "FSET.BF.NAN.AND.FTZ R1, R2, R3, PT",
        "FSET.BF.NUM.AND R1, R2, R3, PT",
        "FSET.BF.LE.AND R1, R2, R3, PT",
    ] {
        assert!(
            ptxv_sass::lower_text(l, None).is_err(),
            "{l} must be refused"
        );
    }
    // `LEA.HI.X Rd, Ra, ~Rb, RZ, s, P` (probes/lea_hi_not): data registers, RZ as the upper
    // word, a thread predicate.
    assert!(ptxv_sass::lower_text("LEA.HI.X R54, R52, ~R55, RZ, 0x7, P0", None).is_ok());
    for l in [
        "LEA.HI.X R54, R52, ~UR5, RZ, 0x7, P0",
        "LEA.HI.X R54, R52, ~R55, R53, 0x7, P0",
        "LEA.HI.X R54, R52, ~R55, RZ, 0x7, !P0",
        "LEA.HI.X R54, R52, ~R55, RZ, 0x7, PT",
    ] {
        assert!(
            ptxv_sass::lower_text(l, None).is_err(),
            "{l} must be refused"
        );
    }
    // A negated register source of F2I (probes/f2i_neg): only `-R` on the 32-bit .NTZ forms.
    assert!(ptxv_sass::lower_text("F2I.U32.TRUNC.NTZ R38, -R38", None).is_ok());
    for l in [
        "F2I.TRUNC.NTZ R0, |R0|",
        "F2I.TRUNC.NTZ R0, -|R0|",
        "F2I.TRUNC.NTZ R0, -UR4",
        "F2I.U64.TRUNC R0, -R2",
        "F2I.F64.TRUNC R0, -R2",
        "F2I.S16.TRUNC.NTZ R0, -R2",
    ] {
        assert!(
            ptxv_sass::lower_text(l, None).is_err(),
            "{l} must be refused"
        );
    }
}

#[test]
fn table_modifier_groups_outside_the_audit_are_refused() {
    let Ok(path) = std::env::var("PTXV_ISA_TABLE") else {
        return;
    };
    let t: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let ins = t["instructions"].as_object().unwrap();
    let audited = ptxv_sass::audited_mods();
    let mut unknown = std::collections::BTreeSet::new();
    let mut tried = 0;
    for (form, v) in ins {
        let base = v["base_op"]
            .as_str()
            .unwrap_or(form.split('_').next().unwrap());
        let Some((_, ok)) = audited.iter().find(|(b, _)| *b == base) else {
            continue;
        };
        for g in v["mod_groups"].as_object().unwrap().keys() {
            let mods: Vec<&str> = g.split(',').filter(|m| !m.is_empty()).collect();
            if mods.iter().all(|m| ok.contains(m)) {
                continue;
            }
            // Operands do not matter: the gate runs before any operand is parsed.
            let line = format!("{}.{} R0, R1, R2", base, mods.join("."));
            tried += 1;
            assert!(
                ptxv_sass::lower_text(&line, None).is_err(),
                "{line}: a table modifier outside the audited set was accepted"
            );
        }
        unknown.insert(base);
    }
    for (b, ok) in audited {
        let known: std::collections::BTreeSet<&str> = ins
            .values()
            .filter(|v| v["base_op"].as_str() == Some(b))
            .flat_map(|v| {
                v["mod_groups"]
                    .as_object()
                    .unwrap()
                    .keys()
                    .flat_map(|g| g.split(',').map(str::to_string).collect::<Vec<_>>())
                    .collect::<Vec<_>>()
            })
            .map(|s| &*Box::leak(s.into_boxed_str()))
            .collect();
        let gaps: Vec<&&str> = ok.iter().filter(|m| !known.contains(**m)).collect();
        if !gaps.is_empty() {
            eprintln!("table has no {b} modifier {gaps:?}");
        }
    }
    eprintln!("{tried} table modifier groups outside the audit, all refused");
}
