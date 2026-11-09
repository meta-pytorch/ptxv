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
