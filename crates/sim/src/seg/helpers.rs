/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::BTreeMap;
use std::ops::Range;
use std::sync::OnceLock;

use ptxv_ir::*;
use serde::Deserialize;
use serde_json::Value;

use crate::machine::Machine;

#[derive(Deserialize)]
struct Template {
    inputs: Vec<u16>,
    result: u16,
    return_register: u16,
    code: Vec<(Op, Pred, bool)>,
}

struct Contract {
    inputs: Vec<Reg>,
    result: Reg,
    return_register: Reg,
}

// Match every instruction, modifier, constant, guard and internal edge. Registers may be
// renamed injectively; the contract's argument/result registers use that same renaming.
fn match_value(
    want: &Value,
    got: &Value,
    regs: &mut BTreeMap<(String, u64), (String, u64)>,
) -> bool {
    match (want, got) {
        (Value::Object(a), Value::Object(b))
            if a.len() == 2 && a.contains_key("file") && a.contains_key("idx") =>
        {
            let Some((af, ai, bf, bi)) = a["file"]
                .as_str()
                .zip(a["idx"].as_u64())
                .zip(
                    b.get("file")
                        .and_then(Value::as_str)
                        .zip(b.get("idx").and_then(Value::as_u64)),
                )
                .map(|((af, ai), (bf, bi))| (af, ai, bf, bi))
            else {
                return false;
            };
            if af != bf || (ai == 255 || bi == 255) && ai != bi {
                return false;
            }
            let key = (af.to_owned(), ai);
            let val = (bf.to_owned(), bi);
            if let Some(old) = regs.get(&key) {
                return *old == val;
            }
            if regs.values().any(|v| *v == val) {
                return false;
            }
            regs.insert(key, val);
            true
        }
        (Value::Object(a), Value::Object(b)) => {
            a.len() == b.len()
                && a.iter()
                    .all(|(k, v)| b.get(k).is_some_and(|w| match_value(v, w, regs)))
        }
        (Value::Array(a), Value::Array(b)) => {
            a.len() == b.len() && a.iter().zip(b).all(|(v, w)| match_value(v, w, regs))
        }
        _ => want == got,
    }
}

fn contract(m: &Machine, target: usize, reciprocal: bool) -> Option<Contract> {
    static DIV: OnceLock<Vec<Template>> = OnceLock::new();
    static RCP: OnceLock<Vec<Template>> = OnceLock::new();
    let templates = if reciprocal {
        RCP.get_or_init(|| {
            serde_json::from_str(include_str!("rcp_rn_helper.json"))
                .expect("valid reciprocal helper template")
        })
    } else {
        DIV.get_or_init(|| {
            serde_json::from_str(include_str!("div_rn_helper.json"))
                .expect("valid division helper template")
        })
    };
    templates
        .iter()
        .find_map(|template| match_contract(&m.k.code, target, template))
}

fn match_contract(instructions: &[Inst], target: usize, template: &Template) -> Option<Contract> {
    let code = instructions.get(target..target.checked_add(template.code.len())?)?;
    if instructions.iter().enumerate().any(|(pc, i)| {
        if (target..target + code.len()).contains(&pc) {
            return false;
        }
        match &i.op {
            Op::Bra { target: q, .. } => (target..target + code.len()).contains(q),
            Op::Call { target: q } => *q != target && (target..target + code.len()).contains(q),
            Op::Brx { targets, .. } => targets
                .iter()
                .any(|q| (target..target + code.len()).contains(q)),
            _ => false,
        }
    }) {
        return None;
    }
    let mut regs = BTreeMap::new();
    let mut barriers = BTreeMap::new();
    for ((want, guard, uniform), got) in template.code.iter().zip(code) {
        let mut op = got.op.clone();
        if let Op::Bra { target: q, .. } = &mut op {
            *q = q.checked_sub(target)?;
            if *q >= code.len() {
                return None;
            }
        }
        match (want, &op) {
            (Op::Bssy { b: a }, Op::Bssy { b }) | (Op::Bsync { b: a }, Op::Bsync { b }) => {
                if barriers.get(a).is_some_and(|old| old != b)
                    || !barriers.contains_key(a) && barriers.values().any(|old| old == b)
                {
                    return None;
                }
                barriers.insert(*a, *b);
                if let Op::Bssy { b } | Op::Bsync { b } = &mut op {
                    *b = *a;
                }
            }
            _ => {}
        }
        if !match_value(
            &serde_json::to_value((want, guard, uniform)).ok()?,
            &serde_json::to_value((&op, got.guard, got.uniform)).ok()?,
            &mut regs,
        ) {
            return None;
        }
    }
    let mapped = |idx| {
        let (_, idx) = regs.get(&("R".to_owned(), idx))?;
        Some(Reg {
            file: File::R,
            idx: (*idx).try_into().ok()?,
        })
    };
    Some(Contract {
        inputs: template
            .inputs
            .iter()
            .map(|&i| mapped(i as u64))
            .collect::<Option<_>>()?,
        result: mapped(template.result as u64)?,
        return_register: mapped(template.return_register as u64)?,
    })
}

#[derive(Clone, PartialEq)]
enum Binding {
    A,
    B,
    Result,
    Constant(u64),
}

fn moved(op: &Op, r: Reg) -> Option<&Val> {
    match op {
        Op::Mov { d, a } if *d == r => Some(a),
        Op::Imad {
            d,
            a: Val::Imm(0),
            b: Val::Imm(0),
            c,
            neg_c: false,
            not_c: false,
            hi: false,
            wide: false,
            cin: None,
            cout: None,
            ..
        } if *d == r => Some(c),
        _ => None,
    }
}

fn binding(
    code: &[Inst],
    v: &Val,
    before: usize,
    floor: usize,
    inputs: &[(Reg, Binding)],
    depth: usize,
) -> Option<Binding> {
    if depth > 32 {
        return None;
    }
    match v {
        Val::Imm(c) => Some(Binding::Constant(*c)),
        Val::Reg(r, 1) => {
            if let Some(p) = (floor..before)
                .rev()
                .find(|&p| code[p].op.dst_regs().contains(r))
            {
                if !code[p].guard.is_always() {
                    return None;
                }
                return binding(code, moved(&code[p].op, *r)?, p, floor, inputs, depth + 1);
            }
            if let Some((_, b)) = inputs.iter().find(|(q, _)| q == r) {
                return Some(b.clone());
            }
            if floor > 0 {
                return binding(code, v, floor, 0, &[], depth + 1);
            }
            None
        }
        _ => None,
    }
}

// ASSUME(N-SEG): the compiler's complete slow helper must implement the recognized ABI.
// A helper name alone provides no semantic contract.
pub(super) fn validates_call(
    m: &Machine,
    call: usize,
    body: Range<usize>,
    floor: usize,
    a: Option<Reg>,
    b: &Val,
    q: Reg,
) -> bool {
    let code = &m.k.code;
    let (lo, hi) = (body.start, body.end);
    if code.iter().enumerate().any(|(pc, i)| {
        if (floor..=hi).contains(&pc) {
            return false;
        }
        match &i.op {
            Op::Bra { target, .. } | Op::Call { target } => (floor + 1..=hi).contains(target),
            Op::Brx { targets, .. } => targets.iter().any(|p| (floor + 1..=hi).contains(p)),
            _ => false,
        }
    }) {
        return false;
    }
    let Op::Call { target } = code[call].op else {
        return false;
    };
    if !code[call].guard.is_always() {
        return false;
    }
    let Some(c) = contract(m, target, a.is_none()) else {
        return false;
    };
    let mut inputs = vec![];
    if let Some(a) = a {
        inputs.push((a, Binding::A));
    }
    if let Val::Reg(b, 1) = b {
        inputs.push((*b, Binding::B));
    }
    let value = |r| Val::Reg(r, 1);
    let read = |r| binding(code, &value(r), call, floor, &inputs, 0);
    let expected_b = match b {
        Val::Imm(c) => Binding::Constant(*c),
        _ => Binding::B,
    };
    let args_ok = if a.is_some() {
        c.inputs.len() == 2
            && read(c.inputs[0]) == Some(Binding::A)
            && read(c.inputs[1]) == Some(expected_b)
    } else {
        c.inputs.len() == 1 && read(c.inputs[0]) == Some(expected_b)
    };
    if !args_ok
        || read(c.return_register) != code.get(call + 1).map(|i| Binding::Constant(i.loc as u64))
    {
        return false;
    }
    if (lo..call).chain(call + 1..hi).any(|p| {
        !code[p].guard.is_always()
            || code[p]
                .op
                .dst_regs()
                .iter()
                .any(|&r| moved(&code[p].op, r).is_none())
            || code[p].op.dst_regs().is_empty() && !matches!(code[p].op, Op::Nop)
    }) {
        return false;
    }
    binding(
        code,
        &value(q),
        hi,
        call + 1,
        &[(c.result, Binding::Result)],
        0,
    ) == Some(Binding::Result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_catalogued_helper_checks_all_fma_operands_and_modifiers() {
        for source in [
            include_str!("div_rn_helper.json"),
            include_str!("rcp_rn_helper.json"),
        ] {
            let templates: Vec<Template> = serde_json::from_str(source).unwrap();
            for template in &templates {
                let code: Vec<_> = template
                    .code
                    .iter()
                    .map(|(op, guard, uniform)| Inst {
                        op: op.clone(),
                        guard: *guard,
                        uniform: *uniform,
                        loc: 0,
                        text: String::new(),
                    })
                    .collect();
                let c = match_contract(&code, 0, template).unwrap();
                assert_eq!(c.result.idx, template.result);
                assert_eq!(c.return_register.idx, template.return_register);
                assert_eq!(
                    c.inputs.iter().map(|r| r.idx).collect::<Vec<_>>(),
                    template.inputs
                );
                for (pc, inst) in code.iter().enumerate() {
                    if !matches!(inst.op, Op::Ffma { .. }) {
                        continue;
                    }
                    for field in 0..6 {
                        let mut mutant = code.clone();
                        let Op::Ffma {
                            a,
                            b,
                            c,
                            neg,
                            rnd,
                            ftz,
                            ..
                        } = &mut mutant[pc].op
                        else {
                            unreachable!()
                        };
                        match field {
                            0 => {
                                *a = if *a == Val::Imm(0) {
                                    Val::Imm(1)
                                } else {
                                    Val::Imm(0)
                                }
                            }
                            1 => {
                                *b = if *b == Val::Imm(0) {
                                    Val::Imm(1)
                                } else {
                                    Val::Imm(0)
                                }
                            }
                            2 => {
                                *c = Some(if *c == Some(Val::Imm(0)) {
                                    Val::Imm(1)
                                } else {
                                    Val::Imm(0)
                                })
                            }
                            3 => neg[0] = !neg[0],
                            4 => *rnd = if *rnd == Rnd::Rn { Rnd::Rz } else { Rnd::Rn },
                            _ => *ftz = !*ftz,
                        }
                        assert!(
                            templates
                                .iter()
                                .all(|t| match_contract(&mutant, 0, t).is_none()),
                            "pc {pc}, field {field}"
                        );
                    }
                }
            }
        }
    }
}
