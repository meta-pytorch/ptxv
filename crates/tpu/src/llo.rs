/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Text parser for libtpu LLO listings (port of fb/targets/tpu/tools/llo.py and the op
//! splitting of tools/llosim.py). Grammar, as observed in dumps from libtpu 0.0.50:
//!
//! ```text
//! listing  := header? bundle*
//! bundle   := ADDR TAG? ':' '>'* '{' op (';;' op)* '}' comment?
//! op       := (DEF '=')? MNEMONIC operands
//! comment  := '/*' ... '*/'   (nests; may span lines)
//! ```
//!
//! Predicates are printed as a leading `(pred)` / `(pred, %tied)` operand, or after the first
//! operand (`vst [mem] (!%p), ...`). Named operands are printed `/*name=*/value`.

use std::collections::BTreeMap;

/// One op of a listing, split into its parts.
#[derive(Clone, Debug, Default)]
pub struct RawOp {
    pub text: String,
    pub def: Option<String>,
    pub mn: String,
    /// (negated, token): `%p3_p1`, `0` or `1`.
    pub pred: Option<(bool, String)>,
    pub tied: Option<String>,
    pub args: Vec<String>,
    pub named: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Default)]
pub struct RawBundle {
    pub addr: u32,
    pub tag: Option<String>,
    pub depth: u8,
    pub ops: Vec<RawOp>,
    pub comments: Vec<String>,
}

/// Remove nested `/* */` comments; returns (code, comments).
pub fn strip_comments(text: &str) -> (String, Vec<String>) {
    let b = text.as_bytes();
    let (mut out, mut comments, mut cur) = (Vec::new(), Vec::new(), Vec::new());
    let mut depth = 0usize;
    let mut i = 0;
    while i < b.len() {
        if b[i..].starts_with(b"/*") {
            if depth == 0 {
                cur.clear();
            } else {
                cur.extend_from_slice(b"/*");
            }
            depth += 1;
            i += 2;
        } else if depth > 0 && b[i..].starts_with(b"*/") {
            depth -= 1;
            if depth == 0 {
                comments.push(String::from_utf8_lossy(&cur).trim().to_string());
            } else {
                cur.extend_from_slice(b"*/");
            }
            i += 2;
        } else {
            if depth > 0 {
                cur.push(b[i]);
            } else {
                out.push(b[i]);
            }
            i += 1;
        }
    }
    (String::from_utf8_lossy(&out).into_owned(), comments)
}

/// Split on `sep` outside brackets / parens / braces; trims and drops empty parts.
pub fn split_top(s: &str, sep: &str) -> Vec<String> {
    let mut parts = Vec::new();
    let mut depth = 0i32;
    let mut cur = String::new();
    let mut i = 0;
    let b = s.as_bytes();
    while i < b.len() {
        let c = b[i] as char;
        match c {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            _ => {}
        }
        if depth == 0 && s[i..].starts_with(sep) {
            parts.push(std::mem::take(&mut cur));
            i += sep.len();
            continue;
        }
        let ch = s[i..].chars().next().expect("should be a char boundary");
        cur.push(ch);
        i += ch.len_utf8();
    }
    parts.push(cur);
    parts
        .into_iter()
        .map(|p| p.trim().to_string())
        .filter(|p| !p.is_empty())
        .collect()
}

/// `/*name=*/value` -> `name=value`.
fn unname(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(i) = rest.find("/*") {
        out.push_str(&rest[..i]);
        let after = &rest[i + 2..];
        let ident_len = after
            .bytes()
            .take_while(|c| c.is_ascii_alphanumeric() || *c == b'_')
            .count();
        if ident_len > 0 && after[ident_len..].starts_with("=*/") {
            out.push_str(&after[..ident_len]);
            out.push('=');
            rest = after[ident_len + 3..].trim_start();
        } else {
            out.push_str("/*");
            rest = after;
        }
    }
    out.push_str(rest);
    out
}

fn is_word(s: &str) -> bool {
    !s.is_empty() && s.bytes().all(|c| c.is_ascii_alphanumeric() || c == b'_')
}

fn is_pred_tok(s: &str) -> bool {
    s == "0" || s == "1" || (s.starts_with('%') && is_word(&s[1..]))
}

/// Leading `(pred)` or `(pred, %tied)` operand of an argument string.
fn split_pred(args: &str) -> (Option<(bool, String)>, Option<String>, String) {
    let none = (None, None, args.to_string());
    let Some(inner) = args.strip_prefix('(') else {
        return none;
    };
    let Some(close) = inner.find(')') else {
        return none;
    };
    let body = &inner[..close];
    let (neg, body) = match body.strip_prefix('!') {
        Some(b) => (true, b),
        None => (false, body),
    };
    let mut parts = body.split(',').map(str::trim);
    let p = parts.next().unwrap_or("");
    let tied = parts.next();
    if parts.next().is_some() || !is_pred_tok(p) {
        return none;
    }
    if let Some(t) = tied
        && !(t.starts_with('%') && is_word(&t[1..]))
    {
        return none;
    }
    let rest = inner[close + 1..].trim_start();
    let rest = rest.strip_prefix(',').unwrap_or(rest).trim_start();
    (
        Some((neg, p.to_string())),
        tied.map(str::to_string),
        rest.to_string(),
    )
}

/// Parse one op's text (comments removed, whitespace collapsed).
pub fn parse_op(text: &str) -> RawOp {
    let mut op = RawOp {
        text: text.to_string(),
        ..Default::default()
    };
    let mut rest = text;
    if text.starts_with('%')
        && let Some(eq) = text.find(" = ")
        && !text[..eq].contains(' ')
    {
        op.def = Some(text[..eq].to_string());
        rest = &text[eq + 3..];
    }
    let (mn, a) = match rest.split_once(' ') {
        Some((m, a)) => (m, a.trim()),
        None => (rest, ""),
    };
    op.mn = mn.to_string();
    let (pred, tied, a) = split_pred(a);
    op.pred = pred;
    op.tied = tied;
    let mut pos: Vec<String> = Vec::new();
    for x0 in split_top(&a, ",") {
        let mut x = x0;
        // `[mem] name=val` (a named operand printed without a comma)
        loop {
            let t = x.trim_end();
            let Some(sp) = t.rfind(char::is_whitespace) else {
                break;
            };
            let (head, tail) = (t[..sp].trim_end(), &t[sp + 1..]);
            let Some((name, val)) = tail.split_once('=') else {
                break;
            };
            if !is_word(name) || val.is_empty() || head.is_empty() {
                break;
            }
            if head.matches('[').count() != head.matches(']').count() {
                break;
            }
            op.named.insert(name.to_string(), val.to_string());
            x = head.to_string();
        }
        if x.starts_with('[') && !x.ends_with(']') {
            let mut depth = 0i32;
            let mut end = x.len();
            for (ci, ch) in x.char_indices() {
                match ch {
                    '[' => depth += 1,
                    ']' => depth -= 1,
                    _ => {}
                }
                if depth == 0 {
                    end = ci + 1;
                    break;
                }
            }
            pos.push(x[..end].to_string());
            x = x[end..].trim().to_string();
            if x.is_empty() {
                continue;
            }
        }
        if !x.starts_with('[')
            && let Some((name, val)) = x.split_once('=')
            && is_word(name)
        {
            op.named.insert(name.to_string(), val.trim().to_string());
            continue;
        }
        pos.push(x);
    }
    for x in pos {
        let pm = x
            .strip_prefix('(')
            .and_then(|r| r.strip_suffix(')'))
            .map(|r| match r.strip_prefix('!') {
                Some(t) => (true, t),
                None => (false, r),
            })
            .filter(|(_, t)| t.starts_with('%') && is_word(&t[1..]));
        if let Some((neg, t)) = pm
            && op.pred.is_none()
        {
            op.pred = Some((neg, t.to_string()));
        } else if x.starts_with("[thread:") {
            continue;
        } else {
            op.args.push(x);
        }
    }
    op
}

fn bundle_head(line: &str) -> Option<(u32, Option<String>, u8, usize)> {
    let t = line.trim_start();
    let off = line.len() - t.len();
    let addr_len = t
        .bytes()
        .take_while(|c| c.is_ascii_hexdigit() || *c == b'x')
        .count();
    if addr_len == 0 {
        return None;
    }
    let a = &t[..addr_len];
    let addr = if let Some(h) = a.strip_prefix("0x") {
        u32::from_str_radix(h, 16).ok()?
    } else {
        a.parse().ok()?
    };
    let mut r = t[addr_len..].trim_start();
    let mut tag = None;
    for tg in ["LH", "LB", "LE", "PB", "PF", "CT"] {
        if let Some(x) = r.strip_prefix(tg)
            && (x.starts_with(':') || x.starts_with(' '))
        {
            tag = Some(tg.to_string());
            r = x.trim_start();
            break;
        }
    }
    let r = r.strip_prefix(':')?.trim_start();
    let depth = r.bytes().take_while(|c| *c == b'>').count();
    let r2 = r[depth..].trim_start();
    if !r2.starts_with('{') {
        return None;
    }
    let brace = line.len() - r2.len();
    let _ = off;
    Some((addr, tag, depth as u8, brace))
}

/// Parse a listing (any pass that prints bundles) into raw bundles.
pub fn parse(text: &str) -> Vec<RawBundle> {
    let mut out = Vec::new();
    let mut cur: Option<(RawBundle, Vec<String>)> = None;
    for line in text.lines() {
        match cur.as_mut() {
            None => {
                let Some((addr, tag, depth, _)) = bundle_head(line) else {
                    continue;
                };
                cur = Some((
                    RawBundle {
                        addr,
                        tag,
                        depth,
                        ..Default::default()
                    },
                    vec![line.to_string()],
                ));
            }
            Some((_, raw)) => raw.push(line.to_string()),
        }
        let (b, raw) = cur.as_ref().expect("set above");
        let joined = raw.join("\n");
        let (code, _) = strip_comments(&joined);
        let opens = code.matches('{').count();
        if opens == 0 || opens != code.matches('}').count() {
            continue;
        }
        let mut b = b.clone();
        let named = unname(&joined);
        let (code, comments) = strip_comments(&named);
        let lo = code.find('{').expect("counted above");
        let hi = code.rfind('}').expect("counted above");
        b.ops = split_top(&code[lo + 1..hi], ";;")
            .iter()
            .map(|o| parse_op(&o.split_whitespace().collect::<Vec<_>>().join(" ")))
            .collect();
        b.comments = comments;
        out.push(b);
        cur = None;
    }
    out
}

/// Region starts named in bundle comments (`Start region N`, `Start/End empty region N`).
pub fn regions(bundles: &[RawBundle]) -> BTreeMap<u32, u32> {
    let mut m = BTreeMap::new();
    for (bi, b) in bundles.iter().enumerate() {
        for c in &b.comments {
            for (key, skip) in [("Start region ", 13), ("Start/End empty region ", 23)] {
                let mut s = c.as_str();
                while let Some(i) = s.find(key) {
                    let r = &s[i + skip..];
                    let n: String = r.chars().take_while(char::is_ascii_digit).collect();
                    if let Ok(v) = n.parse() {
                        m.entry(v).or_insert(bi as u32);
                    }
                    s = &s[i + skip..];
                }
            }
        }
    }
    m
}

/// An `original` listing as one-op bundles plus its region map (region -> op index).
pub fn parse_original(text: &str) -> (Vec<RawBundle>, BTreeMap<u32, u32>) {
    let mut phys = Vec::new();
    let mut buf: Option<String> = None;
    for ln in text.lines() {
        if let Some(b) = buf.as_mut() {
            b.push(' ');
            b.push_str(ln.trim());
            if b.matches("/*").count() <= b.matches("*/").count() {
                phys.push(buf.take().expect("is some"));
            }
            continue;
        }
        if ln.matches("/*").count() > ln.matches("*/").count() {
            buf = Some(ln.to_string());
            continue;
        }
        phys.push(ln.to_string());
    }
    let mut out = Vec::new();
    let mut regions = BTreeMap::new();
    for ln in phys {
        let s = ln.trim();
        if s.is_empty() || s.starts_with("//") || s.starts_with("#allocation") || s == "_" {
            continue;
        }
        if let Some(r) = s.strip_prefix("$region") {
            let n: String = r.chars().take_while(char::is_ascii_digit).collect();
            if r[n.len()..].starts_with(':')
                && let Ok(v) = n.parse()
            {
                regions.entry(v).or_insert(out.len() as u32);
            }
            continue;
        }
        if !s.starts_with('%') {
            continue;
        }
        let named = unname(s);
        let (code, comments) = strip_comments(&named);
        let code = code.split_whitespace().collect::<Vec<_>>().join(" ");
        out.push(RawBundle {
            addr: out.len() as u32,
            tag: None,
            depth: 0,
            ops: vec![parse_op(&code)],
            comments,
        });
    }
    (out, regions)
}

/// One allocation of an `original` listing header.
#[derive(Clone, Debug)]
pub struct RawAlloc {
    pub id: u32,
    pub space: String,
    pub size: u64,
    pub offset: Option<u64>,
    pub fixed: bool,
    pub tag: String,
}

fn num(s: &str) -> Option<u64> {
    match s.strip_prefix("0x") {
        Some(h) => u64::from_str_radix(h, 16).ok(),
        None => s.parse().ok(),
    }
}

/// `#allocationN [shape = '...', space=S, size = Z, offset = O, attrs...]` lines.
pub fn parse_allocs(text: &str) -> Vec<RawAlloc> {
    let mut out = Vec::new();
    for ln in text.lines() {
        let s = ln.trim();
        let Some(r) = s.strip_prefix("#allocation") else {
            continue;
        };
        let id: String = r.chars().take_while(char::is_ascii_digit).collect();
        let Ok(id) = id.parse() else { continue };
        let Some(open) = r.find('[') else { continue };
        let body = &r[open + 1..r.rfind(']').unwrap_or(r.len())];
        let field = |k: &str| {
            body.find(k).map(|i| {
                body[i + k.len()..]
                    .trim_start()
                    .split(|c: char| c == ',' || c.is_whitespace())
                    .next()
                    .unwrap_or("")
                    .to_string()
            })
        };
        let Some(space) = field("space=") else {
            continue;
        };
        let size = field("size =").and_then(|v| num(&v)).unwrap_or(0);
        let offset = field("offset =").and_then(|v| num(&v));
        out.push(RawAlloc {
            id,
            space,
            size,
            offset,
            fixed: body.contains("fixed_abs"),
            tag: body
                .find("tag = '")
                .map(|i| body[i + 7..].split('\'').next().unwrap_or("").to_string())
                .unwrap_or_default(),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ops_and_predicates() {
        let o = parse_op("%s196_s16 = smov (!%p17_p0, %s21_s16), %s20_s15");
        assert_eq!(o.mn, "smov");
        assert_eq!(o.pred, Some((true, "%p17_p0".into())));
        assert_eq!(o.tied.as_deref(), Some("%s21_s16"));
        assert_eq!(o.args, vec!["%s20_s15"]);
        let o =
            parse_op("%104 = vst [vmem:[#allocation3] sm:$0xff] (!%p181_p4), vst_source=%v245_v0");
        assert_eq!(o.pred, Some((true, "%p181_p4".into())));
        assert_eq!(o.args, vec!["[vmem:[#allocation3] sm:$0xff]"]);
        assert_eq!(o.named["vst_source"], "%v245_v0");
        let o = parse_op("%35 = sst [smem:[#allocation3]] %s74_s0");
        assert_eq!(o.args, vec!["[smem:[#allocation3]]", "%s74_s0"]);
        let o = parse_op("%12 = sbr.rel (0) target = $region5");
        assert_eq!(o.pred, Some((false, "0".into())));
    }

    #[test]
    fn bundles_with_multiline_comments() {
        let t = "  0x1 LB: > { %a = sadd.s32 1, %s1_s2  ;;  %b = dma.x /*src=*/%s3_s4 /* c\nd */ } /* Start region 7 */\n";
        let b = parse(t);
        assert_eq!(b.len(), 1);
        assert_eq!(b[0].depth, 1);
        assert_eq!(b[0].tag.as_deref(), Some("LB"));
        assert_eq!(b[0].ops[1].named["src"], "%s3_s4");
        assert_eq!(regions(&b)[&7], 0);
    }
}
