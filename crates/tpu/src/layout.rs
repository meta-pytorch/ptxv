//! Operand memory layouts from the compiled HLO (`*-hlo.txt` next to the listings): the
//! `tpu_custom_call` operand types and its result type carry each operand's minor-to-major order
//! and tiling (`{1,0:T(8,128)}`, `{1,0:T(1,128)}`, `{1,0:T(8,128)(2,1)}`, `{1,0:T(32,128)(4,1)}`,
//! `{0:T(128)}`). Words are placed with XLA tiling semantics; nothing is inferred from array
//! shapes, and an operand whose layout cannot be read is unmodeled. ASSUME(TPU-L-OPERAND-LAYOUT)

use std::path::Path;

use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;
use ptxv_tpu_ir::OperandDecl;

use crate::data::Array;

/// One array type of the HLO: element type, dims, minor-to-major order and tiles.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Layout {
    pub dtype: String,
    pub shape: Vec<u64>,
    pub minor_to_major: Vec<usize>,
    pub tiles: Vec<Vec<u64>>,
    /// The type as printed, for messages.
    pub text: String,
}

impl Layout {
    /// The layout forms of the reference rule (tools/tpu_layout.py `Layout._check`, observed on
    /// 863 kernels): 1-D 32-bit with one tile T(n); otherwise row-major (or 2-D {0,1}) minor-to-
    /// major, first tile (a, 128), then exactly the packing sub-tile ((2,1) 16-bit, (4,1) 8-bit),
    /// a a multiple of the packing. Anything else is unmodeled.
    pub fn check(&self) -> Result<()> {
        let n = self.shape.len();
        let bits = match self.dtype.as_str() {
            "f32" | "s32" | "u32" => 32,
            "bf16" | "f16" | "s16" | "u16" => 16,
            "s8" | "u8" | "f8e4m3fn" | "f8e5m2" | "pred" => 8,
            d => bail!("element type {d} not handled ({})", self.text),
        };
        if n == 0 {
            bail!("scalar operand {}", self.text);
        }
        if n == 1 {
            if self.minor_to_major != [0]
                || self.tiles.len() != 1
                || self.tiles[0].len() != 1
                || bits != 32
            {
                bail!("1-D layout not handled: {}", self.text);
            }
            return Ok(());
        }
        let rowmajor: Vec<usize> = (0..n).rev().collect();
        if self.minor_to_major != rowmajor && !(n == 2 && self.minor_to_major == [0, 1]) {
            bail!("minor-to-major order not handled: {}", self.text);
        }
        if self.tiles.is_empty() || self.tiles[0].len() != 2 || self.tiles[0][1] != 128 {
            bail!("first tile not (a, 128): {}", self.text);
        }
        let pack: Vec<Vec<u64>> = match bits {
            32 => vec![],
            16 => vec![vec![2, 1]],
            _ => vec![vec![4, 1]],
        };
        if self.tiles[1..] != pack[..] {
            bail!("sub-tiles not the {bits}-bit packing: {}", self.text);
        }
        if self.tiles[0][0] % (32 / bits) != 0 {
            bail!("tile rows not a multiple of the packing: {}", self.text);
        }
        Ok(())
    }
}

/// Parse `f32[1,256]{1,0:T(1,128)S(1)}`.
pub fn parse_type(t: &str) -> Result<Layout> {
    let t = t.trim();
    let lb = t
        .find('[')
        .ok_or_else(|| anyhow!("hlo type {t}: no dims"))?;
    let rb = t
        .find(']')
        .ok_or_else(|| anyhow!("hlo type {t}: no dims"))?;
    let dtype = t[..lb].to_string();
    let shape: Vec<u64> = t[lb + 1..rb]
        .split(',')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().parse::<u64>())
        .collect::<Result<_, _>>()
        .map_err(|_| anyhow!("hlo type {t}: dims"))?;
    let rest = &t[rb + 1..];
    let (m2m, tiles) = if let Some(body) = rest.strip_prefix('{') {
        let body = body
            .strip_suffix('}')
            .ok_or_else(|| anyhow!("hlo type {t}: layout"))?;
        let (order, attrs) = body.split_once(':').unwrap_or((body, ""));
        let m2m: Vec<usize> = order
            .split(',')
            .filter(|s| !s.trim().is_empty())
            .map(|s| s.trim().parse::<usize>())
            .collect::<Result<_, _>>()
            .map_err(|_| anyhow!("hlo type {t}: minor-to-major"))?;
        let mut tiles = Vec::new();
        let mut a = attrs;
        while !a.is_empty() {
            let open = a
                .find('(')
                .ok_or_else(|| anyhow!("hlo layout {t}: `{a}`"))?;
            let close = a
                .find(')')
                .ok_or_else(|| anyhow!("hlo layout {t}: `{a}`"))?;
            let (kind, args) = (&a[..open], &a[open + 1..close]);
            match kind {
                k if k == "T" || (k.is_empty() && !tiles.is_empty()) => {
                    let tile: Vec<u64> = args
                        .split(',')
                        .map(|s| s.trim().parse::<u64>())
                        .collect::<Result<_, _>>()
                        .map_err(|_| anyhow!("hlo layout {t}: tile ({args})"))?;
                    if tile.contains(&0) {
                        bail!("hlo layout {t}: tile ({args})");
                    }
                    tiles.push(tile);
                }
                // memory space: the listing's inlined_call_operand gives the space
                "S" => {}
                _ => bail!("hlo layout {t}: attribute {kind}({args}) not modeled"),
            }
            a = &a[close + 1..];
        }
        (m2m, tiles)
    } else {
        bail!("hlo type {t}: no layout");
    };
    let mut sorted = m2m.clone();
    sorted.sort_unstable();
    if sorted != (0..shape.len()).collect::<Vec<_>>() {
        bail!("hlo type {t}: minor-to-major is not a permutation");
    }
    Ok(Layout {
        dtype,
        shape,
        minor_to_major: m2m,
        tiles,
        text: t.to_string(),
    })
}

/// Split a tuple type `(A, B, u32[]{:S(2)})` (or a single type) at top-level commas.
fn split_tuple(t: &str) -> Vec<String> {
    let t = t.trim();
    let Some(inner) = t.strip_prefix('(').and_then(|s| s.strip_suffix(')')) else {
        return vec![t.to_string()];
    };
    let (mut out, mut depth, mut cur) = (Vec::new(), 0i32, String::new());
    for ch in inner.chars() {
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ',' if depth == 0 => {
                out.push(std::mem::take(&mut cur).trim().to_string());
                continue;
            }
            _ => {}
        }
        cur.push(ch);
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// Drop `/*index=5*/`-style comments (printed inside long operand lists and tuple types).
fn strip_comments(t: &str) -> String {
    let mut out = String::with_capacity(t.len());
    let mut rest = t;
    while let Some(i) = rest.find("/*") {
        out.push_str(&rest[..i]);
        rest = rest[i..].find("*/").map_or("", |j| &rest[i + j + 2..]);
    }
    out.push_str(rest);
    out
}

/// `%name = TYPE op(...)`: (name, TYPE, rest from the op name).
fn split_def(line: &str) -> Option<(&str, &str, &str)> {
    let l = line.trim_start();
    let l = l.strip_prefix("ROOT ").unwrap_or(l);
    let l = l.strip_prefix('%')?;
    let (name, rest) = l.split_once(" = ")?;
    // type: up to the space before the op name (types contain no spaces outside tuples)
    let mut depth = 0i32;
    for (i, ch) in rest.char_indices() {
        match ch {
            '(' | '[' | '{' => depth += 1,
            ')' | ']' | '}' => depth -= 1,
            ' ' if depth == 0 => return Some((name, &rest[..i], &rest[i + 1..])),
            _ => {}
        }
    }
    None
}

/// Layouts of the custom call `%<kernel>[.N]` in the HLO dumps of `dir`: (input operand layouts,
/// output layouts).
pub fn kernel_layouts(dir: &Path, kernel: &str) -> Result<(Vec<Layout>, Vec<Layout>)> {
    let mut hits: Vec<(Vec<Layout>, Vec<Layout>)> = Vec::new();
    let mut files: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.to_string_lossy().ends_with("-hlo.txt"))
        .collect();
    files.sort();
    for f in files {
        let text = strip_comments(&std::fs::read_to_string(&f)?);
        let lines: Vec<&str> = text.lines().collect();
        for (li, line) in lines.iter().enumerate() {
            if !line.contains("custom_call_target=\"tpu_custom_call\"") {
                continue;
            }
            let Some((name, ty, rest)) = split_def(line) else {
                continue;
            };
            let base = name.split_once('.').map_or(name, |(b, sfx)| {
                if sfx.chars().all(|c| c.is_ascii_digit()) {
                    b
                } else {
                    name
                }
            });
            if base != kernel {
                continue;
            }
            let args = rest
                .strip_prefix("custom-call(")
                .and_then(|r| r.split_once(')'))
                .map(|x| x.0)
                .ok_or_else(|| anyhow!("{}: custom-call operands", f.display()))?;
            let mut ins = Vec::new();
            for a in args.split(',').map(str::trim).filter(|a| !a.is_empty()) {
                let a = a.strip_prefix('%').unwrap_or(a);
                let def = lines[..li]
                    .iter()
                    .rev()
                    .filter_map(|l| split_def(l))
                    .find(|(n, _, _)| *n == a)
                    .ok_or_else(|| anyhow!("{}: no definition of %{a}", f.display()))?;
                if def.1.starts_with('(') {
                    bail!("{}: operand %{a} has a tuple type", f.display());
                }
                ins.push(parse_type(def.1)?);
            }
            let outs = split_tuple(ty)
                .iter()
                .filter(|t| t.contains('['))
                .map(|t| parse_type(t))
                .collect::<Result<Vec<_>>>()?;
            hits.push((ins, outs));
        }
    }
    let Some(first) = hits.first().cloned() else {
        bail!(
            "no tpu_custom_call %{kernel} in the HLO dumps of {}",
            dir.display()
        )
    };
    if hits.iter().any(|h| *h != first) {
        bail!("%{kernel} appears with different layouts in several dumps");
    }
    Ok(first)
}

/// The layout of `decl` (inputs by operand index, outputs after the inputs), checked against the
/// declared element type and shape.
pub fn for_decl<'a>(
    layouts: &'a (Vec<Layout>, Vec<Layout>),
    decl: &OperandDecl,
) -> Result<&'a Layout> {
    let (ins, outs) = layouts;
    let i = decl.index as usize;
    let l = if decl.output {
        outs.get(i.checked_sub(ins.len()).unwrap_or(usize::MAX))
            .or(if outs.len() == 1 { outs.first() } else { None })
    } else {
        ins.get(i)
    }
    .ok_or_else(|| anyhow!("operand {i}: no HLO layout"))?;
    l.check()?;
    if l.shape != decl.shape || l.dtype != decl.dtype {
        bail!(
            "operand {i}: HLO type {} vs listing {}{:?}",
            l.text,
            decl.dtype,
            decl.shape
        );
    }
    Ok(l)
}

/// Per-dimension element-offset tables: offset(idx) = sum_d table[d][idx[d]]; total elements.
fn offset_tables(l: &Layout) -> Result<(Vec<Vec<u64>>, u64)> {
    let n = l.shape.len();
    // current dims (major to minor): (logical dim, divisor, modulus, extent)
    let mut dims: Vec<(usize, u64, Option<u64>, u64)> = (0..n)
        .rev()
        .map(|k| {
            let d = l.minor_to_major[k];
            (d, 1, None, l.shape[d])
        })
        .collect();
    for t in &l.tiles {
        let k = t.len();
        if k > dims.len() {
            bail!("layout {}: tile of rank {k} on {} dims", l.text, dims.len());
        }
        let split = dims.len() - k;
        let mut q = dims[..split].to_vec();
        let mut r = Vec::new();
        for (j, &(d, div, m, ext)) in dims[split..].iter().enumerate() {
            let tj = t[j];
            if m.is_some_and(|m| m % tj != 0) {
                bail!("layout {}: tile {tj} does not divide {m:?}", l.text);
            }
            q.push((d, div * tj, m.map(|m| m / tj), ext.div_ceil(tj)));
            r.push((d, div, Some(tj), tj));
        }
        q.extend(r);
        dims = q;
    }
    let mut strides = vec![0u64; dims.len()];
    let mut s = 1u64;
    for (i, dm) in dims.iter().enumerate().rev() {
        strides[i] = s;
        s *= dm.3;
    }
    let tables = (0..n)
        .map(|d| {
            (0..l.shape[d])
                .map(|x| {
                    dims.iter()
                        .zip(&strides)
                        .filter(|(dm, _)| dm.0 == d)
                        .map(|(&(_, div, m, _), &st)| {
                            let c = x / div;
                            m.map_or(c, |m| c % m) * st
                        })
                        .sum()
                })
                .collect()
        })
        .collect();
    Ok((tables, s))
}

fn elems_per_word(itemsize: usize) -> Result<u64> {
    match itemsize {
        1 | 2 | 4 => Ok(4 / itemsize as u64),
        _ => bail!("itemsize {itemsize}"),
    }
}

/// Number of words of an operand with layout `l`.
pub fn n_words(l: &Layout, itemsize: usize) -> Result<usize> {
    Ok(offset_tables(l)?.1.div_ceil(elems_per_word(itemsize)?) as usize)
}

/// Visit every element: (row-major logical index, element offset in memory).
fn each(l: &Layout, mut f: impl FnMut(usize, u64)) -> Result<()> {
    let (tables, _) = offset_tables(l)?;
    let n = l.shape.len();
    let total: u64 = l.shape.iter().product();
    let mut idx = vec![0u64; n];
    for flat in 0..total as usize {
        f(flat, (0..n).map(|d| tables[d][idx[d] as usize]).sum());
        for d in (0..n).rev() {
            idx[d] += 1;
            if idx[d] < l.shape[d] {
                break;
            }
            idx[d] = 0;
        }
    }
    Ok(())
}

pub fn to_words(a: &Array, l: &Layout) -> Result<Vec<u32>> {
    let epw = elems_per_word(a.itemsize)?;
    let bits = 8 * a.itemsize as u64;
    let mask = (u64::MAX >> (64 - bits)) as u32;
    let mut out = vec![0u32; n_words(l, a.itemsize)?];
    each(l, |i, e| {
        out[(e / epw) as usize] |= (a.data[i] & mask) << ((e % epw) * bits);
    })?;
    Ok(out)
}

pub fn from_words(words: &[u32], l: &Layout, itemsize: usize, descr: &str) -> Result<Array> {
    let mut a = Array::zeros(&l.shape, itemsize, descr);
    let epw = elems_per_word(itemsize)?;
    let bits = 8 * itemsize as u64;
    let mask = (u64::MAX >> (64 - bits)) as u32;
    each(l, |i, e| {
        a.data[i] = (words[(e / epw) as usize] >> ((e % epw) * bits)) & mask;
    })?;
    Ok(a)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data;

    fn arr(shape: &[u64], isz: usize) -> Array {
        let mut a = Array::zeros(shape, isz, "<u4");
        for (i, v) in a.data.iter_mut().enumerate() {
            *v = (i as u32).wrapping_mul(2_654_435_761) & (u32::MAX >> (32 - 8 * isz));
        }
        a
    }

    /// The HLO layouts of the corpus reproduce the hand-written layouts of data.rs
    /// (T(8,128), T(8,128)(2,1), T(32,128)(4,1), T(1,128)).
    #[test]
    fn hlo_layouts_match_known_packings() {
        for (t, isz) in [
            ("f32[24,260]{1,0:T(8,128)}", 4),
            ("bf16[24,130]{1,0:T(8,128)(2,1)S(1)}", 2),
            ("s8[40,130]{1,0:T(32,128)(4,1)}", 1),
            ("f32[1,256]{1,0:T(1,128)}", 4),
        ] {
            let l = parse_type(t).unwrap();
            let a = arr(&l.shape, isz);
            let w = to_words(&a, &l).unwrap();
            assert_eq!(w, data::to_words(&a, "vmem").unwrap(), "{t}");
            assert_eq!(from_words(&w, &l, isz, "<u4").unwrap(), a, "{t}");
        }
    }

    /// minor-to-major {0,1}: the physical array is the transpose.
    #[test]
    fn transposed_layout() {
        let l = parse_type("f32[128,8]{0,1:T(8,128)}").unwrap();
        let a = arr(&[128, 8], 4);
        let t = parse_type("f32[8,128]{1,0:T(8,128)}").unwrap();
        let mut at = Array::zeros(&[8, 128], 4, "<u4");
        for i in 0..128 {
            for j in 0..8 {
                at.data[j * 128 + i] = a.data[i * 8 + j];
            }
        }
        assert_eq!(to_words(&a, &l).unwrap(), to_words(&at, &t).unwrap());
    }

    #[test]
    fn one_d_tiles_and_tuples() {
        let l = parse_type("s32[64]{0:T(128)}").unwrap();
        assert_eq!(n_words(&l, 4).unwrap(), 128);
        let parts =
            split_tuple("(f32[8,128]{1,0:T(8,128)S(1)}, f32[8,128]{1,0:T(8,128)}, u32[]{:S(2)})");
        assert_eq!(parts.len(), 3);
        assert!(parse_type("f32[8,128]{1,0:T(8,128)L(2)}").is_err());
    }
}
