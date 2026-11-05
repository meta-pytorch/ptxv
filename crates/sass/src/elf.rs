/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Just enough ELF64 to read a cubin's constant banks: section contents, the symbol table, and
//! the `R_CUDA_64` relocations the linker leaves in `.nv.constant4` (addresses of device
//! functions such as `__assertfail` and of string literals, for `CALL.ABS` and its arguments).

use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;

/// Relocated symbols get synthetic addresses `SYM_BASE + SYM_STRIDE * i`: the model never
/// dereferences them, it only recognizes them (a `CALL.ABS` to `__assertfail`).
pub const SYM_BASE: u64 = 0x7f00_0000_0000;
pub const SYM_STRIDE: u64 = 0x1_0000;

pub struct Section {
    pub name: String,
    pub typ: u32,
    pub link: u32,
    pub info: u32,
    pub data: Vec<u8>,
}

pub struct Elf {
    pub sections: Vec<Section>,
}

fn rd<const N: usize>(b: &[u8], o: usize) -> Result<[u8; N]> {
    Ok(b.get(o..o + N)
        .ok_or_else(|| anyhow!("truncated ELF"))?
        .try_into()?)
}

/// A constant bank's bytes and the synthetic address given to each relocated symbol
/// (address + addend, name).
pub type ConstBank = (Vec<u8>, Vec<(u64, String)>);

impl Elf {
    pub fn parse(b: &[u8]) -> Result<Elf> {
        if b.get(..6) != Some(b"\x7fELF\x02\x01") {
            bail!("not an ELF64 little-endian file");
        }
        let u16_at = |o| rd::<2>(b, o).map(u16::from_le_bytes);
        let u32_at = |o| rd::<4>(b, o).map(u32::from_le_bytes);
        let u64_at = |o| rd::<8>(b, o).map(u64::from_le_bytes);
        let shoff = u64_at(0x28)? as usize;
        let (shentsize, shnum, shstrndx) = (
            u16_at(0x3a)? as usize,
            u16_at(0x3c)? as usize,
            u16_at(0x3e)? as usize,
        );
        let mut raw = Vec::new();
        for i in 0..shnum {
            let o = shoff + i * shentsize;
            raw.push((
                u32_at(o)?,
                u32_at(o + 4)?,
                u64_at(o + 0x18)? as usize,
                u64_at(o + 0x20)? as usize,
                u32_at(o + 0x28)?,
                u32_at(o + 0x2c)?,
            ));
        }
        let stroff = raw.get(shstrndx).ok_or_else(|| anyhow!("bad shstrndx"))?.2;
        let mut sections = Vec::new();
        for &(name, typ, off, size, link, info) in &raw {
            let n0 = stroff + name as usize;
            let n1 = n0 + b[n0..].iter().position(|&c| c == 0).unwrap_or(0);
            let data = if typ == 8 {
                vec![] // NOBITS
            } else {
                b.get(off..off + size)
                    .ok_or_else(|| anyhow!("truncated ELF"))?
                    .to_vec()
            };
            sections.push(Section {
                name: String::from_utf8_lossy(&b[n0..n1]).into_owned(),
                typ,
                link,
                info,
                data,
            });
        }
        Ok(Elf { sections })
    }

    pub fn find(&self, name: &str) -> Option<usize> {
        self.sections.iter().position(|s| s.name == name)
    }

    /// Name of symbol `i` (or of its section, for unnamed section symbols).
    fn symbol(&self, i: usize) -> Result<String> {
        let st = self
            .sections
            .iter()
            .find(|s| s.typ == 2)
            .ok_or_else(|| anyhow!("no symbol table"))?;
        let strs = &self.sections[st.link as usize].data;
        let e = rd::<24>(&st.data, 24 * i)?;
        let n = u32::from_le_bytes(e[0..4].try_into()?) as usize;
        let shndx = u16::from_le_bytes(e[6..8].try_into()?) as usize;
        let end = n + strs[n..].iter().position(|&c| c == 0).unwrap_or(0);
        let name = String::from_utf8_lossy(&strs[n..end]).into_owned();
        Ok(if name.is_empty() {
            self.sections
                .get(shndx)
                .map(|s| s.name.clone())
                .unwrap_or_default()
        } else {
            name
        })
    }

    /// Constant bank `bank` as the kernel sees it: `.nv.constant<bank>.<kernel>`, else the
    /// module-wide `.nv.constant<bank>`, with relocations applied. Returns the bytes and the
    /// synthetic address given to each relocated symbol (address + addend, name).
    pub fn bank(&self, bank: u8, kernel: &str) -> Result<Option<ConstBank>> {
        let own = format!(".nv.constant{bank}.{kernel}");
        let module = format!(".nv.constant{bank}");
        let Some(si) = self.find(&own).or_else(|| self.find(&module)) else {
            return Ok(None);
        };
        let mut data = self.sections[si].data.clone();
        let mut syms: Vec<(u64, String)> = Vec::new();
        let mut names: Vec<String> = Vec::new();
        for rs in self
            .sections
            .iter()
            .filter(|s| s.typ == 4 && s.info as usize == si)
        {
            for e in rs.data.chunks(24) {
                let off = u64::from_le_bytes(e[0..8].try_into()?) as usize;
                let info = u64::from_le_bytes(e[8..16].try_into()?);
                let add = i64::from_le_bytes(e[16..24].try_into()?);
                let (typ, sym) = (info as u32, (info >> 32) as usize);
                if typ != 2 {
                    bail!("relocation type {typ} in {} not modeled", rs.name);
                }
                let name = self.symbol(sym)?;
                let k = match names.iter().position(|n| *n == name) {
                    Some(k) => k,
                    None => {
                        names.push(name.clone());
                        names.len() - 1
                    }
                };
                let a = (SYM_BASE + SYM_STRIDE * k as u64).wrapping_add(add as u64);
                data.get_mut(off..off + 8)
                    .ok_or_else(|| anyhow!("relocation past {}", self.sections[si].name))?
                    .copy_from_slice(&a.to_le_bytes());
                syms.push((a, name));
            }
        }
        Ok(Some((data, syms)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_mm_assertfail_slot() {
        let p = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/r3_grouped_mm/kernel.cubin"
        );
        // Third-party binary, not distributed (fixtures/VENDOR_FIXTURES.toml): skip if absent.
        let Ok(bytes) = std::fs::read(p) else {
            eprintln!("SKIP: vendor fixture files not present: {p} (tools/regen_fixture.py)");
            return;
        };
        let elf = Elf::parse(&bytes).unwrap();
        let (data, syms) = elf.bank(4, "no_such_kernel").unwrap().unwrap();
        let a = u64::from_le_bytes(data[0x160..0x168].try_into().unwrap());
        assert!(syms.iter().any(|(x, n)| *x == a && n == "__assertfail"));
    }
}
