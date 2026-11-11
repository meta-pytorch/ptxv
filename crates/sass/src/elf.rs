/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Just enough ELF64 to read a cubin's constant banks: section contents, the symbol table, and
//! the `R_CUDA_64` relocations the linker leaves in `.nv.constant4` (addresses of device
//! functions such as `__assertfail` and of string literals, for `CALL.ABS` and its arguments,
//! and of module data globals, `__device__` variables).

use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;

/// Relocated symbols get synthetic addresses `SYM_BASE + SYM_STRIDE * i`: the model never
/// dereferences them, it only recognizes them (a `CALL.ABS` to `__assertfail`). Relocations
/// naming a data global are also listed in `ConstBank::data_relocs`; the launch replaces those
/// words with the address of the global's buffer (`ptxv_sim::build_launch`).
pub const SYM_BASE: u64 = 0x7f00_0000_0000;
pub const SYM_STRIDE: u64 = 0x1_0000;

pub struct Section {
    pub name: String,
    pub typ: u32,
    /// Size in bytes (also for NOBITS sections, whose `data` is empty).
    pub size: usize,
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

/// A constant bank's bytes, relocated.
pub struct ConstBank {
    pub data: Vec<u8>,
    /// The synthetic address given to each relocated symbol (address + addend, name).
    pub syms: Vec<(u64, String)>,
    /// The relocated words naming a data global: (byte offset in the bank, global, addend).
    pub data_relocs: Vec<(u32, DataSym, i64)>,
}

/// A symbol table entry.
pub struct Symbol {
    /// Its name, or its section's name for an unnamed section symbol.
    pub name: String,
    pub value: u64,
    pub size: u64,
    pub shndx: usize,
}

/// A module data global: a symbol defined in `.nv.global` (zero-initialized) or
/// `.nv.global.init` (initialized), as `cuModuleGetGlobal` sees it.
#[derive(Clone, Debug, PartialEq)]
pub struct DataSym {
    pub name: String,
    pub size: u64,
    /// Initial bytes (`.nv.global.init`); None: zeros (`.nv.global`).
    pub init: Option<Vec<u8>>,
}

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
                size,
                data,
            });
        }
        Ok(Elf { sections })
    }

    pub fn find(&self, name: &str) -> Option<usize> {
        self.sections.iter().position(|s| s.name == name)
    }

    /// Symbol `i` of the symbol table.
    pub fn symbol(&self, i: usize) -> Result<Symbol> {
        let st = self
            .sections
            .iter()
            .find(|s| s.typ == 2)
            .ok_or_else(|| anyhow!("no symbol table"))?;
        let strs = &self.sections[st.link as usize].data;
        let e = rd::<24>(&st.data, 24 * i)?;
        let n = u32::from_le_bytes(e[0..4].try_into()?) as usize;
        let shndx = u16::from_le_bytes(e[6..8].try_into()?) as usize;
        let value = u64::from_le_bytes(e[8..16].try_into()?);
        let size = u64::from_le_bytes(e[16..24].try_into()?);
        let end = n + strs[n..].iter().position(|&c| c == 0).unwrap_or(0);
        let name = String::from_utf8_lossy(&strs[n..end]).into_owned();
        let name = if name.is_empty() {
            self.sections
                .get(shndx)
                .map(|s| s.name.clone())
                .unwrap_or_default()
        } else {
            name
        };
        Ok(Symbol {
            name,
            value,
            size,
            shndx,
        })
    }

    /// The data global a symbol names, if it is defined in `.nv.global` or `.nv.global.init`
    /// (an unnamed section symbol: the whole section).
    pub fn data_symbol(&self, s: &Symbol) -> Option<DataSym> {
        let sec = self.sections.get(s.shndx)?;
        if sec.name != ".nv.global" && sec.name != ".nv.global.init" {
            return None;
        }
        let (off, size) = if s.name == sec.name && s.size == 0 {
            (0, sec.size as u64)
        } else {
            (s.value, s.size)
        };
        let init = (sec.typ != 8).then(|| {
            let mut b = sec
                .data
                .get(off as usize..)
                .unwrap_or(&[])
                .iter()
                .take(size as usize)
                .copied()
                .collect::<Vec<u8>>();
            b.resize(size as usize, 0);
            b
        });
        Some(DataSym {
            name: s.name.clone(),
            size,
            init,
        })
    }

    /// Every data global whose address some constant bank of `kernel` holds, in order of
    /// first relocation (banks 1..18).
    pub fn kernel_globals(&self, kernel: &str) -> Result<Vec<DataSym>> {
        let mut out: Vec<DataSym> = Vec::new();
        for bank in 1..18u8 {
            if let Some(cb) = self.bank(bank, kernel)? {
                for (_, g, _) in cb.data_relocs {
                    if !out.iter().any(|x| x.name == g.name) {
                        out.push(g);
                    }
                }
            }
        }
        Ok(out)
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
        let mut data_relocs = Vec::new();
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
                let symbol = self.symbol(sym)?;
                if let Some(g) = self.data_symbol(&symbol) {
                    data_relocs.push((off as u32, g, add));
                }
                let name = symbol.name;
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
        Ok(Some(ConstBank {
            data,
            syms,
            data_relocs,
        }))
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
        let ConstBank { data, syms, .. } = elf.bank(4, "no_such_kernel").unwrap().unwrap();
        let a = u64::from_le_bytes(data[0x160..0x168].try_into().unwrap());
        assert!(syms.iter().any(|(x, n)| *x == a && n == "__assertfail"));
    }

    #[test]
    fn s_globals_data_relocs() {
        let p = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/s_globals/kernel.cubin"
        );
        let elf = Elf::parse(&std::fs::read(p).unwrap()).unwrap();
        let cb = elf.bank(4, "readout").unwrap().unwrap();
        let mut r: Vec<(u32, String, u64, bool)> = cb
            .data_relocs
            .iter()
            .map(|(o, g, a)| (*o, g.name.clone(), g.size, g.init.is_some() && *a == 0))
            .collect();
        r.sort();
        assert_eq!(
            r,
            [
                (0, "counter".to_string(), 4, false),
                (8, "table".to_string(), 32, true)
            ]
        );
        let t = elf.kernel_globals("bump").unwrap();
        let table = t.iter().find(|g| g.name == "table").unwrap();
        let words: Vec<u32> = table
            .init
            .as_ref()
            .unwrap()
            .chunks(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(words, [3, 1, 4, 1, 5, 9, 2, 6]);
    }
}
