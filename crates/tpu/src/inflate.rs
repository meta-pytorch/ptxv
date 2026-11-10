/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Minimal DEFLATE decoder (RFC 1951) and `.npz` member reader, for the measured EUP tables
//! shipped as compressed numpy archives (fb/targets/tpu/tools/tpunum_f32_*.npz). No dependency.

use anyhow::Result;
use anyhow::anyhow;
use anyhow::bail;

struct Bits<'a> {
    d: &'a [u8],
    pos: usize,
    bit: u32,
    nbits: u32,
}

impl<'a> Bits<'a> {
    fn new(d: &'a [u8]) -> Self {
        Bits {
            d,
            pos: 0,
            bit: 0,
            nbits: 0,
        }
    }

    fn need(&mut self, n: u32) -> Result<()> {
        while self.nbits < n {
            let b = *self
                .d
                .get(self.pos)
                .ok_or_else(|| anyhow!("deflate: truncated"))?;
            self.pos += 1;
            self.bit |= (b as u32) << self.nbits;
            self.nbits += 8;
        }
        Ok(())
    }

    fn get(&mut self, n: u32) -> Result<u32> {
        if n == 0 {
            return Ok(0);
        }
        self.need(n)?;
        let v = self.bit & ((1u32 << n) - 1);
        self.bit >>= n;
        self.nbits -= n;
        Ok(v)
    }

    fn align(&mut self) {
        self.bit = 0;
        self.nbits = 0;
    }
}

/// Canonical Huffman decoder: counts per length and symbols sorted by code.
struct Huff {
    count: [u16; 16],
    sym: Vec<u16>,
}

impl Huff {
    fn new(lens: &[u8]) -> Self {
        let mut count = [0u16; 16];
        for &l in lens {
            count[l as usize] += 1;
        }
        count[0] = 0;
        let mut offs = [0u16; 16];
        for i in 1..16 {
            offs[i] = offs[i - 1] + count[i - 1];
        }
        let mut sym = vec![0u16; lens.len()];
        for (s, &l) in lens.iter().enumerate() {
            if l != 0 {
                sym[offs[l as usize] as usize] = s as u16;
                offs[l as usize] += 1;
            }
        }
        Huff { count, sym }
    }

    fn decode(&self, b: &mut Bits) -> Result<u16> {
        let (mut code, mut first, mut index) = (0i32, 0i32, 0i32);
        for len in 1..16 {
            code |= b.get(1)? as i32;
            let c = self.count[len] as i32;
            if code - c < first {
                return Ok(self.sym[(index + (code - first)) as usize]);
            }
            index += c;
            first += c;
            first <<= 1;
            code <<= 1;
        }
        bail!("deflate: bad code")
    }
}

const LBASE: [u16; 29] = [
    3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 15, 17, 19, 23, 27, 31, 35, 43, 51, 59, 67, 83, 99, 115, 131,
    163, 195, 227, 258,
];
const LEXT: [u8; 29] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 0,
];
const DBASE: [u16; 30] = [
    1, 2, 3, 4, 5, 7, 9, 13, 17, 25, 33, 49, 65, 97, 129, 193, 257, 385, 513, 769, 1025, 1537,
    2049, 3073, 4097, 6145, 8193, 12289, 16385, 24577,
];
const DEXT: [u8; 30] = [
    0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
    13,
];

/// Inflate a raw DEFLATE stream.
pub fn inflate(d: &[u8]) -> Result<Vec<u8>> {
    let mut b = Bits::new(d);
    let mut out: Vec<u8> = Vec::new();
    loop {
        let last = b.get(1)?;
        match b.get(2)? {
            0 => {
                b.align();
                let p = b.pos;
                if p + 4 > d.len() {
                    bail!("deflate: truncated stored block");
                }
                let len = u16::from_le_bytes([d[p], d[p + 1]]) as usize;
                out.extend_from_slice(
                    d.get(p + 4..p + 4 + len)
                        .ok_or_else(|| anyhow!("deflate: truncated"))?,
                );
                b.pos = p + 4 + len;
            }
            t @ (1 | 2) => {
                let (lit, dist) = if t == 1 {
                    let mut l = [0u8; 288];
                    l[..144].fill(8);
                    l[144..256].fill(9);
                    l[256..280].fill(7);
                    l[280..].fill(8);
                    (Huff::new(&l), Huff::new(&[5u8; 30]))
                } else {
                    let hlit = b.get(5)? as usize + 257;
                    let hdist = b.get(5)? as usize + 1;
                    let hclen = b.get(4)? as usize + 4;
                    const ORD: [usize; 19] = [
                        16, 17, 18, 0, 8, 7, 9, 6, 10, 5, 11, 4, 12, 3, 13, 2, 14, 1, 15,
                    ];
                    let mut cl = [0u8; 19];
                    for &o in ORD.iter().take(hclen) {
                        cl[o] = b.get(3)? as u8;
                    }
                    let ch = Huff::new(&cl);
                    let mut lens = Vec::with_capacity(hlit + hdist);
                    while lens.len() < hlit + hdist {
                        let s = ch.decode(&mut b)?;
                        match s {
                            0..=15 => lens.push(s as u8),
                            16 => {
                                let p = *lens.last().ok_or_else(|| anyhow!("deflate: repeat"))?;
                                for _ in 0..3 + b.get(2)? {
                                    lens.push(p);
                                }
                            }
                            17 => lens.extend(std::iter::repeat_n(0, 3 + b.get(3)? as usize)),
                            _ => lens.extend(std::iter::repeat_n(0, 11 + b.get(7)? as usize)),
                        }
                    }
                    (Huff::new(&lens[..hlit]), Huff::new(&lens[hlit..]))
                };
                loop {
                    let s = lit.decode(&mut b)? as usize;
                    if s < 256 {
                        out.push(s as u8);
                    } else if s == 256 {
                        break;
                    } else {
                        let i = s - 257;
                        let len = LBASE[i] as usize + b.get(LEXT[i] as u32)? as usize;
                        let di = dist.decode(&mut b)? as usize;
                        let dd = DBASE[di] as usize + b.get(DEXT[di] as u32)? as usize;
                        if dd > out.len() {
                            bail!("deflate: distance too far");
                        }
                        let start = out.len() - dd;
                        for k in 0..len {
                            out.push(out[start + k]);
                        }
                    }
                }
            }
            _ => bail!("deflate: reserved block type"),
        }
        if last == 1 {
            return Ok(out);
        }
    }
}

/// The bytes of member `name` (e.g. `diff.npy`) of a zip archive (`.npz`): stored or deflated.
pub fn zip_member(z: &[u8], name: &str) -> Result<Vec<u8>> {
    let u16le = |i: usize| u16::from_le_bytes([z[i], z[i + 1]]) as usize;
    let u32le = |i: usize| u32::from_le_bytes([z[i], z[i + 1], z[i + 2], z[i + 3]]) as usize;
    // central directory: end record "PK\x05\x06"
    let eocd = (0..z.len().saturating_sub(21))
        .rev()
        .find(|&i| &z[i..i + 4] == b"PK\x05\x06")
        .ok_or_else(|| anyhow!("zip: no end record"))?;
    let (n, mut p) = (u16le(eocd + 10), u32le(eocd + 16));
    for _ in 0..n {
        if &z[p..p + 4] != b"PK\x01\x02" {
            bail!("zip: bad central entry");
        }
        let (method, csize) = (u16le(p + 10), u32le(p + 20));
        let (nlen, xlen, clen) = (u16le(p + 28), u16le(p + 30), u16le(p + 32));
        let off = u32le(p + 42);
        let fname = std::str::from_utf8(&z[p + 46..p + 46 + nlen]).unwrap_or("");
        if fname == name {
            let lh = off;
            let data = lh + 30 + u16le(lh + 26) + u16le(lh + 28);
            let raw = z
                .get(data..data + csize)
                .ok_or_else(|| anyhow!("zip: truncated member"))?;
            return match method {
                0 => Ok(raw.to_vec()),
                8 => inflate(raw),
                m => bail!("zip: method {m}"),
            };
        }
        p += 46 + nlen + xlen + clen;
    }
    bail!("zip: no member {name}")
}

/// Member names of a zip archive, in central-directory order.
pub fn zip_names(z: &[u8]) -> Result<Vec<String>> {
    let u16le = |i: usize| u16::from_le_bytes([z[i], z[i + 1]]) as usize;
    let u32le = |i: usize| u32::from_le_bytes([z[i], z[i + 1], z[i + 2], z[i + 3]]) as usize;
    let eocd = (0..z.len().saturating_sub(21))
        .rev()
        .find(|&i| &z[i..i + 4] == b"PK\x05\x06")
        .ok_or_else(|| anyhow!("zip: no end record"))?;
    let (n, mut p) = (u16le(eocd + 10), u32le(eocd + 16));
    let mut out = Vec::with_capacity(n);
    for _ in 0..n {
        if &z[p..p + 4] != b"PK\x01\x02" {
            bail!("zip: bad central entry");
        }
        let (nlen, xlen, clen) = (u16le(p + 28), u16le(p + 30), u16le(p + 32));
        out.push(String::from_utf8_lossy(&z[p + 46..p + 46 + nlen]).into_owned());
        p += 46 + nlen + xlen + clen;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stored_and_fixed() {
        // "hello" as a stored block, and as a fixed-Huffman block (zlib output without header)
        assert_eq!(
            inflate(&[1, 5, 0, 0xFA, 0xFF, b'h', b'e', b'l', b'l', b'o']).unwrap(),
            b"hello"
        );
        assert_eq!(
            inflate(&[0xCB, 0x48, 0xCD, 0xC9, 0xC9, 0x07, 0x00]).unwrap(),
            b"hello"
        );
    }
}
