// Copyright (c) Meta Platforms, Inc. and affiliates.
// This source code is licensed under the BSD-style license in the root LICENSE.

use std::{env, error::Error, fs};

use ptxv_sim::{fp, mufu_sample};

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    let [operation, input, output] = args.as_slice() else {
        return Err("expected OP INPUT.u32 OUTPUT.u32".into());
    };
    let bytes = fs::read(input)?;
    if bytes.is_empty() || !bytes.len().is_multiple_of(4) {
        return Err("input must contain nonempty, complete little-endian u32 words".into());
    }
    let words: Vec<u32> = bytes
        .chunks_exact(4)
        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let mut result = Vec::new();
    if operation == "mma.bf16" {
        if !words.len().is_multiple_of(33) {
            return Err("mma.bf16 requires rows of C, A[16], B[16] (33 u32 words)".into());
        }
        for row in words.chunks_exact(33) {
            if row[1..].iter().any(|&w| w > 0xffff) {
                return Err("BF16 input contains nonzero upper bits".into());
            }
            let bf = |w: u32| f32::from_bits(w << 16) as f64;
            let terms: Vec<_> = row[1..17]
                .iter()
                .zip(&row[17..])
                .map(|(&a, &b)| fp::Term::prod(bf(a), bf(b), true))
                .collect();
            let y = fp::DEFAULT_MMA.apply_terms(f32::from_bits(row[0]), &terms);
            result.extend_from_slice(&y.to_bits().to_le_bytes());
        }
    } else {
        let (func, fmt, ptx) = mufu_sample::op(operation)?;
        for x in words {
            let y = mufu_sample::model(func, fmt, ptx, x)
                .ok_or_else(|| format!("unsupported {operation} input {x:#010x}"))?;
            result.extend_from_slice(&y.to_le_bytes());
        }
    }
    fs::write(output, result)?;
    Ok(())
}
