/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! Hash maps and sets with a fixed hasher, for all of errdist. std's maps draw random keys per
//! process, so their iteration order, and every floating-point sum, tie-break or site order
//! taken in that order, changed from run to run; an estimate must be reproducible.
//! `HashMap::new()` does not exist for these types: build them with `default()`.

use std::hash::BuildHasherDefault;
use std::hash::DefaultHasher;

pub type HashMap<K, V> = std::collections::HashMap<K, V, BuildHasherDefault<DefaultHasher>>;
pub type HashSet<K> = std::collections::HashSet<K, BuildHasherDefault<DefaultHasher>>;
/// A map with another fixed hasher (a multiplicative hash on integer keys).
pub type HashMapWith<K, V, H> = std::collections::HashMap<K, V, BuildHasherDefault<H>>;

#[cfg(test)]
mod tests {
    /// Every map and set in errdist is one of these (std's would bring back the random order).
    #[test]
    fn errdist_maps_use_the_fixed_hasher() {
        let mut bad = Vec::new();
        let mut dirs = vec![std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/errdist")];
        while let Some(d) = dirs.pop() {
            for e in std::fs::read_dir(&d).unwrap() {
                let p = e.unwrap().path();
                if p.is_dir() {
                    dirs.push(p);
                } else if p.extension().is_some_and(|x| x == "rs") && !p.ends_with("det.rs") {
                    let s = std::fs::read_to_string(&p).unwrap();
                    for (i, l) in s.lines().enumerate() {
                        if l.contains("std::collections")
                            && (l.contains("HashMap") || l.contains("HashSet"))
                        {
                            bad.push(format!("{}:{}", p.display(), i + 1));
                        }
                    }
                }
            }
        }
        assert!(bad.is_empty(), "std hash maps in errdist: {bad:?}");
    }
}
