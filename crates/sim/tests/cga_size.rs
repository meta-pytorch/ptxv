/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! SR_CgaSize and the cluster special registers against GB300 (probes/cga_size,
//! S-CGA-SHAPE). Each CTA's thread 0 records the raw SR_CgaSize and the PTX cluster registers
//! (which ptxas reads from the driver table at `c[0x0][x - 0xa0 * SR_CgaSize]`). Recorded for
//! clusters of 1..16 without a preferred dimension (SR_CgaSize 0 throughout) and for launches
//! with one (SR_CgaSize 1 in preferred clusters, 0 in regular ones, mixed in one launch). The
//! SASS front-end must reproduce every CTA's record: launches without a preferred dimension
//! from the regular shape, the others from the shape the CTA ran in on the GPU. Every group of
//! CTAs of the preferred size is wholly one shape on the GPU.

use std::path::PathBuf;
use std::sync::Arc;

use ptxv_case::Case;
use ptxv_sim::build_launch;
use ptxv_sim::machine::Machine;

fn root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

const W: usize = 16;

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4)
        .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

#[test]
fn cga_size_matches_gb300() {
    let probe = root().join("probes/cga_size");
    let dir = std::env::temp_dir().join(format!("ptxv-test-cga-{}", std::process::id()));
    let mut configs = 0;
    let mut mixed = 0;
    let mut names: Vec<_> = std::fs::read_dir(probe.join("expected"))
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect();
    names.sort();
    for path in names {
        let name = path.file_stem().unwrap().to_str().unwrap().to_string();
        // g<grid>_c<cluster>_p<preferred>_s<large smem>
        let f: Vec<u32> = name.split('_').map(|p| p[1..].parse().unwrap()).collect();
        let (grid, cl, pref) = (f[0], f[1], f[2]);
        let mut t = format!(
            "name = \"cga_size_{name}\"\ndescription = \"probes/cga_size {name}\"\ncubin = \"{}\"\nentry = \"k\"\ngrid = [{grid}, 1, 1]\n\
             block = [32, 1, 1]\noutputs = [\"out\"]\n",
            probe.join("patched.cubin").display()
        );
        if cl > 0 {
            t += &format!("cluster = [{cl}, 1, 1]\n");
        }
        if pref > 0 {
            t += &format!("preferred_cluster = [{pref}, 1, 1]\n");
        }
        t += &format!(
            "[[buffers]]\nname = \"out\"\ndtype = \"u32\"\nshape = [{}]\ninit = \"zeros\"\n\
             [[params]]\nkind = \"ptr\"\nbuffer = \"out\"\n\
             [[variants]]\nid = \"good\"\nexpect = [\"clean\"]\n",
            grid as usize * W
        );
        let d = dir.join(&name);
        std::fs::create_dir_all(&d).unwrap();
        std::fs::write(d.join("case.toml"), t).unwrap();
        let case = Case::load(&d).unwrap();
        let k = Arc::new(ptxv_sass::load_cubin(&probe.join("patched.cubin"), "k").unwrap());
        let launch = build_launch(&case, &k).unwrap();
        let mut shapes = vec![launch.clone()];
        shapes.extend(launch.preferred_shape());
        assert_eq!(shapes.len(), 1 + (pref > 0) as usize, "{name}");
        let model: Vec<Vec<u32>> = shapes
            .into_iter()
            .map(|l| {
                let (m, fs, ok) = ptxv_sim::explore::run_once(Machine::new(k.clone(), Arc::new(l)));
                assert!(ok && fs.is_empty(), "{name}: {fs:#?}");
                words(&m.outputs())
            })
            .collect();
        let hw = words(&std::fs::read(&path).unwrap());
        let mut seen = [false; 2];
        for cta in 0..grid as usize {
            let r = &hw[W * cta..W * cta + W];
            let shape = match r[0] {
                0 => 0,
                0xffff_ff60 => 1, // 1 * -0xa0
                x => panic!("{name}: CTA {cta}: raw SR_CgaSize * -0xa0 = {x:#x}"),
            };
            seen[shape] = true;
            if pref > 0 {
                // The CTA's group (preferred size) ran wholly in one shape.
                let g0 = cta / pref as usize * pref as usize;
                assert_eq!(hw[W * g0], r[0], "{name}: CTA {cta}: group split");
            }
            let mine = &model[shape][W * cta..W * cta + W];
            for i in 0..W {
                assert_eq!(
                    mine[i], r[i],
                    "{name}: CTA {cta} (shape {shape}) word {i}: model {:?} GB300 {:?}",
                    mine, r
                );
            }
        }
        if pref == 0 {
            assert!(
                !seen[1],
                "{name}: SR_CgaSize 1 without a preferred dimension"
            );
        }
        mixed += (seen[0] && seen[1]) as usize;
        configs += 1;
    }
    assert_eq!(configs, 14);
    // The recorded launches with large shared memory mixed both shapes.
    assert!(mixed >= 5, "{mixed} mixed launches");
    let _ = std::fs::remove_dir_all(&dir);
}
