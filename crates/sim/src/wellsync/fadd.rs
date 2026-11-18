/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use std::collections::HashSet;

use ptxv_memory_write::atomic_confluence;
use ptxv_memory_write::atomic_exec::FloatingArithmetic;

use crate::atomic_exec::Arithmetic;

const MAX_STATES: usize = 4096;
const MAX_WORK: u64 = 8_000_000;

struct Candidate {
    alphabet: Vec<u32>,
    domain: Vec<u32>,
    layers: Vec<Vec<u32>>,
}

fn candidate_layers(values: &[u32], start: u32, max_work: u64) -> Option<Candidate> {
    if values.len() > 1024 {
        return None;
    }
    let mut alphabet = values.to_vec();
    alphabet.sort_unstable();
    alphabet.dedup();
    if alphabet.len() > 32 {
        return None;
    }
    if values.len() < 2 {
        return Some(Candidate {
            alphabet,
            domain: Vec::new(),
            layers: Vec::new(),
        });
    }
    let width = alphabet.len() as u64;
    let mut layers = vec![vec![start]];
    let mut domain = HashSet::from([start]);
    let mut layer_entries = 1u64;
    if width * width + width > max_work {
        return None;
    }
    while layers.len() + 1 < values.len() {
        let mut next = HashSet::new();
        for &state in layers.last().expect("initial layer") {
            for &value in &alphabet {
                next.insert(Arithmetic::add_f32_ftz(state, value));
                if next.len() > MAX_STATES {
                    return None;
                }
            }
        }
        domain.extend(next.iter().copied());
        if domain.len() > MAX_STATES {
            return None;
        }
        layer_entries += next.len() as u64;
        let work = domain.len() as u64 * width * width + layer_entries * width;
        if work > max_work {
            return None;
        }
        let mut next: Vec<_> = next.into_iter().collect();
        next.sort_unstable();
        layers.push(next);
    }
    let mut domain: Vec<_> = domain.into_iter().collect();
    domain.sort_unstable();
    Some(Candidate {
        alphabet,
        domain,
        layers,
    })
}

pub(super) fn order_independent(values: &[u32], start: u32, want: u32) -> bool {
    candidate_layers(values, start, MAX_WORK).is_some_and(|candidate| {
        atomic_confluence::check::<Arithmetic>(
            values,
            &candidate.alphabet,
            &candidate.domain,
            &candidate.layers,
            start,
            want,
            MAX_WORK,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn all_orders(values: &mut [u32], at: usize, start: u32, out: &mut HashSet<u32>) {
        if at == values.len() {
            out.insert(atomic_confluence::reference::<Arithmetic>(values, start));
            return;
        }
        for index in at..values.len() {
            values.swap(at, index);
            all_orders(values, at + 1, start, out);
            values.swap(at, index);
        }
    }

    #[test]
    fn exact_sums_certify_without_assuming_floating_associativity() {
        let values = [1.0f32, 0.5, -2.0, 3.5].map(f32::to_bits);
        let want = 3.0f32.to_bits();
        assert!(order_independent(&values, 0, want));
        let mut outcomes = HashSet::new();
        all_orders(&mut values.clone(), 0, 0, &mut outcomes);
        assert_eq!(outcomes, HashSet::from([want]));
    }

    #[test]
    fn rounding_sensitive_reduction_is_rejected() {
        let values = [1.0e20f32, -1.0e20, 1.0].map(f32::to_bits);
        let want = atomic_confluence::reference::<Arithmetic>(&values, 0);
        assert!(!order_independent(&values, 0, want));
        let mut outcomes = HashSet::new();
        all_orders(&mut values.clone(), 0, 0, &mut outcomes);
        assert!(outcomes.len() > 1);
    }

    #[test]
    fn special_values_are_compared_as_callback_bits() {
        for values in [
            [0x80000000, 0, 0x3f800000],
            [0x7fc01234, 0x3f800000, 0xffc05678],
            [0x7f800000, 0xff800000, 0x3f800000],
            [1, 0x80000001, 0],
        ] {
            let want = atomic_confluence::reference::<Arithmetic>(&values, 0);
            if order_independent(&values, 0, want) {
                let mut outcomes = HashSet::new();
                all_orders(&mut values.clone(), 0, 0, &mut outcomes);
                assert_eq!(outcomes, HashSet::from([want]));
            }
        }
    }

    #[test]
    fn capacity_and_work_exhaustion_never_certify() {
        let values = [1.0f32, 0.5, -2.0].map(f32::to_bits);
        let candidate = candidate_layers(&values, 0, MAX_WORK).unwrap();
        let want = atomic_confluence::reference::<Arithmetic>(&values, 0);
        assert!(!atomic_confluence::check::<Arithmetic>(
            &values,
            &candidate.alphabet,
            &candidate.domain,
            &candidate.layers,
            0,
            want,
            0
        ));
        assert!(candidate_layers(&values, 0, 0).is_none());
        assert!(!order_independent(&[0; 1025], 0, 0));
        assert!(!atomic_confluence::check::<Arithmetic>(
            &[0, 0],
            &[0],
            &[0],
            &[vec![0; 4097]],
            0,
            0,
            u64::MAX
        ));
    }

    #[test]
    fn forged_prefix_layers_and_wrong_reference_are_rejected() {
        let values = [1.0f32, 0.5, -2.0].map(f32::to_bits);
        let mut candidate = candidate_layers(&values, 0, MAX_WORK).unwrap();
        let want = atomic_confluence::reference::<Arithmetic>(&values, 0);
        candidate.layers[1].clear();
        assert!(!atomic_confluence::check::<Arithmetic>(
            &values,
            &candidate.alphabet,
            &candidate.domain,
            &candidate.layers,
            0,
            want,
            MAX_WORK
        ));
        let candidate = candidate_layers(&values, 0, MAX_WORK).unwrap();
        assert!(!atomic_confluence::check::<Arithmetic>(
            &values,
            &candidate.alphabet,
            &candidate.domain,
            &candidate.layers,
            0,
            want ^ 1,
            MAX_WORK
        ));
        assert!(!atomic_confluence::check::<Arithmetic>(
            &values,
            &candidate.alphabet,
            &candidate.domain,
            &candidate.layers,
            0x40000000,
            want,
            MAX_WORK
        ));
    }

    #[test]
    fn repeated_operands_scale_by_distinct_alphabet() {
        let values = [1.0f32.to_bits(); 128];
        let candidate = candidate_layers(&values, 0, MAX_WORK).unwrap();
        assert_eq!(candidate.alphabet.len(), 1);
        assert_eq!(candidate.layers.len(), 127);
        assert!(candidate.layers.iter().all(|layer| layer.len() == 1));
        assert!(order_independent(&values, 0, 128.0f32.to_bits()));
    }

    #[test]
    fn red_probe_exact_sums_certify_without_factorial_order_enumeration() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../probes/red_vec");
        let case = ptxv_case::Case::load(path).unwrap();
        let text = case.variant_ptx("good").unwrap();
        let kernel =
            std::sync::Arc::new(ptxv_ptx::parse_kernel(&text, Some(&case.m.entry)).unwrap());
        let launch = std::sync::Arc::new(crate::build_launch(&case, &kernel).unwrap());
        let result = crate::wellsync::certify(crate::machine::Machine::new(kernel, launch));
        eprintln!(
            "RED exact-sum certificate: steps={}, seconds={:.3}, reasons={:?}",
            result.steps, result.seconds, result.reasons
        );
        let verdict = result
            .verdict
            .expect("every scalar add order leaves the exact sum");
        assert_eq!(verdict.classes(), ["clean"]);
        assert_eq!(verdict.outputs.len(), 1);
        let report = case.expect_report(&verdict.outputs[0]).unwrap();
        assert!(!report.is_empty() && report.iter().all(|line| line.contains("bitwise equal")));
    }
}
