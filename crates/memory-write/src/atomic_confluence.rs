/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

//! A bounded certificate for order independence of one unobserved fp32-add word.

use crate::atomic_exec::FloatingArithmetic;

pub fn contains(values: &[u32], value: u32) -> bool {
    let mut index = 0;
    while index < values.len() {
        if values[index] == value {
            return true;
        }
        index += 1;
    }
    false
}

pub fn subset(values: &[u32], allowed: &[u32]) -> bool {
    let mut index = 0;
    while index < values.len() {
        if !contains(allowed, values[index]) {
            return false;
        }
        index += 1;
    }
    true
}

pub fn commutes_row<F: FloatingArithmetic>(values: &[u32], state: u32, first: u32) -> bool {
    let left = F::add_f32_ftz(state, first);
    let mut index = 0;
    while index < values.len() {
        let second = values[index];
        let right = F::add_f32_ftz(state, second);
        if F::add_f32_ftz(left, second) != F::add_f32_ftz(right, first) {
            return false;
        }
        index += 1;
    }
    true
}

pub fn commutes_state<F: FloatingArithmetic>(values: &[u32], state: u32) -> bool {
    let mut index = 0;
    while index < values.len() {
        if !commutes_row::<F>(values, state, values[index]) {
            return false;
        }
        index += 1;
    }
    true
}

pub fn commutes_layer<F: FloatingArithmetic>(values: &[u32], states: &[u32]) -> bool {
    let mut index = 0;
    while index < states.len() {
        if !commutes_state::<F>(values, states[index]) {
            return false;
        }
        index += 1;
    }
    true
}

pub fn extends_row<F: FloatingArithmetic>(values: &[u32], next: &[u32], state: u32) -> bool {
    let mut index = 0;
    while index < values.len() {
        if !contains(next, F::add_f32_ftz(state, values[index])) {
            return false;
        }
        index += 1;
    }
    true
}

pub fn extends_layer<F: FloatingArithmetic>(values: &[u32], current: &[u32], next: &[u32]) -> bool {
    let mut index = 0;
    while index < current.len() {
        if !extends_row::<F>(values, next, current[index]) {
            return false;
        }
        index += 1;
    }
    true
}

pub fn reference<F: FloatingArithmetic>(values: &[u32], start: u32) -> u32 {
    let mut state = start;
    let mut index = 0;
    while index < values.len() {
        state = F::add_f32_ftz(state, values[index]);
        index += 1;
    }
    state
}

/// Layers may overapproximate prefixes, including repetitions of the supplied operands.
/// A successful check permits every permutation; capacity rejection never certifies.
pub fn check<F: FloatingArithmetic>(
    values: &[u32],
    alphabet: &[u32],
    domain: &[u32],
    layers: &[Vec<u32>],
    start: u32,
    want: u32,
    max_work: u64,
) -> bool {
    if values.len() > 1024 || alphabet.len() > 32 || domain.len() > 4096 {
        return false;
    }
    if values.len() < 2 {
        return layers.is_empty() && reference::<F>(values, start) == want;
    }
    if layers.len() != values.len() - 1 || !contains(&layers[0], start) || !subset(values, alphabet)
    {
        return false;
    }
    let work = domain.len() as u64 * alphabet.len() as u64 * alphabet.len() as u64;
    if work > max_work || !commutes_layer::<F>(alphabet, domain) {
        return false;
    }
    check_layers::<F>(
        values, alphabet, domain, layers, start, want, max_work, work,
    )
}

#[expect(
    clippy::too_many_arguments,
    reason = "Keep the source-extracted certificate helper parameters explicit"
)]
pub fn check_layers<F: FloatingArithmetic>(
    values: &[u32],
    alphabet: &[u32],
    domain: &[u32],
    layers: &[Vec<u32>],
    start: u32,
    want: u32,
    max_work: u64,
    mut work: u64,
) -> bool {
    let mut depth = 0;
    while depth < layers.len() {
        let states = &layers[depth];
        if states.len() > 4096 {
            return false;
        }
        work += states.len() as u64 * alphabet.len() as u64;
        if work > max_work {
            return false;
        }
        if !subset(states, domain) {
            return false;
        }
        if depth + 1 < layers.len() && !extends_layer::<F>(alphabet, states, &layers[depth + 1]) {
            return false;
        }
        depth += 1;
    }
    reference::<F>(values, start) == want
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Wrapping;
    impl FloatingArithmetic for Wrapping {
        fn add_f32_ftz(old: u32, value: u32) -> u32 {
            old.wrapping_add(value)
        }
        fn add_f64(old: u64, value: u64) -> u64 {
            old.wrapping_add(value)
        }
        fn add_bf16(old: u16, value: u16) -> u16 {
            old.wrapping_add(value)
        }
    }

    struct LateFailure;
    impl FloatingArithmetic for LateFailure {
        fn add_f32_ftz(old: u32, value: u32) -> u32 {
            if old == 3 && value == 3 {
                100
            } else if old == 5 && value == 2 {
                200
            } else {
                old | (1 << (value - 1))
            }
        }
        fn add_f64(_: u64, _: u64) -> u64 {
            panic!("not called")
        }
        fn add_bf16(_: u16, _: u16) -> u16 {
            panic!("not called")
        }
    }

    #[test]
    fn certified_layers_cover_duplicate_operands_and_wrapping() {
        assert!(check::<Wrapping>(
            &[1, 2, 3],
            &[1, 2, 3],
            &[0, 1, 2, 3],
            &[vec![0], vec![1, 2, 3]],
            0,
            6,
            100
        ));
        assert!(check::<Wrapping>(
            &[1, u32::MAX],
            &[1, u32::MAX],
            &[1],
            &[vec![1]],
            1,
            1,
            100
        ));
        assert!(check::<Wrapping>(&[], &[], &[], &[], 7, 7, 0));
        assert!(check::<Wrapping>(&[2], &[2], &[], &[], 7, 9, 0));
        assert!(check::<Wrapping>(
            &[1; 128],
            &[1],
            &(0..127).collect::<Vec<_>>(),
            &(0..127).map(|n| vec![n]).collect::<Vec<_>>(),
            0,
            128,
            1000
        ));
    }

    #[test]
    fn every_relevant_prefix_domain_requires_commutation() {
        assert!(commutes_layer::<LateFailure>(&[1, 2, 3], &[0]));
        let want = reference::<LateFailure>(&[1, 2, 3], 0);
        assert!(!check::<LateFailure>(
            &[1, 2, 3],
            &[1, 2, 3],
            &[0, 1, 2, 4],
            &[vec![0], vec![1, 2, 4]],
            0,
            want,
            100
        ));
    }

    #[test]
    fn seed_closure_operand_and_domain_membership_are_required() {
        assert!(!check::<Wrapping>(
            &[1, 2, 3],
            &[1, 2, 3],
            &[0],
            &[vec![0], vec![]],
            0,
            6,
            100
        ));
        assert!(!check::<Wrapping>(
            &[0, 0],
            &[0],
            &[0],
            &[vec![0]],
            9,
            9,
            100
        ));
        assert!(!check::<Wrapping>(
            &[1, 2, 3],
            &[1, 2],
            &[0, 1, 2, 3],
            &[vec![0], vec![1, 2, 3]],
            0,
            6,
            100
        ));
        assert!(!check::<Wrapping>(
            &[1, 2, 3],
            &[1, 2, 3],
            &[0, 1, 2],
            &[vec![0], vec![1, 2, 3]],
            0,
            6,
            100
        ));
        assert!(!check::<Wrapping>(
            &[1, 2, 3],
            &[1, 2, 3],
            &[0, 1, 2, 3],
            &[vec![0], vec![1, 2, 3]],
            0,
            7,
            100
        ));
    }

    #[test]
    fn bounds_and_work_fail_closed() {
        assert!(!check::<Wrapping>(
            &[0; 1025],
            &[],
            &[],
            &[],
            0,
            0,
            u64::MAX
        ));
        assert!(!check::<Wrapping>(
            &[0, 0],
            &[0; 33],
            &[0],
            &[vec![0]],
            0,
            0,
            u64::MAX
        ));
        assert!(!check::<Wrapping>(
            &[0, 0],
            &[0],
            &[0; 4097],
            &[vec![0]],
            0,
            0,
            u64::MAX
        ));
        assert!(!check::<Wrapping>(
            &[0, 0],
            &[0],
            &[0],
            &[vec![0; 4097]],
            0,
            0,
            u64::MAX
        ));
        assert!(!check::<Wrapping>(
            &[1, 2],
            &[1, 2],
            &[0],
            &[vec![0]],
            0,
            3,
            3
        ));
        assert!(!check::<Wrapping>(
            &[1, 2, 3],
            &[1, 2, 3],
            &[0],
            &[vec![0]],
            0,
            6,
            100
        ));
    }
}
