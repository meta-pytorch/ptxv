/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::File;
use crate::Reg;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TopologyError {
    TooManyThreads,
    TooManyWarps,
    InvalidWarp,
}

fn owner_for_warp(first: &mut Vec<Option<u32>>, warp: usize, thread: u32) -> u32 {
    match first[warp] {
        Some(owner) => owner,
        None => {
            first[warp] = Some(thread);
            thread
        }
    }
}

/// The first thread of each warp owns its uniform registers and predicates.
pub fn uniform_owners(warp_ids: &[u32], warp_count: usize) -> Result<Vec<u32>, TopologyError> {
    if warp_ids.len() > u32::MAX as usize {
        return Err(TopologyError::TooManyThreads);
    }
    if warp_count > warp_ids.len() {
        return Err(TopologyError::TooManyWarps);
    }
    let mut first = Vec::new();
    while first.len() < warp_count {
        first.push(None);
    }
    let mut owners = Vec::new();
    let mut thread = 0;
    while thread < warp_ids.len() {
        let warp = warp_ids[thread] as usize;
        if warp >= warp_count {
            return Err(TopologyError::InvalidWarp);
        }
        let owner = owner_for_warp(&mut first, warp, thread as u32);
        owners.push(owner);
        thread += 1;
    }
    Ok(owners)
}

/// Uniform files use their warp owner; lane files keep the supplied thread.
#[inline]
pub fn canonical_key(thread: usize, owners: &[u32], register: Reg) -> (u32, u8, u16) {
    let owner = match register.file {
        File::UR | File::UP if thread < owners.len() => owners[thread],
        _ => thread as u32,
    };
    (owner, register.file as u8, register.idx)
}

/// Whether a register value is protected from writes by another lane.
#[inline]
pub fn is_thread_private(register: Reg) -> bool {
    matches!(register.file, File::V | File::R | File::P)
}

/// Whether a per-thread register record refers to the same physical location.
#[inline]
pub fn record_aliases(
    thread: usize,
    owners: &[u32],
    register: Reg,
    record: (u32, u8, u16),
) -> bool {
    if register.file as u8 != record.1 || register.idx != record.2 {
        return false;
    }
    let current = canonical_key(thread, owners, register);
    let previous = canonical_key(record.0 as usize, owners, register);
    current.0 == previous.0
}

pub fn normalize_predicate(front: crate::Front, predicate: crate::Pred) -> crate::Pred {
    let constant = match predicate.reg {
        None => true,
        Some(register) => match register.file {
            crate::File::P => front == crate::Front::Sass && register.idx == 7,
            crate::File::UP => register.idx == 7,
            _ => false,
        },
    };
    if constant {
        crate::Pred {
            reg: None,
            neg: predicate.neg,
        }
    } else {
        predicate
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComparisonPlan {
    pub key: (u32, u8, u16),
    pub remember: bool,
}

pub fn comparison_plan(
    thread: usize,
    front: crate::Front,
    destination: Reg,
    plain: bool,
) -> ComparisonPlan {
    let normalized = normalize_predicate(
        front,
        crate::Pred {
            reg: Some(destination),
            neg: false,
        },
    );
    ComparisonPlan {
        key: (thread as u32, destination.file as u8, destination.idx),
        remember: plain
            && matches!(destination.file, crate::File::P | crate::File::UP)
            && normalized.reg.is_some(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn temporal_knowledge_excludes_warp_shared_locations() {
        let owners = [0, 0];
        for file in [File::V, File::R, File::UR, File::P, File::UP] {
            for idx in [0, 7, u16::MAX] {
                let register = Reg { file, idx };
                assert_eq!(
                    is_thread_private(register),
                    canonical_key(0, &owners, register) != canonical_key(1, &owners, register),
                );
            }
        }
    }

    #[test]
    fn uniform_files_alias_exactly_within_each_warp() {
        let warps = [2, 0, 2, 1, 0, 1];
        let owners = uniform_owners(&warps, 3).unwrap();
        assert_eq!(owners, [0, 1, 0, 3, 1, 3]);
        for left in 0..warps.len() {
            for right in 0..warps.len() {
                for file in [File::UR, File::UP] {
                    let register = Reg { file, idx: 7 };
                    assert_eq!(
                        record_aliases(
                            left,
                            &owners,
                            register,
                            (right as u32, file as u8, register.idx)
                        ),
                        warps[left] == warps[right]
                    );
                    assert_eq!(
                        canonical_key(left, &owners, register)
                            == canonical_key(right, &owners, register),
                        warps[left] == warps[right]
                    );
                }
                for file in [File::V, File::R, File::P] {
                    let register = Reg { file, idx: 7 };
                    assert_eq!(
                        canonical_key(left, &owners, register)
                            == canonical_key(right, &owners, register),
                        left == right
                    );
                }
            }
        }
    }

    #[test]
    fn register_namespaces_and_indices_stay_distinct() {
        let owners = uniform_owners(&[0, 0], 1).unwrap();
        let predicate = Reg {
            file: File::P,
            idx: 3,
        };
        let uniform_predicate = Reg {
            file: File::UP,
            idx: 3,
        };
        assert_ne!(
            canonical_key(0, &owners, predicate),
            canonical_key(0, &owners, uniform_predicate)
        );
        assert_ne!(
            canonical_key(0, &owners, uniform_predicate),
            canonical_key(
                0,
                &owners,
                Reg {
                    idx: 4,
                    ..uniform_predicate
                }
            )
        );
        assert_eq!(
            canonical_key(7, &[], uniform_predicate),
            (7, File::UP as u8, 3)
        );
        assert!(!record_aliases(
            0,
            &owners,
            uniform_predicate,
            (1, File::P as u8, 3)
        ));
        assert!(!record_aliases(
            0,
            &owners,
            uniform_predicate,
            (1, File::UP as u8, 4)
        ));
        assert!(record_aliases(
            0,
            &owners,
            uniform_predicate,
            (1, File::UP as u8, 3)
        ));
    }

    #[test]
    fn owner_indices_keep_all_32_bits() {
        let mut warps: Vec<u32> = (0..258).collect();
        warps.push(257);
        let owners = uniform_owners(&warps, 258).unwrap();
        assert_eq!(owners[257], 257);
        assert_eq!(owners[258], 257);
        for file in [File::UR, File::UP] {
            let register = Reg { file, idx: 3 };
            assert_eq!(
                canonical_key(257, &owners, register),
                canonical_key(258, &owners, register)
            );
            assert_ne!(
                canonical_key(257, &owners, register),
                canonical_key(1, &owners, register)
            );
            assert!(record_aliases(258, &owners, register, (257, file as u8, 3)));
            assert!(!record_aliases(258, &owners, register, (1, file as u8, 3)));
        }
    }

    #[test]
    fn malformed_topologies_fail_before_producing_an_owner_table() {
        assert_eq!(uniform_owners(&[], 1), Err(TopologyError::TooManyWarps));
        assert_eq!(uniform_owners(&[1], 1), Err(TopologyError::InvalidWarp));
        assert_eq!(uniform_owners(&[], 0), Ok(vec![]));
    }
}

#[cfg(test)]
mod constant_tests {
    use super::*;
    use crate::File;
    use crate::Front;
    use crate::Pred;
    use crate::Reg;

    #[test]
    fn normalize_preserves_architectural_constants_and_negation() {
        for front in [Front::Ptx, Front::Sass] {
            for file in [File::V, File::R, File::UR, File::P, File::UP] {
                for index in [0, 6, 7, 8, u16::MAX] {
                    for neg in [false, true] {
                        let predicate = Pred {
                            reg: Some(Reg { file, idx: index }),
                            neg,
                        };
                        let normalized = normalize_predicate(front, predicate);
                        let constant = index == 7
                            && (file == File::UP || file == File::P && front == Front::Sass);
                        assert_eq!(normalized.reg, if constant { None } else { predicate.reg });
                        assert_eq!(normalized.neg, neg);
                    }
                }
            }
            for neg in [false, true] {
                let predicate = Pred { reg: None, neg };
                assert_eq!(normalize_predicate(front, predicate), predicate);
            }
        }
    }
}

#[cfg(test)]
mod comparison_plan_tests {
    use super::*;
    use crate::File;
    use crate::Front;
    use crate::Reg;

    #[test]
    fn ignored_writes_never_remember_and_raw_trace_keys_remain_distinct() {
        for front in [Front::Ptx, Front::Sass] {
            for file in [File::V, File::R, File::UR, File::P, File::UP] {
                for index in [0, 6, 7, 8, u16::MAX] {
                    for thread in [0, 257, 258, u32::MAX as usize] {
                        let destination = Reg { file, idx: index };
                        let ignored = !matches!(file, File::P | File::UP)
                            || index == 7 && (file == File::UP || front == Front::Sass);
                        for plain in [false, true] {
                            let plan = comparison_plan(thread, front, destination, plain);
                            assert_eq!(plan.key, (thread as u32, file as u8, index));
                            assert_eq!(plan.remember, plain && !ignored);
                        }
                    }
                }
            }
        }
    }
}
