/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::atomic_components::Plan;
use crate::atomic_exec::Operation;
use crate::atomic_integer::IntegerOp;

pub fn plan(
    ptx: bool,
    operation: Operation,
    bytes: u8,
    result_used: bool,
    predicate: bool,
    explicit_new: bool,
) -> Option<Plan> {
    if ptx
        && result_used
        && !predicate
        && !explicit_new
        && bytes == 4
        && supported_operation(operation)
    {
        Some(Plan {
            first: true,
            second: false,
            third: false,
            fourth: false,
        })
    } else {
        None
    }
}

pub fn supported_operation(operation: Operation) -> bool {
    matches!(
        operation,
        Operation::AddF32Ftz
            | Operation::Integer(
                IntegerOp::Or
                    | IntegerOp::And
                    | IntegerOp::Xor
                    | IntegerOp::Add
                    | IntegerOp::Exch
                    | IntegerOp::MaxU
                    | IntegerOp::Inc
                    | IntegerOp::Dec
            )
    )
}

pub fn request(
    operation: Operation,
    source: crate::symbolic::Metadata,
) -> crate::atomic_exec::AtomicRequest {
    crate::atomic_exec::AtomicRequest {
        op: operation,
        bytes: 4,
        value: source.captured as u64,
        compare: 0,
    }
}

pub fn replacement_defined(operation: Operation, old_defined: bool, source_defined: bool) -> bool {
    match operation {
        Operation::Integer(IntegerOp::Exch) => source_defined,
        _ => old_defined && source_defined,
    }
}

pub fn bundled_order_unmodeled(
    ptx: bool,
    result_used: bool,
    continued: bool,
    same_warp: bool,
    same_word: bool,
    shared_result: bool,
) -> bool {
    ptx && result_used && !continued && same_warp && (same_word || shared_result)
}

pub fn claim_allowed(actor: u32, forced: Option<u32>) -> bool {
    match forced {
        None => true,
        Some(forced) => actor == forced,
    }
}

pub fn ticket_progress(active: bool, ready: bool) -> bool {
    !active || ready
}

pub fn ticket_pair_eligible(
    actor: u32,
    warp: u32,
    pc: u32,
    other_actor: u32,
    other_warp: u32,
    other_pc: u32,
    continued: bool,
) -> bool {
    actor == other_actor || warp != other_warp || pc != other_pc || continued
}

/// (warp, region, owner, offset); region zero denotes an unresolved word.
pub type LaneClaim = (u32, u8, usize, u64);

pub fn claim_pair_unmodeled(
    ptx: bool,
    result_used: bool,
    continued: bool,
    shared_result: bool,
    left: LaneClaim,
    right: LaneClaim,
) -> bool {
    let same_word = left.1 != 0 && left.1 == right.1 && left.2 == right.2 && left.3 == right.3;
    bundled_order_unmodeled(
        ptx,
        result_used,
        continued,
        left.0 == right.0,
        same_word,
        shared_result,
    )
}

pub fn unmodeled_with(
    ptx: bool,
    result_used: bool,
    continued: bool,
    shared_result: bool,
    left: LaneClaim,
    claims: &[LaneClaim],
    mut index: usize,
) -> bool {
    while index < claims.len() {
        if claim_pair_unmodeled(
            ptx,
            result_used,
            continued,
            shared_result,
            left,
            claims[index],
        ) {
            return true;
        }
        index += 1;
    }
    false
}

pub fn unmodeled_claims(
    ptx: bool,
    result_used: bool,
    continued: bool,
    shared_result: bool,
    claims: &[LaneClaim],
) -> bool {
    let mut index = 0;
    while index < claims.len() {
        if unmodeled_with(
            ptx,
            result_used,
            continued,
            shared_result,
            claims[index],
            claims,
            index + 1,
        ) {
            return true;
        }
        index += 1;
    }
    false
}

pub fn returned_word_claims_unmodeled(
    continued: bool,
    shared_result: bool,
    claims: &[LaneClaim],
) -> bool {
    unmodeled_claims(true, true, continued, shared_result, claims)
}

pub fn local_claims_unmodeled(
    monitored: bool,
    scalar_red: bool,
    certificate_collection: bool,
    continued: bool,
    shared_result: bool,
    claims: &[LaneClaim],
) -> bool {
    let conflict = returned_word_claims_unmodeled(continued, monitored && shared_result, claims);
    (monitored || scalar_red && !certificate_collection) && conflict
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn local_admission_defers_only_returnless_float_collection() {
        let colliding = [(0, 1, 0, 0), (0, 1, 0, 0), (0, 1, 0, 0)];
        let private = [(0, 1, 0, 0), (0, 1, 0, 4)];
        assert!(local_claims_unmodeled(
            true, false, true, false, false, &colliding
        ));
        assert!(local_claims_unmodeled(
            false, true, false, false, false, &colliding
        ));
        assert!(!local_claims_unmodeled(
            false, true, true, false, false, &colliding
        ));
        assert!(!local_claims_unmodeled(
            false, false, false, false, false, &colliding
        ));
        assert!(!local_claims_unmodeled(
            true, true, false, true, true, &colliding
        ));
        assert!(!local_claims_unmodeled(
            true, false, false, false, false, &private
        ));
        assert!(local_claims_unmodeled(
            true, false, false, false, true, &private
        ));
        assert!(!local_claims_unmodeled(
            false, true, false, false, true, &private
        ));
    }

    #[test]
    fn complete_claim_scan_matches_all_resolved_local_pairs() {
        let pool = [
            (0, 0, 0, 0),
            (0, 1, 0, 0),
            (0, 1, 0, 4),
            (1, 1, 0, 0),
            (0, 2, 0, 0),
            (0, 1, 1, 0),
        ];
        for length in 0..=4 {
            for encoded in 0..pool.len().pow(length as u32) {
                let mut encoding = encoded;
                let claims: Vec<_> = (0..length)
                    .map(|_| {
                        let claim = pool[encoding % pool.len()];
                        encoding /= pool.len();
                        claim
                    })
                    .collect();
                for ptx in [false, true] {
                    for used in [false, true] {
                        for continued in [false, true] {
                            for shared in [false, true] {
                                let expected = ptx
                                    && used
                                    && !continued
                                    && claims.iter().enumerate().any(|(index, left)| {
                                        claims[index + 1..].iter().any(|right| {
                                            left.0 == right.0
                                                && (shared
                                                    || (left.1 != 0
                                                        && left.1 == right.1
                                                        && left.2 == right.2
                                                        && left.3 == right.3))
                                        })
                                    });
                                assert_eq!(
                                    unmodeled_claims(ptx, used, continued, shared, &claims),
                                    expected
                                );
                                if ptx && used {
                                    assert_eq!(
                                        returned_word_claims_unmodeled(continued, shared, &claims),
                                        expected
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn only_supported_returning_ptx_word_atomics_select_a_lane_plan() {
        let ops = [
            IntegerOp::Or,
            IntegerOp::And,
            IntegerOp::Xor,
            IntegerOp::Add,
            IntegerOp::Exch,
            IntegerOp::MinU,
            IntegerOp::MaxU,
            IntegerOp::MinS,
            IntegerOp::MaxS,
            IntegerOp::Inc,
            IntegerOp::Dec,
            IntegerOp::Cas,
            IntegerOp::CasSpin,
        ];
        for bytes in 0..=u8::MAX {
            for ptx in [false, true] {
                for used in [false, true] {
                    for predicate in [false, true] {
                        for explicit_new in [false, true] {
                            for operation in ops {
                                let selected = plan(
                                    ptx,
                                    Operation::Integer(operation),
                                    bytes,
                                    used,
                                    predicate,
                                    explicit_new,
                                );
                                assert_eq!(
                                    selected.is_some(),
                                    ptx && used
                                        && !predicate
                                        && !explicit_new
                                        && bytes == 4
                                        && [
                                            IntegerOp::Or,
                                            IntegerOp::And,
                                            IntegerOp::Xor,
                                            IntegerOp::Add,
                                            IntegerOp::Exch,
                                            IntegerOp::MaxU,
                                            IntegerOp::Inc,
                                            IntegerOp::Dec
                                        ]
                                        .contains(&operation)
                                );
                                if let Some(selected) = selected {
                                    assert_eq!(
                                        crate::atomic_components::choices(selected),
                                        vec![crate::atomic_components::Component::First]
                                    );
                                }
                            }
                        }
                    }
                }
            }
            for operation in [
                Operation::AddF32Ftz,
                Operation::AddF64,
                Operation::AddBf16,
                Operation::AddBf16x2,
            ] {
                assert_eq!(
                    plan(true, operation, bytes, true, false, false).is_some(),
                    operation == Operation::AddF32Ftz && bytes == 4
                );
            }
        }
    }

    #[test]
    fn frozen_request_preserves_operation_and_exchange_ignores_only_the_old_definedness() {
        for operation in [
            Operation::Integer(IntegerOp::Add),
            Operation::Integer(IntegerOp::Exch),
            Operation::Integer(IntegerOp::Cas),
            Operation::AddF32Ftz,
            Operation::AddBf16,
        ] {
            for value in [0, 1, u32::MAX] {
                let source = crate::symbolic::prepare(value, Some(17), true, false);
                let request = request(operation, source);
                assert_eq!(request.op, operation);
                assert_eq!(request.bytes, 4);
                assert_eq!(request.value, value as u64);
                assert_eq!(request.compare, 0);
            }
            for old in [false, true] {
                for source in [false, true] {
                    assert_eq!(
                        replacement_defined(operation, old, source),
                        source && (operation == Operation::Integer(IntegerOp::Exch) || old)
                    );
                }
            }
        }
    }

    #[test]
    fn returning_uncontinued_interfering_pairs_are_unmodeled() {
        for ptx in [false, true] {
            for used in [false, true] {
                for continued in [false, true] {
                    for same_warp in [false, true] {
                        for same_word in [false, true] {
                            for shared_result in [false, true] {
                                let interfering = same_word || shared_result;
                                let gap = bundled_order_unmodeled(
                                    ptx,
                                    used,
                                    continued,
                                    same_warp,
                                    same_word,
                                    shared_result,
                                );
                                assert_eq!(
                                    gap,
                                    ptx && used && !continued && same_warp && interfering
                                );
                                if ptx && used && same_warp && interfering && !gap {
                                    assert!(continued);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn forced_claims_and_empty_guards_keep_their_enabledness() {
        for actor in [0, 1, 2, u32::MAX] {
            assert!(claim_allowed(actor, None));
            for forced in [0, 1, 2, u32::MAX] {
                assert_eq!(claim_allowed(actor, Some(forced)), actor == forced);
            }
        }
        for active in [false, true] {
            for ready in [false, true] {
                assert_eq!(ticket_progress(active, ready), !active || ready);
            }
        }
        assert!(!ticket_pair_eligible(0, 1, 7, 1, 1, 7, false));
        assert!(ticket_pair_eligible(0, 1, 7, 1, 1, 7, true));
        assert!(ticket_pair_eligible(0, 1, 7, 1, 2, 7, false));
        assert!(ticket_pair_eligible(0, 1, 7, 1, 1, 8, false));
        assert!(ticket_pair_eligible(0, 1, 7, 0, 1, 7, false));
    }
}
