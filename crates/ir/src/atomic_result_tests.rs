/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::AtomOp;
use crate::File;
use crate::Front;
use crate::Reg;
use crate::atomic_result::local_order_monitored;
use crate::atomic_result::observed;

#[test]
fn predicate_only_outputs_are_observed() {
    for front in [Front::Ptx, Front::Sass] {
        for file in [File::P, File::UP] {
            let predicate = Some(Reg { file, idx: 0 });
            assert!(observed(front, None, predicate));
            assert!(local_order_monitored(
                front,
                AtomOp::FAddFtz,
                None,
                predicate
            ));
        }
    }
}

#[test]
fn constant_predicate_destinations_do_not_count_as_outputs() {
    assert!(!observed(
        Front::Sass,
        None,
        Some(Reg {
            file: File::P,
            idx: 7
        })
    ));
    assert!(observed(
        Front::Ptx,
        None,
        Some(Reg {
            file: File::P,
            idx: 7
        })
    ));
    for front in [Front::Ptx, Front::Sass] {
        assert!(!observed(front, None, None));
        assert!(!observed(
            front,
            None,
            Some(Reg {
                file: File::UP,
                idx: 7
            })
        ));
    }
}

#[test]
fn discarded_compare_exchange_results_still_require_monitoring() {
    for front in [Front::Ptx, Front::Sass] {
        assert!(local_order_monitored(front, AtomOp::Cas, None, None));
        assert!(local_order_monitored(front, AtomOp::CasSpin, None, None));
    }
}

#[test]
fn word_outputs_are_observed_with_an_ignored_predicate() {
    for front in [Front::Ptx, Front::Sass] {
        let word = Some(Reg {
            file: File::R,
            idx: 2,
        });
        let predicate = Some(Reg {
            file: File::UP,
            idx: 7,
        });
        assert!(observed(front, word, predicate));
        assert!(local_order_monitored(front, AtomOp::Add, word, predicate));
    }
}

#[test]
fn discarded_exchange_results_still_require_monitoring() {
    for front in [Front::Ptx, Front::Sass] {
        assert!(local_order_monitored(front, AtomOp::Exch, None, None));
    }
}
