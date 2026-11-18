/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::AtomOp;
use crate::Front;
use crate::Pred;
use crate::Reg;

pub fn observed(front: Front, word: Option<Reg>, predicate: Option<Reg>) -> bool {
    word.is_some()
        || crate::register_owner::normalize_predicate(
            front,
            Pred {
                reg: predicate,
                neg: false,
            },
        )
        .reg
        .is_some()
}

pub fn local_order_monitored(
    front: Front,
    operation: AtomOp,
    word: Option<Reg>,
    predicate: Option<Reg>,
) -> bool {
    matches!(operation, AtomOp::Cas | AtomOp::CasSpin | AtomOp::Exch)
        || observed(front, word, predicate)
}
