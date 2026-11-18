/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::BoolOp;
use crate::FCmp;
use crate::File;
use crate::Op;
use crate::Pred;
use crate::Reg;
use crate::Val;

#[derive(Clone, Copy, Debug)]
pub struct Plan {
    pub comparison: FCmp,
    pub negate: bool,
    pub swap: bool,
}

/// Recognizes a maximum only in a total ordered, non-NaN value interpretation.
pub fn real_max(plan: Plan) -> bool {
    match plan.comparison {
        FCmp::Gt | FCmp::Ge | FCmp::Gtu | FCmp::Geu => plan.negate == plan.swap,
        FCmp::Lt | FCmp::Le | FCmp::Ltu | FCmp::Leu => plan.negate != plan.swap,
        _ => false,
    }
}

pub trait Comparator {
    fn compare(&self, comparison: FCmp, x: u32, y: u32) -> bool;
}

pub fn from_operation(operation: &Op, negate: bool, swap: bool) -> Option<Plan> {
    match operation {
        Op::Fsetp {
            cmp,
            neg,
            abs,
            c,
            bop: BoolOp::And,
            ftz: false,
            ..
        } if !neg[0] && !neg[1] && !abs[0] && !abs[1] && c.reg.is_none() && !c.neg => Some(Plan {
            comparison: *cmp,
            negate,
            swap,
        }),
        _ => None,
    }
}

pub fn evaluate<C: Comparator>(plan: Plan, x: u32, y: u32, comparator: &C) -> u32 {
    if comparator.compare(plan.comparison, x, y) != plan.negate {
        if plan.swap { y } else { x }
    } else if plan.swap {
        x
    } else {
        y
    }
}

/// Registers in this pair belong to the slice, independently of kernel register numbers.
pub fn operations(plan: Plan) -> (Op, Op) {
    let x = Reg {
        file: File::R,
        idx: 0,
    };
    let y = Reg {
        file: File::R,
        idx: 1,
    };
    let predicate = Reg {
        file: File::P,
        idx: 0,
    };
    let destination = Reg {
        file: File::R,
        idx: 2,
    };
    let (first, second) = if plan.swap { (y, x) } else { (x, y) };
    (
        Op::Fsetp {
            d: predicate,
            cmp: plan.comparison,
            a: Val::Reg(x, 1),
            b: Val::Reg(y, 1),
            neg: [false; 2],
            abs: [false; 2],
            c: Pred {
                reg: None,
                neg: false,
            },
            bop: BoolOp::And,
            ftz: false,
        },
        Op::Sel {
            d: destination,
            a: Val::Reg(first, 1),
            b: Val::Reg(second, 1),
            p: Pred {
                reg: Some(predicate),
                neg: plan.negate,
            },
        },
    )
}

pub trait NodeReader {
    fn node_of(&self, value: &Val) -> Option<u32>;
}

fn same_register(left: Reg, right: Reg) -> bool {
    left.file as u8 == right.file as u8 && left.idx == right.idx
}

/// The compared nodes are captured when FSETP executes; SEL operands are observed now.
#[allow(
    clippy::question_mark,
    reason = "Aeneas cannot yet lower Option ControlFlow"
)]
pub fn from_pair<R: NodeReader>(
    front: crate::Front,
    comparison: &Op,
    selection: &Op,
    x: u32,
    y: u32,
    reader: &R,
) -> Option<Plan> {
    let Some(plan) = from_operation(comparison, false, false) else {
        return None;
    };
    let Op::Fsetp { d, .. } = comparison else {
        return None;
    };
    let Op::Sel { a, b, p, .. } = selection else {
        return None;
    };
    if !crate::register_owner::comparison_plan(0, front, *d, true).remember {
        return None;
    }
    let Some(predicate) = crate::register_owner::normalize_predicate(front, *p).reg else {
        return None;
    };
    if !same_register(*d, predicate) {
        return None;
    }
    let Some(first) = reader.node_of(a) else {
        return None;
    };
    let Some(second) = reader.node_of(b) else {
        return None;
    };
    let swap = if first == x && second == y {
        false
    } else if first == y && second == x {
        true
    } else {
        return None;
    };
    Some(Plan {
        comparison: plan.comparison,
        negate: p.neg,
        swap,
    })
}

#[derive(Clone, Copy, Debug)]
pub struct Evidence {
    pub destination: Reg,
    pub comparison: FCmp,
    pub left: u32,
    pub right: u32,
}

#[allow(
    clippy::question_mark,
    reason = "Aeneas cannot yet lower Option ControlFlow"
)]
pub fn capture<R: NodeReader>(front: crate::Front, operation: &Op, reader: &R) -> Option<Evidence> {
    let Some(plan) = from_operation(operation, false, false) else {
        return None;
    };
    let Op::Fsetp { d, a, b, .. } = operation else {
        return None;
    };
    if !crate::register_owner::comparison_plan(0, front, *d, true).remember {
        return None;
    }
    let Some(left) = reader.node_of(a) else {
        return None;
    };
    let Some(right) = reader.node_of(b) else {
        return None;
    };
    Some(Evidence {
        destination: *d,
        comparison: plan.comparison,
        left,
        right,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn comparison() -> Op {
        Op::Fsetp {
            d: Reg {
                file: File::P,
                idx: 3,
            },
            cmp: FCmp::Gtu,
            a: Val::Imm(17),
            b: Val::Imm(23),
            neg: [false; 2],
            abs: [false; 2],
            c: Pred::TRUE,
            bop: BoolOp::And,
            ftz: false,
        }
    }

    #[test]
    fn supported_comparison_preserves_actual_mode_and_selector_policy() {
        for negate in [false, true] {
            for swap in [false, true] {
                let plan = from_operation(&comparison(), negate, swap).unwrap();
                assert_eq!(plan.comparison, FCmp::Gtu);
                assert_eq!(plan.negate, negate);
                assert_eq!(plan.swap, swap);
                let (compare, select) = operations(plan);
                let Op::Fsetp {
                    d: predicate,
                    cmp,
                    neg,
                    abs,
                    c,
                    bop,
                    ftz,
                    ..
                } = compare
                else {
                    panic!()
                };
                assert_eq!(cmp, FCmp::Gtu);
                assert_eq!(
                    (neg, abs, c, bop, ftz),
                    ([false; 2], [false; 2], Pred::TRUE, BoolOp::And, false)
                );
                let Op::Sel {
                    a: Val::Reg(a, _),
                    b: Val::Reg(b, _),
                    p,
                    ..
                } = select
                else {
                    panic!()
                };
                assert_eq!((a.idx, b.idx), if swap { (1, 0) } else { (0, 1) });
                assert_eq!(
                    p,
                    Pred {
                        reg: Some(predicate),
                        neg: negate
                    }
                );
            }
        }
    }

    #[test]
    fn unsupported_comparison_modes_are_conservative() {
        for field in 0..8 {
            let mut operation = comparison();
            let Op::Fsetp {
                neg,
                abs,
                c,
                bop,
                ftz,
                ..
            } = &mut operation
            else {
                panic!()
            };
            match field {
                0 => neg[0] = true,
                1 => neg[1] = true,
                2 => abs[0] = true,
                3 => abs[1] = true,
                4 => {
                    c.reg = Some(Reg {
                        file: File::P,
                        idx: 1,
                    })
                }
                5 => c.neg = true,
                6 => *bop = BoolOp::Xor,
                _ => *ftz = true,
            }
            assert!(from_operation(&operation, false, false).is_none());
        }
        assert!(from_operation(&Op::Nop, false, false).is_none());
    }

    #[test]
    fn evaluation_preserves_callback_result_for_all_selector_policies() {
        struct Compare(bool);
        impl Comparator for Compare {
            fn compare(&self, comparison: FCmp, x: u32, y: u32) -> bool {
                assert_eq!(comparison, FCmp::Gtu);
                assert_eq!((x, y), (0, 0x8000_0000));
                self.0
            }
        }
        for answer in [false, true] {
            for negate in [false, true] {
                for swap in [false, true] {
                    let plan = from_operation(&comparison(), negate, swap).unwrap();
                    let first = if swap { 0x8000_0000 } else { 0 };
                    let second = if swap { 0 } else { 0x8000_0000 };
                    let expected = if answer != negate { first } else { second };
                    assert_eq!(evaluate(plan, 0, 0x8000_0000, &Compare(answer)), expected);
                }
            }
        }
    }
}

#[cfg(test)]
mod operand_tests {
    use std::cell::RefCell;

    use super::*;

    struct Sources {
        values: Vec<(Val, u32)>,
        reads: RefCell<Vec<Val>>,
    }
    impl NodeReader for Sources {
        fn node_of(&self, value: &Val) -> Option<u32> {
            self.reads.borrow_mut().push(value.clone());
            self.values
                .iter()
                .find(|(v, _)| v == value)
                .map(|(_, node)| *node)
        }
    }
    fn sources() -> Sources {
        Sources {
            values: vec![(Val::Imm(17), 7), (Val::Imm(23), 11)],
            reads: RefCell::new(vec![]),
        }
    }
    fn pair(destination: Reg) -> (Op, Op) {
        (
            Op::Fsetp {
                d: destination,
                cmp: FCmp::Gtu,
                a: Val::Imm(17),
                b: Val::Imm(23),
                neg: [false; 2],
                abs: [false; 2],
                c: Pred::TRUE,
                bop: BoolOp::And,
                ftz: false,
            },
            Op::Sel {
                d: Reg {
                    file: File::R,
                    idx: 5,
                },
                a: Val::Imm(23),
                b: Val::Imm(17),
                p: Pred {
                    reg: Some(destination),
                    neg: true,
                },
            },
        )
    }
    #[test]
    fn actual_operands_determine_capture_mode_negation_and_order() {
        let (compare, select) = pair(Reg {
            file: File::P,
            idx: 3,
        });
        let reader = sources();
        let captured = capture(crate::Front::Sass, &compare, &reader).unwrap();
        assert_eq!(
            (
                captured.destination,
                captured.comparison,
                captured.left,
                captured.right
            ),
            (
                Reg {
                    file: File::P,
                    idx: 3
                },
                FCmp::Gtu,
                7,
                11
            )
        );
        assert_eq!(*reader.reads.borrow(), [Val::Imm(17), Val::Imm(23)]);
        reader.reads.borrow_mut().clear();
        let plan = from_pair(
            crate::Front::Sass,
            &compare,
            &select,
            captured.left,
            captured.right,
            &reader,
        )
        .unwrap();
        assert_eq!(
            (plan.comparison, plan.negate, plan.swap),
            (FCmp::Gtu, true, true)
        );
        assert_eq!(*reader.reads.borrow(), [Val::Imm(23), Val::Imm(17)]);
        let mut forward = select.clone();
        let Op::Sel { a, b, .. } = &mut forward else {
            panic!()
        };
        std::mem::swap(a, b);
        assert!(
            !from_pair(crate::Front::Sass, &compare, &forward, 7, 11, &reader)
                .unwrap()
                .swap
        );
        assert!(from_pair(crate::Front::Sass, &compare, &select, 71, 113, &reader).is_none());
    }
    #[test]
    fn ignored_predicates_and_distinct_namespaces_never_create_a_pair() {
        for (front, destination, supported) in [
            (
                crate::Front::Sass,
                Reg {
                    file: File::P,
                    idx: 7,
                },
                false,
            ),
            (
                crate::Front::Ptx,
                Reg {
                    file: File::P,
                    idx: 7,
                },
                true,
            ),
            (
                crate::Front::Sass,
                Reg {
                    file: File::UP,
                    idx: 7,
                },
                false,
            ),
            (
                crate::Front::Ptx,
                Reg {
                    file: File::UP,
                    idx: 7,
                },
                false,
            ),
            (
                crate::Front::Ptx,
                Reg {
                    file: File::V,
                    idx: 0,
                },
                false,
            ),
        ] {
            let (compare, select) = pair(destination);
            let reader = sources();
            assert_eq!(capture(front, &compare, &reader).is_some(), supported);
            reader.reads.borrow_mut().clear();
            assert_eq!(
                from_pair(front, &compare, &select, 7, 11, &reader).is_some(),
                supported
            );
            if !supported {
                assert!(reader.reads.borrow().is_empty());
            }
        }
        let (compare, mut select) = pair(Reg {
            file: File::P,
            idx: 3,
        });
        let Op::Sel { p, .. } = &mut select else {
            panic!()
        };
        p.reg.as_mut().unwrap().file = File::UP;
        let reader = sources();
        assert!(from_pair(crate::Front::Sass, &compare, &select, 7, 11, &reader).is_none());
        assert!(reader.reads.borrow().is_empty());
    }
    #[test]
    fn missing_observations_preserve_the_query_prefix() {
        let (compare, select) = pair(Reg {
            file: File::P,
            idx: 3,
        });
        let mut reader = sources();
        reader.values.clear();
        assert!(capture(crate::Front::Sass, &compare, &reader).is_none());
        assert_eq!(*reader.reads.borrow(), [Val::Imm(17)]);
        reader.reads.borrow_mut().clear();
        assert!(from_pair(crate::Front::Sass, &compare, &select, 7, 11, &reader).is_none());
        assert_eq!(*reader.reads.borrow(), [Val::Imm(23)]);
        reader.reads.borrow_mut().clear();
        assert!(from_pair(crate::Front::Sass, &compare, &Op::Nop, 7, 11, &reader).is_none());
        assert!(reader.reads.borrow().is_empty());
    }
}

#[cfg(test)]
mod real_max_tests {
    use super::*;

    struct IntegerOrder;
    impl Comparator for IntegerOrder {
        fn compare(&self, comparison: FCmp, x: u32, y: u32) -> bool {
            let (x, y) = (x as i32, y as i32);
            match comparison {
                FCmp::Gt | FCmp::Gtu => x > y,
                FCmp::Ge | FCmp::Geu => x >= y,
                FCmp::Lt | FCmp::Ltu => x < y,
                FCmp::Le | FCmp::Leu => x <= y,
                _ => panic!("unsupported ordered comparison"),
            }
        }
    }

    #[test]
    fn accepted_plans_select_the_ordered_maximum_including_ties() {
        for comparison in [
            FCmp::Gt,
            FCmp::Ge,
            FCmp::Gtu,
            FCmp::Geu,
            FCmp::Lt,
            FCmp::Le,
            FCmp::Ltu,
            FCmp::Leu,
        ] {
            for negate in [false, true] {
                for swap in [false, true] {
                    let plan = Plan {
                        comparison,
                        negate,
                        swap,
                    };
                    for x in [-17i32, 0, 23] {
                        for y in [-17i32, 0, 23] {
                            let result = evaluate(plan, x as u32, y as u32, &IntegerOrder) as i32;
                            if real_max(plan) {
                                assert_eq!(result, x.max(y));
                            } else {
                                assert_eq!(result, x.min(y));
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn equality_and_nan_classifications_never_claim_a_real_maximum() {
        for comparison in [
            FCmp::F,
            FCmp::Eq,
            FCmp::Ne,
            FCmp::Num,
            FCmp::Nan,
            FCmp::Equ,
            FCmp::Neu,
            FCmp::T,
        ] {
            for negate in [false, true] {
                for swap in [false, true] {
                    assert!(!real_max(Plan {
                        comparison,
                        negate,
                        swap
                    }));
                }
            }
        }
    }
}
