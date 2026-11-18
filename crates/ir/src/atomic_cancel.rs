/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::Addr;
use crate::AtomOp;
use crate::File;
use crate::Front;
use crate::Inst;
use crate::IntOp;
use crate::Op;
use crate::Reg;
use crate::Sem;
use crate::Val;

pub fn same_register(left: Reg, right: Reg) -> bool {
    left.file as u8 == right.file as u8 && left.idx == right.idx
}

fn private_value(value: &Val) -> bool {
    match value {
        Val::Reg(register, _) => register.file == File::V,
        _ => true,
    }
}

fn scalar_payload(value: &Val) -> bool {
    match value {
        Val::Reg(register, slots) => register.file == File::V && *slots == 1,
        _ => true,
    }
}

fn reads(value: &Val, register: Reg) -> bool {
    match value {
        Val::Reg(source, _) => same_register(*source, register),
        _ => false,
    }
}

fn optional_private(value: &Option<Val>) -> bool {
    match value {
        Some(value) => private_value(value),
        None => true,
    }
}

fn optional_reads(value: &Option<Val>, register: Reg) -> bool {
    match value {
        Some(value) => reads(value, register),
        None => false,
    }
}

fn address_private(address: &Addr) -> bool {
    optional_private(&address.base) && optional_private(&address.base2)
}

fn address_reads(address: &Addr, register: Reg) -> bool {
    optional_reads(&address.base, register) || optional_reads(&address.base2, register)
}

pub fn plain_atom(operation: &Op) -> bool {
    matches!(operation, Op::Atom { op: AtomOp::Add, bytes: 4, sync, p: None, new: None, .. }
        if sync.sem == Sem::Relaxed)
}

pub fn linear_private(operation: &Op) -> bool {
    match operation {
        Op::Mov { d, a } | Op::CvtInt { d, a, .. } => d.file == File::V && private_value(a),
        Op::Int { d, a, b, .. } | Op::MulWide { d, a, b, .. } => {
            d.file == File::V && private_value(a) && private_value(b)
        }
        Op::Ld { d, addr, .. } => d.file == File::V && address_private(addr),
        Op::St { addr, v, .. } => address_private(addr) && scalar_payload(v),
        Op::Atom { addr, v, d, .. } => {
            plain_atom(operation)
                && address_private(addr)
                && scalar_payload(v)
                && match d {
                    Some(destination) => destination.file == File::V,
                    None => true,
                }
        }
        Op::Exit => true,
        _ => false,
    }
}

pub fn reads_return(operation: &Op, register: Reg) -> bool {
    match operation {
        Op::Mov { a, .. } | Op::CvtInt { a, .. } => reads(a, register),
        Op::Int { a, b, .. } | Op::MulWide { a, b, .. } => reads(a, register) || reads(b, register),
        Op::Ld { addr, .. } => address_reads(addr, register),
        Op::St { addr, v, .. } | Op::Atom { addr, v, .. } => {
            address_reads(addr, register) || reads(v, register)
        }
        Op::Exit => false,
        _ => true,
    }
}

pub fn overwrites_return(operation: &Op, register: Reg) -> bool {
    match operation {
        Op::Mov { d, .. } | Op::Int { d, .. } | Op::CvtInt { d, .. } | Op::MulWide { d, .. } => {
            same_register(*d, register)
        }
        Op::Ld { d, n, bytes, .. } => *n > 0 && *bytes > 0 && same_register(*d, register),
        Op::Atom { d: Some(d), .. } => same_register(*d, register),
        _ => false,
    }
}

pub fn cancels(operation: &Op, register: Reg) -> bool {
    match operation {
        Op::Int {
            op: IntOp::Sub | IntOp::Xor,
            bits: 32,
            d,
            a: Val::Reg(left, 1),
            b: Val::Reg(right, 1),
        } => d.file == File::V && same_register(*left, register) && same_register(*right, register),
        _ => false,
    }
}

pub fn cancelled_return(front: Front, code: &[Inst], pc: usize) -> bool {
    if front != Front::Ptx || pc >= code.len() || code[pc].uniform || !code[pc].guard.is_always() {
        return false;
    }
    let original = match &code[pc].op {
        Op::Atom {
            d: Some(register), ..
        } if plain_atom(&code[pc].op) && register.file == File::V => *register,
        _ => return false,
    };
    let next = pc + 1;
    if next >= code.len()
        || code[next].uniform
        || !code[next].guard.is_always()
        || !cancels(&code[next].op, original)
    {
        return false;
    }
    if overwrites_return(&code[next].op, original) {
        return true;
    }
    let mut position = next + 1;
    let mut accepted = true;
    let mut finished = false;
    while position < code.len() && accepted && !finished {
        let instruction = &code[position];
        if instruction.uniform
            || !instruction.guard.is_always()
            || !linear_private(&instruction.op)
            || reads_return(&instruction.op, original)
        {
            accepted = false;
        } else if matches!(instruction.op, Op::Exit) || overwrites_return(&instruction.op, original)
        {
            finished = true;
        } else {
            position += 1;
        }
    }
    accepted && finished
}

pub fn certificate_candidate(front: Front, code: &[Inst]) -> bool {
    if front != Front::Ptx {
        return false;
    }
    let mut found = false;
    let mut pc = 0;
    let mut accepted = true;
    while pc < code.len() && accepted {
        let instruction = &code[pc];
        if instruction.uniform || !instruction.guard.is_always() || !linear_private(&instruction.op)
        {
            accepted = false;
        } else {
            if matches!(instruction.op, Op::Atom { d: Some(_), .. }) {
                accepted = cancelled_return(front, code, pc);
                found = true;
            }
            pc += 1;
        }
    }
    accepted && found
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemSync;
    use crate::Pred;
    use crate::Scope;
    use crate::Space;

    fn reg(index: u16) -> Reg {
        Reg {
            file: File::V,
            idx: index,
        }
    }
    fn instruction(op: Op) -> Inst {
        Inst {
            guard: Pred::TRUE,
            op,
            uniform: false,
            loc: 0,
            text: String::new(),
        }
    }
    fn atomic() -> Inst {
        instruction(Op::Atom {
            op: AtomOp::Add,
            space: Space::Global,
            bytes: 4,
            addr: Addr {
                base: Some(Val::Reg(reg(0), 1)),
                base2: None,
                off: 0,
            },
            v: Val::Imm(1),
            d: Some(reg(3)),
            sync: MemSync {
                sem: Sem::Relaxed,
                scope: Scope::Gpu,
            },
            p: None,
            new: None,
        })
    }
    fn cancel(op: IntOp) -> Inst {
        instruction(Op::Int {
            op,
            bits: 32,
            d: reg(5),
            a: Val::Reg(reg(3), 1),
            b: Val::Reg(reg(3), 1),
        })
    }
    fn program(op: IntOp) -> Vec<Inst> {
        vec![atomic(), cancel(op), instruction(Op::Exit)]
    }

    #[test]
    fn sub_and_xor_cancel_only_exact_scalar_private_returns() {
        for op in [IntOp::Sub, IntOp::Xor] {
            let code = program(op);
            assert!(cancelled_return(Front::Ptx, &code, 0));
            assert!(certificate_candidate(Front::Ptx, &code));
            assert!(!certificate_candidate(Front::Sass, &code));
        }
        for op in [IntOp::Add, IntOp::Mul, IntOp::And, IntOp::DivU] {
            assert!(!certificate_candidate(Front::Ptx, &program(op)));
        }
    }

    #[test]
    fn guards_width_aliases_and_observations_refuse_cancellation() {
        for change in 0..9 {
            let mut code = program(IntOp::Sub);
            match change {
                0 => {
                    code[1].guard = Pred {
                        reg: Some(Reg {
                            file: File::P,
                            idx: 0,
                        }),
                        neg: false,
                    }
                }
                1 => {
                    let Op::Int { bits, .. } = &mut code[1].op else {
                        unreachable!()
                    };
                    *bits = 64;
                }
                2 => {
                    let Op::Int { b, .. } = &mut code[1].op else {
                        unreachable!()
                    };
                    *b = Val::Reg(reg(4), 1);
                }
                3 => {
                    let Op::Int { b, .. } = &mut code[1].op else {
                        unreachable!()
                    };
                    *b = Val::Reg(reg(3), 2);
                }
                4 => code.insert(
                    2,
                    instruction(Op::Mov {
                        d: reg(9),
                        a: Val::Reg(reg(3), 1),
                    }),
                ),
                5 => code.insert(
                    2,
                    instruction(Op::St {
                        space: Space::Global,
                        bytes: 8,
                        addr: Addr {
                            base: Some(Val::Reg(reg(0), 1)),
                            base2: None,
                            off: 0,
                        },
                        v: Val::Reg(reg(2), 2),
                        sync: None,
                    }),
                ),
                6 => code[1].uniform = true,
                7 => {
                    let Op::Atom { sync, .. } = &mut code[0].op else {
                        unreachable!()
                    };
                    sync.sem = Sem::Acquire;
                }
                8 => {
                    let Op::Int { d, .. } = &mut code[1].op else {
                        unreachable!()
                    };
                    d.file = File::UR;
                }
                _ => unreachable!(),
            }
            assert!(
                !certificate_candidate(Front::Ptx, &code),
                "near miss {change}"
            );
        }
    }

    #[test]
    fn unguarded_overwrite_ends_the_old_return_lifetime() {
        let mut code = program(IntOp::Xor);
        code.insert(
            2,
            instruction(Op::Mov {
                d: reg(3),
                a: Val::Imm(11),
            }),
        );
        code.insert(
            3,
            instruction(Op::Mov {
                d: reg(6),
                a: Val::Reg(reg(3), 1),
            }),
        );
        assert!(certificate_candidate(Front::Ptx, &code));
        code[2].guard = Pred {
            reg: Some(Reg {
                file: File::P,
                idx: 0,
            }),
            neg: false,
        };
        assert!(!certificate_candidate(Front::Ptx, &code));
    }
}
