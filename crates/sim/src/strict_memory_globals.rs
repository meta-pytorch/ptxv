/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use ptxv_ir::Addr;
use ptxv_ir::Inst;
use ptxv_ir::Op;
use ptxv_ir::SReg;
use ptxv_ir::Space;
use ptxv_ir::Val;

use crate::strict_memory_admission::Refusal;

pub fn global_address(index: u32, bindings: &[(u8, u32, u64)]) -> Option<u64> {
    let offset = match index.checked_mul(8) {
        Some(offset) => offset,
        None => return None,
    };
    let mut found = None;
    let mut cursor = 0;
    while cursor < bindings.len() && found.is_none() {
        let (bank, location, value) = bindings[cursor];
        if bank == ptxv_ir::GLOBAL_RELOC_BANK && location == offset {
            found = Some(value);
        }
        cursor += 1;
    }
    found
}

pub fn address_value(value: &Option<Val>, bindings: &[(u8, u32, u64)]) -> Option<u64> {
    match value {
        None => Some(0),
        Some(Val::Imm(value)) => Some(*value),
        Some(Val::SReg(SReg::GlobalAddr(index))) => global_address(*index, bindings),
        _ => None,
    }
}

pub fn resolve_address(address: &Addr, bindings: &[(u8, u32, u64)]) -> Option<u64> {
    let first = match address_value(&address.base, bindings) {
        Some(value) => value,
        None => return None,
    };
    let second = match address_value(&address.base2, bindings) {
        Some(value) => value,
        None => return None,
    };
    Some(
        (address.off as u64)
            .wrapping_add(first)
            .wrapping_add(second),
    )
}

pub fn specialize_instruction(
    instruction: &Inst,
    bindings: &[(u8, u32, u64)],
    pc: usize,
) -> Result<Inst, Refusal> {
    let op = match &instruction.op {
        Op::Ld {
            d,
            n: 1,
            space: Space::Global,
            bytes,
            addr,
            sync,
            sext,
            nc,
        } => {
            let resolved = match resolve_address(addr, bindings) {
                Some(address) => address,
                None => return Err(Refusal::Address(pc)),
            };
            Op::Ld {
                d: *d,
                n: 1,
                space: Space::Global,
                bytes: *bytes,
                addr: Addr {
                    base: Some(Val::Imm(resolved)),
                    base2: None,
                    off: 0,
                },
                sync: *sync,
                sext: *sext,
                nc: *nc,
            }
        }
        Op::St {
            space: Space::Global,
            bytes,
            addr,
            v: Val::Imm(value),
            sync,
        } => {
            let resolved = match resolve_address(addr, bindings) {
                Some(address) => address,
                None => return Err(Refusal::Address(pc)),
            };
            Op::St {
                space: Space::Global,
                bytes: *bytes,
                addr: Addr {
                    base: Some(Val::Imm(resolved)),
                    base2: None,
                    off: 0,
                },
                v: Val::Imm(*value),
                sync: *sync,
            }
        }
        Op::Nop => Op::Nop,
        Op::Exit => Op::Exit,
        _ => return Err(Refusal::Instruction(pc)),
    };
    Ok(Inst {
        guard: instruction.guard,
        op,
        uniform: instruction.uniform,
        loc: instruction.loc,
        text: String::new(),
    })
}

pub fn specialize_code(code: &[Inst], bindings: &[(u8, u32, u64)]) -> Result<Vec<Inst>, Refusal> {
    let mut result = Vec::new();
    let mut failure = None;
    let mut pc = 0;
    while pc < code.len() && failure.is_none() {
        match specialize_instruction(&code[pc], bindings, pc) {
            Ok(instruction) => result.push(instruction),
            Err(reason) => failure = Some(reason),
        }
        pc += 1;
    }
    match failure {
        Some(reason) => Err(reason),
        None => Ok(result),
    }
}

#[cfg(test)]
mod tests {
    use ptxv_ir::File;
    use ptxv_ir::Pred;
    use ptxv_ir::Reg;
    use ptxv_memory_write::memory_safety::AccessKind;
    use ptxv_memory_write::memory_safety::Instruction;

    use super::*;

    #[test]
    fn global_bindings_use_first_match_and_refuse_missing_or_overflowed_offsets() {
        let bindings = [(0, 8, 12), (31, 8, 1234), (31, 8, 9999)];
        assert_eq!(global_address(1, &bindings), Some(1234));
        assert_eq!(global_address(0, &bindings), None);
        assert_eq!(global_address(u32::MAX, &bindings), None);
        assert_eq!(global_address(0, &[]), None);
    }

    #[test]
    fn specialized_addresses_preserve_wrapping_signed_offsets_and_both_bases() {
        let address = Addr {
            base: Some(Val::SReg(SReg::GlobalAddr(2))),
            base2: Some(Val::Imm(10)),
            off: -12,
        };
        assert_eq!(resolve_address(&address, &[(31, 16, 1)]), Some(u64::MAX));
        let dynamic = Addr {
            base: Some(Val::Reg(
                Reg {
                    file: File::V,
                    idx: 0,
                },
                1,
            )),
            ..address
        };
        assert_eq!(resolve_address(&dynamic, &[(31, 16, 1)]), None);
    }

    #[test]
    fn specializing_code_preserves_pcs_and_all_semantic_load_fields() {
        let load = Inst {
            guard: Pred::TRUE,
            op: Op::Ld {
                d: Reg {
                    file: File::V,
                    idx: 0,
                },
                n: 1,
                space: Space::Global,
                bytes: 8,
                addr: Addr {
                    base: Some(Val::SReg(SReg::GlobalAddr(0))),
                    base2: None,
                    off: 8,
                },
                sync: None,
                sext: false,
                nc: false,
            },
            uniform: false,
            loc: 91,
            text: "ld.global.u64 %r0, [x+8]".into(),
        };
        let exit = Inst {
            op: Op::Exit,
            loc: 92,
            ..load.clone()
        };
        let specialized = specialize_code(&[load.clone(), exit], &[(31, 0, 4096)]).unwrap();
        assert_eq!(specialized.len(), 2);
        assert_eq!(specialized[0].guard, load.guard);
        assert_eq!(specialized[0].uniform, load.uniform);
        assert_eq!(specialized[0].loc, load.loc);
        assert!(specialized[0].text.is_empty());
        let projected = crate::strict_memory_admission::project_code(&specialized, 1).unwrap();
        assert!(
            matches!(&projected[..], [Instruction::Access(request), Instruction::Halt]
            if request.address == 4104 && request.width == 8 && request.kind == AccessKind::Load)
        );
        let mut refused = load;
        refused.uniform = true;
        let specialized = specialize_code(&[refused], &[(31, 0, 4096)]).unwrap();
        assert!(
            crate::strict_memory_admission::project_instruction(&specialized[0], 1, 0, false)
                .is_err()
        );
    }
}
