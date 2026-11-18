/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

use crate::effects;
use crate::effects::AccessKind;
use crate::effects::Effect;
use crate::effects::Visibility;
use crate::machine_recorder::Access;
use crate::machine_registers::WarpRegisters;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct WarpCore {
    pub registers: WarpRegisters,
    /// Lanes recorded by BSSY for each convergence barrier (E-BSYNC).
    pub bmask: [u32; 16],
    /// Warp LDGDEPBAR executions, counted across divergent paths (E-CPASYNC).
    pub cp_seq: u32,
}

impl std::ops::Deref for WarpCore {
    type Target = WarpRegisters;

    fn deref(&self) -> &Self::Target {
        &self.registers
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaskOp {
    Insert(u32),
    Replace(u32),
    Remove(u32),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaskWrite {
    pub warp: u32,
    pub barrier: u8,
    pub op: MaskOp,
}

pub fn mask_access(warp: u32, barrier: u8) -> Access {
    Access {
        resource: effects::convergence_barrier(warp, barrier),
        kind: AccessKind::Write,
        visibility: Visibility::All,
    }
}

pub fn mask_effect(warp: u32, barrier: u8) -> Effect {
    let access = mask_access(warp, barrier);
    Effect::Access {
        resource: access.resource,
        kind: access.kind,
        visibility: access.visibility,
    }
}

pub fn write_mask(state: &mut WarpCore, write: MaskWrite) -> Option<Access> {
    let index = usize::from(write.barrier);
    state.bmask[index] = match write.op {
        MaskOp::Insert(lanes) => state.bmask[index] | lanes,
        MaskOp::Replace(lanes) => lanes,
        MaskOp::Remove(lanes) => state.bmask[index] & !lanes,
    };
    Some(mask_access(write.warp, write.barrier))
}

pub fn commit_async(state: &mut WarpCore) {
    state.cp_seq += 1;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state() -> WarpCore {
        WarpCore {
            registers: WarpRegisters {
                ur: vec![11, 12],
                ur_written: [13, 14, 15, 16],
                up: 17,
                up_def: 18,
            },
            bmask: [0x8000_0001; 16],
            cp_seq: 19,
        }
    }

    #[test]
    fn mask_mutations_return_the_exact_owner_and_preserve_other_state() {
        for barrier in [0, 15] {
            for op in [MaskOp::Insert(3), MaskOp::Replace(3), MaskOp::Remove(3)] {
                let mut actual = state();
                let before = actual.clone();
                let access = write_mask(
                    &mut actual,
                    MaskWrite {
                        warp: 23,
                        barrier,
                        op,
                    },
                );
                assert_eq!(access, Some(mask_access(23, barrier)));
                assert_eq!(actual.registers, before.registers);
                assert_eq!(actual.cp_seq, before.cp_seq);
                for index in 0..16 {
                    assert_eq!(
                        actual.bmask[index],
                        if index == usize::from(barrier) {
                            match op {
                                MaskOp::Insert(_) => 0x8000_0003,
                                MaskOp::Replace(_) => 3,
                                MaskOp::Remove(_) => 0x8000_0000,
                            }
                        } else {
                            before.bmask[index]
                        }
                    );
                }
            }
        }
    }

    #[test]
    fn async_commit_preserves_registers_and_convergence_masks() {
        let mut actual = state();
        let before = actual.clone();
        commit_async(&mut actual);
        assert_eq!(actual.cp_seq, 20);
        assert_eq!(actual.registers, before.registers);
        assert_eq!(actual.bmask, before.bmask);
    }
}
