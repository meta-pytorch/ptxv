/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.SourceModel
import ValueFlow.Storage.MachineRegistersDefinedness

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace ValueFlow.Source

open Storage.MachineRegistersDefinedness

 theorem bitAt_value (words : List U64) (index : Nat) :
    bitAt words index = decide ((words[index / 64]?.getD 0#u64).val /
      2 ^ (index % 64) % 2 = 1) := by
  simp only [bitAt, BitVec.getLsbD, Nat.testBit_eq_decide_div_mod_eq,
    List.getElem!_eq_getElem?_getD]
  rfl

 theorem predicateModel_bits (registers : V.Registers) (index : Nat) :
    predicateModel registers index =
      if index / 64 < registers.raw.preds.val.length ∧ index / 64 < registers.raw.pdef.val.length then
        if bitAt registers.raw.pdef.val index then some (bitAt registers.raw.preds.val index) else none
      else none := by
  simp only [predicateModel, bitAt_value, decide_eq_true_eq]

 theorem wordModel_bits (registers : V.Registers) (index : Nat) :
    wordModel registers index =
      if index < registers.raw.regs.val.length ∧ index < registers.pointers.val.length ∧
          index / 64 < registers.raw.written.val.length then
        if bitAt registers.raw.written.val index then some {
          bits := (registers.raw.regs.val[index]?.getD 0#u64).val
          undefined := undefinedBits registers.raw.vund.val index
          pointer := (registers.pointers.val[index]?.getD none).map pointerModel }
        else none
      else none := by
  simp only [wordModel, bitAt_value, decide_eq_true_eq]

end ValueFlow.Source
