/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.Code.Funs

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace Instruction

def loadedMemoryWord (data : Array U8 32#usize) (source : Nat) : BitVec 32 :=
  (BitVec.fromLEBytes [data.val[4 * source]!.bv, data.val[4 * source + 1]!.bv,
    data.val[4 * source + 2]!.bv, data.val[4 * source + 3]!.bv]).cast (by simp)

@[step] theorem masked_load_word_spec (data : Array U8 32#usize) (source : U8)
    (hsource : source.val < 8) :
    masked_load.word data source ⦃ result => result.bv = loadedMemoryWord data source.val ⦄ := by
  have hlength : data.val.length = 32 := by simp
  unfold masked_load.word
  simp only [lift, bind_ok]
  step as ⟨first, hfirst⟩; try scalar_tac
  have hfirstval : first.val = 4 * source.val := by scalar_tac
  step as ⟨byte0, hbyte0⟩; try scalar_tac
  step as ⟨index1, hindex1⟩; try scalar_tac
  step as ⟨byte1, hbyte1⟩; try scalar_tac
  step as ⟨index2, hindex2⟩; try scalar_tac
  step as ⟨byte2, hbyte2⟩; try scalar_tac
  step as ⟨index3, hindex3⟩; try scalar_tac
  step as ⟨byte3, hbyte3⟩; try scalar_tac
  have hi1 : index1.val = 4 * source.val + 1 := by scalar_tac
  have hi2 : index2.val = 4 * source.val + 2 := by scalar_tac
  have hi3 : index3.val = 4 * source.val + 3 := by scalar_tac
  have hb0 : byte0 = data.val[4 * source.val]! := by
    simpa only [hfirstval, getElem!_pos data.val (4 * source.val) (by omega)] using hbyte0
  have hb1 : byte1 = data.val[4 * source.val + 1]! := by
    simpa only [hi1, getElem!_pos data.val (4 * source.val + 1) (by omega)] using hbyte1
  have hb2 : byte2 = data.val[4 * source.val + 2]! := by
    simpa only [hi2, getElem!_pos data.val (4 * source.val + 2) (by omega)] using hbyte2
  have hb3 : byte3 = data.val[4 * source.val + 3]! := by
    simpa only [hi3, getElem!_pos data.val (4 * source.val + 3) (by omega)] using hbyte3
  change (BitVec.fromLEBytes [byte0.bv, byte1.bv, byte2.bv, byte3.bv]).cast (by simp) =
    loadedMemoryWord data source.val
  unfold loadedMemoryWord
  cases hb0
  cases hb1
  cases hb2
  cases hb3
  rfl

end Instruction
