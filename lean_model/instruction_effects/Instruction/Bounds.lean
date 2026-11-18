/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.Model

open Aeneas Aeneas.Std PtxvIr
set_option autoImplicit false
namespace Instruction

theorem operandReads_length (value : Val) : (operandReads value).length ≤ 255 := by
  cases value with
  | Reg first count =>
    simp only [operandReads, scalarSlots, consecutive_length]
    split <;> scalar_tac
  | Imm _ | Const _ _ | SReg _ => simp [operandReads]

theorem vectorReads_length (value : Val) : (vectorReads value).length ≤ 255 := by
  cases value with
  | Reg first count =>
    simp only [vectorReads, consecutive_length]
    split
    · scalar_tac
    · simp only [scalarSlots]; split <;> scalar_tac
  | Imm _ | Const _ _ | SReg _ => simp [vectorReads]

theorem storeValueReads_length (asynchronous : Bool) (bytes : U8) (value : Val) :
    (storeValueReads asynchronous bytes value).length ≤ 255 := by
  cases value with
  | Reg first count =>
    simp only [storeValueReads, consecutive_length, storeSlots, scalarSlots]
    split
    · scalar_tac
    · cases asynchronous <;> simp only [Bool.false_eq_true, ite_false, ite_true]
      all_goals split <;> first | scalar_tac | (split <;> scalar_tac)
  | Imm _ | Const _ _ | SReg _ => simp [storeValueReads]

theorem addressReads_length (address : Addr) : (addressReads address).length ≤ 510 := by
  rcases address with ⟨base, base2, offset⟩
  cases base with
  | none =>
    cases base2 with
    | none => simp [addressReads]
    | some right => simpa [addressReads] using (Nat.le_trans (operandReads_length right) (by omega : 255 ≤ 510))
  | some left =>
    have hl := operandReads_length left
    cases base2 with
    | none => simpa [addressReads] using (Nat.le_trans hl (by omega : 255 ≤ 510))
    | some right =>
      have hr := operandReads_length right
      simp only [addressReads, Option.toList_some, List.flatMap_cons, List.flatMap_nil,
        List.append_nil, List.length_append]
      omega

theorem offset_sequence_bound (file : File) (start count : Nat) (hpos : 0 < count)
    (hindex : Bounded ((List.range count).map fun index => (⟨file, start + index⟩ : Key))) :
    start + count ≤ 65536 := by
  have hm : (⟨file, start + (count - 1)⟩ : Key) ∈
      (List.range count).map (fun index => (⟨file, start + index⟩ : Key)) := by
    apply List.mem_map.mpr
    exact ⟨count - 1, List.mem_range.mpr (by omega), rfl⟩
  have hb := hindex _ hm
  dsimp only at hb
  omega

theorem casReplacement_length (bytes : U8) (compare : Val) (replacement : Option Val) :
    (casReplacement bytes compare replacement).length ≤ 255 := by
  cases replacement with
  | some value => simpa [casReplacement] using operandReads_length value
  | none =>
    cases compare with
    | Reg first _ => simp only [casReplacement, List.length_map, List.length_range]; split <;> omega
    | Imm _ | Const _ _ | SReg _ => simp [casReplacement]

end Instruction
