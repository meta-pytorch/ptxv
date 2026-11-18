/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.St256Model

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction

@[step] theorem st256_sources_loop_spec (low high : Reg)
    (windows : alloc.vec.Vec (Reg × U32 × U32)) (index : U16)
    (hbound : index.val ≤ 8)
    (hbounded : windowsBounded (st256Prefix low high 8))
    (hshape : windows.val.map byteWindow = st256Prefix low high index.val)
    (hlength : windows.val.length = index.val) :
    payload.st256_sources_loop low high windows index ⦃ result =>
      result.val.map byteWindow = st256Prefix low high 8 ∧ result.val.length = 8 ⦄ := by
  unfold payload.st256_sources_loop
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec (Reg × U32 × U32) × U16 => 8 - state.2.val)
    (fun state : alloc.vec.Vec (Reg × U32 × U32) × U16 => state.2.val ≤ 8 ∧
      state.1.val.map byteWindow = st256Prefix low high state.2.val ∧
      state.1.val.length = state.2.val)
  · rintro ⟨current, index⟩ ⟨hbound, hshape, hlength⟩
    dsimp only at hbound hshape hlength
    unfold payload.st256_sources_loop.body
    by_cases hlt : index.val < 8
    · have htest : index < 8#u16 := by scalar_tac
      simp only [htest, ite_true]
      have hchoice : (if index < 4#u16 then ok low else ok high) =
          ok (if index.val < 4 then low else high) := by
        by_cases hl : index.val < 4
        · have ht : index < 4#u16 := by scalar_tac
          simp [hl, ht]
        · have ht : ¬index < 4#u16 := by scalar_tac
          simp [hl, ht]
      rw [hchoice]
      simp only [bind_ok]
      have hregister : (if index.val < 4 then low else high).idx.val + index.val % 4 < 65536 := by
        apply hbounded (st256Window low high index.val).source
        apply List.mem_map.mpr
        refine ⟨st256Window low high index.val, ?_, rfl⟩
        exact List.mem_map.mpr ⟨index.val, List.mem_range.mpr hlt, rfl⟩
      step as ⟨offset, hoffset⟩; try scalar_tac
      step as ⟨position, hposition⟩; try scalar_tac
      simp only [lift, bind_ok]
      step as ⟨firstByte, hfirstByte⟩; try scalar_tac
      step with alloc.vec.Vec.push_spec current
        ({ (if index.val < 4 then low else high) with idx := position }, firstByte, 4#u32)
        (by scalar_tac) as ⟨pushed, hpushed⟩
      step as ⟨next, hnext⟩; try scalar_tac
      refine ⟨?_, ?_, ?_, ?_⟩
      · scalar_tac
      · have hn : next.val = index.val + 1 := by scalar_tac
        have hp : position.val = (if index.val < 4 then low else high).idx.val + index.val % 4 := by
          scalar_tac
        have hb : firstByte.val = 4 * index.val := by scalar_tac
        rw [hpushed, List.map_append, hshape, hn, st256_prefix_succ]
        simp [byteWindow, st256Window, key, hp, hb]
      · simp only [hpushed, List.length_append, List.length_cons, List.length_nil] at *
        scalar_tac
      · scalar_tac
    · have htest : ¬index < 8#u16 := by scalar_tac
      simp only [htest, ite_false, WP.spec_ok]
      have hi : index.val = 8 := by omega
      exact ⟨by simpa [hi] using hshape, by simpa [hi] using hlength⟩
  · exact ⟨hbound, hshape, hlength⟩

@[step] theorem st256_sources_spec (operation : Op)
    (hbounded : ∀ windows, st256Windows operation = some windows → windowsBounded windows) :
    payload.st256_sources operation ⦃ result =>
      result.map (fun windows => windows.val.map byteWindow) = st256Windows operation ⦄ := by
  unfold payload.st256_sources
  cases operation <;> try (simp [st256Windows])
  rename_i low high address sync
  step with st256_sources_loop_spec low high (alloc.vec.Vec.new _) 0#u16 (by simp)
    (hbounded _ rfl) (by simp [st256Prefix]) (by simp) as ⟨windows, hshape, hlength⟩
  simpa [st256Windows] using hshape

end Instruction
