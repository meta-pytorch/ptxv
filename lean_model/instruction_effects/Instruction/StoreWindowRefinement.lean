/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.StoreWindowModel
import Instruction.AtomicRefinement

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction

@[simp] theorem empty_window_vector :
    (alloc.vec.Vec.new (Reg × U32 × U32)).val = [] := rfl

@[step] theorem clipped_size_spec (total start stride : U32) :
    store_window.clipped_size total start stride ⦃ result =>
      result.val = min (total.val - start.val) stride.val ⦄ := by
  unfold store_window.clipped_size
  split
  · rename_i htest
    step as ⟨remaining, hremaining⟩
    simp only [core.cmp.impls.OrdU32.min]
    split <;> scalar_tac
  · rename_i htest
    simp only [WP.spec_ok]
    scalar_tac

@[step] theorem append_windows_spec (first : Reg) (count : U16) (stride total : U32)
    (hindex : first.idx.val + count.val ≤ 65536)
    (hcount : count.val ≤ 255) (hstride : stride.val ≤ 255) :
    store_window.append_windows first count stride total ⦃ result =>
      result.val.map byteWindow = sourceWindows first count.val stride.val total.val ∧
      result.val.length = count.val ⦄ := by
  unfold store_window.append_windows store_window.append_windows_loop
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec (Reg × U32 × U32) × U16 => count.val - state.2.val)
    (fun state : alloc.vec.Vec (Reg × U32 × U32) × U16 =>
      state.2.val ≤ count.val ∧
      state.1.val.map byteWindow = sourceWindows first state.2.val stride.val total.val ∧
      state.1.val.length = state.2.val)
  · rintro ⟨result, index⟩ ⟨hbound, hmap, hlength⟩
    dsimp only at hbound hmap hlength
    unfold store_window.append_windows_loop.body
    by_cases hlt : index.val < count.val
    · have htest : index < count := by scalar_tac
      simp only [htest, ite_true, lift, bind_ok]
      have hproduct : index.val * stride.val ≤ 255 * 255 :=
        Nat.mul_le_mul (by omega) hstride
      step as ⟨start, hstart⟩; try scalar_tac
      step as ⟨size, hsize⟩
      step as ⟨position, hposition⟩; try scalar_tac
      step with alloc.vec.Vec.push_spec result ({ first with idx := position }, start, size)
        (by scalar_tac) as ⟨pushed, hpushed⟩
      step as ⟨next, hnext⟩; try scalar_tac
      refine ⟨?_, ?_, ?_, ?_⟩
      · scalar_tac
      · rw [hpushed, List.map_append, hmap]
        have hnextval : next.val = index.val + 1 := by scalar_tac
        rw [hnextval, source_windows_succ]
        congr 1
        simp only [List.map_cons, List.map_nil, List.cons.injEq, and_true]
        have hpositionval : position.val = first.idx.val + index.val := by scalar_tac
        have hstartval : start.val = index.val * stride.val := by scalar_tac
        simp [byteWindow, key, hpositionval, hstartval, hsize]
      · simp only [hpushed, List.length_append, List.length_cons, List.length_nil] at *
        scalar_tac
      · scalar_tac
    · have htest : ¬index < count := by scalar_tac
      simp only [htest, ite_false, WP.spec_ok]
      have heq : index.val = count.val := by omega
      exact ⟨by simpa [heq] using hmap, by simpa [heq] using hlength⟩
  · simp [sourceWindows]

@[step] theorem repeated_virtual_spec (front : Front) (file : File) (elements bytes : U8) :
    register_effect.repeated_virtual front file elements bytes ⦃ result =>
      result = decide (front = .Sass ∧ file = .V ∧ elements.val = 1 ∧ bytes.val > 8) ⦄ := by
  have heq : elements = 1#u8 ↔ elements.val = 1 := by scalar_tac
  have hb8eq : bytes > 8#u8 ↔ bytes.val > 8 := by scalar_tac
  cases front <;> cases file <;>
    simp [register_effect.repeated_virtual, Front.Insts.CoreCmpPartialEqFront.eq,
      Front.read_discriminant, File.Insts.CoreCmpPartialEqFile.eq,
      File.read_discriminant, heq, hb8eq]
  all_goals by_cases he : elements.val = 1 <;> simp [he]

@[step] theorem repeated_windows_spec (first : Reg) (bytes : U8) :
    store_window.repeated_windows first bytes ⦃ result =>
      result.val.map byteWindow = repeatedWindows first ((bytes.val + 7) / 8) bytes.val ⦄ := by
  unfold store_window.repeated_windows
  step as ⟨total, htotal⟩
  have htotalval : total.val = bytes.val := by scalar_tac
  step as ⟨padded, hpadded⟩; try scalar_tac
  step as ⟨count, hcount⟩; try scalar_tac
  have hcountval : count.val = (bytes.val + 7) / 8 := by scalar_tac
  have hcountbound : count.val ≤ 32 := by scalar_tac
  unfold store_window.repeated_windows_loop
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec (Reg × U32 × U32) × U32 => count.val - state.2.val)
    (fun state : alloc.vec.Vec (Reg × U32 × U32) × U32 =>
      state.2.val ≤ count.val ∧
      state.1.val.map byteWindow = repeatedWindows first state.2.val bytes.val ∧
      state.1.val.length = state.2.val)
  · rintro ⟨result, index⟩ ⟨hbound, hmap, hlength⟩
    dsimp only at hbound hmap hlength
    unfold store_window.repeated_windows_loop.body
    by_cases hlt : index.val < count.val
    · have htest : index < count := by scalar_tac
      simp only [htest, ite_true, lift, bind_ok]
      step as ⟨start, hstart⟩; try scalar_tac
      step as ⟨size, hsize⟩
      step with alloc.vec.Vec.push_spec result (first, start, size)
        (by scalar_tac) as ⟨pushed, hpushed⟩
      step as ⟨next, hnext⟩; try scalar_tac
      refine ⟨?_, ?_, ?_, ?_⟩
      · scalar_tac
      · rw [hpushed, List.map_append, hmap]
        have hnextval : next.val = index.val + 1 := by scalar_tac
        rw [hnextval, repeated_windows_succ]
        congr 1
        simp only [List.map_cons, List.map_nil, List.cons.injEq, and_true]
        have hstartval : start.val = index.val * 8 := by scalar_tac
        simp [byteWindow, hstartval, hsize, htotalval]
      · simp only [hpushed, List.length_append, List.length_cons, List.length_nil] at *
        scalar_tac
      · scalar_tac
    · have htest : ¬index < count := by scalar_tac
      simp only [htest, ite_false, WP.spec_ok]
      have heq : index.val = count.val := by omega
      simpa [heq, hcountval] using hmap
  · simp [repeatedWindows]

end Instruction
