/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.PayloadAssembly
import Instruction.PayloadSupportModel

open Aeneas Aeneas.Std PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace Instruction

theorem supported_payload_stride_bound (front : Front) (first : Reg) (elements bytes : U8)
    (exchange : Bool) (hsupported : payloadLayoutSupported front first elements bytes exchange) :
    layoutStride first.file elements bytes exchange ≤ 8 := by
  obtain ⟨_hfront, hvalue⟩ := hsupported
  cases exchange with
  | false =>
    simp only [payloadWidthSupported, Bool.false_eq_true, ite_false] at hvalue
    by_cases hf : first.file = .V
    · by_cases he : 2 ≤ elements.val
      · have hv : bytes.val % elements.val = 0 ∧ bytes.val / elements.val ≤ 8 := by
          simpa [hf, he] using hvalue
        simpa [layoutStride, hf, he] using hv.2
      · by_cases hb : bytes.val > 8 <;> simp [layoutStride, hf, he, hb]
        omega
    · simp [layoutStride, hf]
  | true =>
    simp only [payloadWidthSupported, ite_true] at hvalue
    by_cases hf : first.file = .V
    · simp [layoutStride, hf]
      omega
    · simp [layoutStride, hf]

def payloadWindowsOrdered (windows : List ByteWindow) : Prop :=
  ∀ index (hi : index < windows.length), beforeWindow (windows.take index) (windows[index]'hi)

theorem layout_windows_valid (first : Reg) (elements bytes : U8) (exchange : Bool)
    (hstride : layoutStride first.file elements bytes exchange ≤ 8)
    (windows : List (Reg × U32 × U32))
    (hmap : windows.map byteWindow = layoutWindows first elements bytes exchange) :
    payloadWindowsValid windows bytes.val := by
  intro index hi
  have hm : byteWindow (windows[index]'hi) ∈ layoutWindows first elements bytes exchange := by
    rw [← hmap]
    exact List.mem_map.mpr ⟨windows[index]'hi, List.getElem_mem hi, rfl⟩
  obtain ⟨hcount, hwithin⟩ := source_window_bounds first
    (layoutCount first.file elements bytes exchange)
    (layoutStride first.file elements bytes exchange) bytes.val _ hm
  refine ⟨Nat.le_trans hcount hstride, hwithin, ?_⟩
  have hmodelOrder : payloadWindowsOrdered (layoutWindows first elements bytes exchange) := by
    intro position hp
    apply source_windows_ordered first (layoutCount first.file elements bytes exchange)
      (layoutStride first.file elements bytes exchange) bytes.val position
    simpa [layoutWindows] using hp
  have hrawOrder : payloadWindowsOrdered (windows.map byteWindow) := by
    rw [hmap]
    exact hmodelOrder
  have hordered := hrawOrder index (by simpa using hi)
  simpa only [← List.map_take, List.getElem_map] using hordered

theorem operation_windows_valid (front : Front) (operation : Op) (total : Nat)
    (hsupported : supportedPayloadTotal front operation = some total)
    (windows : List (Reg × U32 × U32))
    (hmap : windows.map byteWindow = storeWindows front operation) :
    payloadWindowsValid windows total := by
  classical
  unfold supportedPayloadTotal at hsupported
  cases hd : storeDescriptor operation with
  | none => simp [hd] at hsupported
  | some descriptor =>
    rcases descriptor with ⟨first, elements, bytes, exchange⟩
    simp only [hd] at hsupported
    split at hsupported
    · rename_i hs
      cases hsupported
      have hlayout : storeWindows front operation = layoutWindows first elements bytes exchange := by
        simp only [storeWindows, hd]
        by_cases hf : first.file = .V
        · cases front with
          | Ptx => simp [hf]
          | Sass => exact False.elim (hs.1 ⟨rfl, hf⟩)
        · simp [hf]
      exact layout_windows_valid first elements bytes exchange
        (supported_payload_stride_bound front first elements bytes exchange hs)
        windows (hmap.trans hlayout)
    · contradiction

end Instruction
