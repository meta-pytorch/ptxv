/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.PayloadSupportModel
import Instruction.StoreWindowProducer
import Instruction.AtomicRefinement

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction

@[step] theorem payload_front_supported_spec (front : Front) (file : File) :
    payload.front_supported front file ⦃ result =>
      result = match front with | .Ptx => true | .Sass => decide (file ≠ .V) ⦄ := by
  cases front with
  | Ptx => simp [payload.front_supported]
  | Sass => exact file_ne file .V

@[step] theorem payload_width_supported_spec (file : File) (elements bytes : U8) (exchange : Bool) :
    payload.width_supported file elements bytes exchange ⦃ result =>
      result = decide (payloadWidthSupported file elements bytes exchange) ⦄ := by
  have h4 : bytes = 4#u8 ↔ bytes.val = 4 := by scalar_tac
  have h8 : bytes = 8#u8 ↔ bytes.val = 8 := by scalar_tac
  have hsmall : bytes ≤ 8#u8 ↔ bytes.val ≤ 8 := by scalar_tac
  have helements : elements ≥ 2#u8 ↔ 2 ≤ elements.val := by scalar_tac
  have hplain :
      (if bytes <= 8#u8 then ok true else do
        let remainder ← bytes % 4#u8
        ok (decide (remainder = 0#u8))) ⦃ result =>
          result = decide (bytes.val ≤ 8 ∨ bytes.val % 4 = 0) ⦄ := by
    by_cases hs : bytes.val ≤ 8
    · simp [hsmall, hs]
    · simp only [hsmall, hs, ite_false]
      step as ⟨remainder, hremainder⟩ <;> scalar_tac
  unfold payload.width_supported
  cases exchange with
  | true =>
    by_cases hb4 : bytes.val = 4 <;> by_cases hb8 : bytes.val = 8 <;>
      simp [h4, h8, hb4, hb8, payloadWidthSupported]
  | false =>
    simp only [Bool.false_eq_true, ite_false]
    step with file_eq file .V as ⟨virtual, hvirtual⟩
    rw [hvirtual]
    by_cases hf : file = .V
    · simp only [hf, decide_true, ite_true]
      by_cases he : 2 ≤ elements.val
      · simp only [helements, he, ite_true]
        step as ⟨remainder, hremainder⟩; try scalar_tac
        have hzero : remainder = 0#u8 ↔ remainder.val = 0 := by scalar_tac
        by_cases hm : bytes.val % elements.val = 0
        · simp only [hzero, hremainder, hm, ite_true]
          step as ⟨quotient, hquotient⟩; try scalar_tac
          have hq : quotient ≤ 8#u8 ↔ quotient.val ≤ 8 := by scalar_tac
          simp [hq, hquotient, payloadWidthSupported, hf, he, hm]
        · simp [hzero, hremainder, hm, payloadWidthSupported, hf, he]
      · simpa [helements, he, payloadWidthSupported, hf] using hplain
    · simpa [hf, payloadWidthSupported] using hplain

@[step] theorem payload_supported_total_spec (front : Front) (operation : Op) :
    payload.supported_total front operation ⦃ result =>
      result.map UScalar.val = supportedPayloadTotal front operation ⦄ := by
  classical
  unfold payload.supported_total
  step with store_descriptor_spec operation as ⟨descriptor, hdescriptor⟩
  rw [hdescriptor]
  cases hd : storeDescriptor operation with
  | none => simp [hd, supportedPayloadTotal]
  | some descriptor =>
    rcases descriptor with ⟨first, elements, bytes, exchange⟩
    simp only [hd]
    step with payload_front_supported_spec front first.file as ⟨frontAllowed, hfrontAllowed⟩
    have hf : frontAllowed = decide (¬(front = .Sass ∧ first.file = .V)) := by
      cases front <;> simpa using hfrontAllowed
    rw [hf]
    simp only [decide_eq_true_eq]
    by_cases hfront : ¬(front = .Sass ∧ first.file = .V)
    · rw [if_pos hfront]
      step with payload_width_supported_spec first.file elements bytes exchange as ⟨widthAllowed, hwidthAllowed⟩
      rw [hwidthAllowed]
      by_cases hwidth : payloadWidthSupported first.file elements bytes exchange
      all_goals simp [hwidth, supportedPayloadTotal, hd, payloadLayoutSupported, hfront]
    · rw [if_neg hfront]
      simp [hfront, supportedPayloadTotal, hd, payloadLayoutSupported]

end Instruction
