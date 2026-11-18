/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.Code.Funs
import Instruction.LookaheadControlModel

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace LookaheadControl

def outcomeView : control_transfer.Outcome → Plan
  | .Next => .next
  | .Jump target => .jump target.val
  | .Fork targets => .alternatives (targets.val.map (·.val))
  | .Wild _ => .giveUp

def fallthroughFits : control_transfer.Outcome → Prop
  | .Fork targets => targets.val.length + 1 ≤ Usize.max
  | _ => True

@[step] theorem source_with_fallthrough_preserves_both_paths
    (taken : control_transfer.Outcome) (pc : U32) (fits : fallthroughFits taken) :
    control_transfer.with_fallthrough taken pc ⦃ result =>
      permits pc.val (outcomeView result) (.at (pc.val + 1)) ∧
      ∀ observation, permits pc.val (outcomeView taken) observation →
        permits pc.val (outcomeView result) observation ⦄ := by
  unfold control_transfer.with_fallthrough
  simp only [lift, bind_ok]
  cases checked : U32.checked_add pc 1#u32 with
  | none => simp [WP.spec_ok, outcomeView, permits]
  | some next =>
    have nextExact := U32.checked_add_bv_spec pc 1#u32
    rw [checked] at nextExact
    have nextValue : next.val = pc.val + 1 := by scalar_tac
    cases taken with
    | Next => simp [WP.spec_ok, outcomeView, permits]
    | Wild fault => simp [WP.spec_ok, outcomeView, permits]
    | Jump target =>
      step as ⟨first, firstExact⟩ ; try scalar_tac
      step as ⟨targets, targetsExact⟩ ; try scalar_tac
      simp_all [outcomeView, permits]
      intro observation covered
      cases observation <;> simp_all
    | Fork targets =>
      have room : targets.val.length < Usize.max := by simpa [fallthroughFits] using fits
      step as ⟨result, resultExact⟩ ; try scalar_tac
      simp_all [outcomeView, permits]
      intro observation covered
      cases observation <;> simp_all

end LookaheadControl
