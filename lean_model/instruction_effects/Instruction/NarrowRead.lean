/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.Model
import Instruction.Code.Funs

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace Instruction

structure ReadWindow where
  source : Val
  start : Nat
  width : Nat

def narrowReadWindow (operation : Op) : Option ReadWindow :=
  match operation with
  | .FCvt _ source .F16 _ _ _ _ high | .FCvt _ source .Bf16 _ _ _ _ high |
      .CvtF32F16 _ source high => some ⟨source, if high then 16 else 0, 16⟩
  | .I2F _ source _ _ width high _ =>
      if width.val = 8 ∨ width.val = 16 then some ⟨source, if high then 16 else 0, width.val⟩ else none
  | _ => none

def windowObservation (window : Option (Val × U32 × U32)) : Option ReadWindow :=
  window.map fun (source, start, width) => ⟨source, start.val, width.val⟩

def consumesBit (window : ReadWindow) (source : Val) (bit : Nat) : Prop :=
  source = window.source ∧ window.start ≤ bit ∧ bit < window.start + window.width

@[step] theorem narrow_read_window_spec (operation : Op) :
    read_window.narrow operation
      ⦃ result => windowObservation result = narrowReadWindow operation ⦄ := by
  cases operation <;> try (solve | simp [read_window.narrow, windowObservation, narrowReadWindow])
  case FCvt destination source format target round ftz integral high =>
    cases format <;> cases high <;> simp [read_window.narrow, windowObservation, narrowReadWindow]
  case CvtF32F16 destination source high =>
    cases high <;> simp [read_window.narrow, windowObservation, narrowReadWindow]
  case I2F destination source signed round width high wide =>
    have h8 : width.val = 8 ↔ width = 8#u8 := by scalar_tac
    have h16 : width.val = 16 ↔ width = 16#u8 := by scalar_tac
    by_cases hw8 : width = 8#u8 <;> by_cases hw16 : width = 16#u8 <;> cases high <;>
      simp [read_window.narrow, windowObservation, narrowReadWindow, h8, h16, hw8, hw16]
    all_goals simp_all [lift, bind_ok]

theorem consumes_exact_interval (window : ReadWindow) (source : Val) (bit : Nat) :
    consumesBit window source bit ↔
      source = window.source ∧ bit ∈ Finset.Ico window.start (window.start + window.width) := by
  simp [consumesBit]

theorem narrow_window_within_source_word (operation : Op) (window : ReadWindow)
    (hwindow : narrowReadWindow operation = some window) :
    window.start + window.width ≤ 32 := by
  cases operation <;> try (solve | simp [narrowReadWindow] at hwindow)
  case FCvt destination source format target round ftz integral high =>
    cases format <;> cases high <;> simp [narrowReadWindow] at hwindow <;>
      subst window <;> simp
  case CvtF32F16 destination source high =>
    cases high <;> simp [narrowReadWindow] at hwindow <;> subst window <;> simp
  case I2F destination source signed round width high wide =>
    by_cases hw : width.val = 8 ∨ width.val = 16
    · simp [narrowReadWindow, hw] at hwindow
      subst window
      cases high <;> simp <;> omega
    · simp [narrowReadWindow, hw] at hwindow

end Instruction
