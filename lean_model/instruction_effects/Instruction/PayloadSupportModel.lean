/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.StoreWindowModel

open Aeneas Aeneas.Std PtxvIr
set_option autoImplicit false
namespace Instruction

abbrev payloadWidthSupported (file : File) (elements bytes : U8) (exchange : Bool) : Prop :=
  if exchange then bytes.val = 4 ∨ bytes.val = 8
  else if file = .V ∧ 2 ≤ elements.val then
    bytes.val % elements.val = 0 ∧ bytes.val / elements.val ≤ 8
  else bytes.val ≤ 8 ∨ bytes.val % 4 = 0

def payloadLayoutSupported (front : Front) (first : Reg) (elements bytes : U8)
    (exchange : Bool) : Prop :=
  ¬(front = .Sass ∧ first.file = .V) ∧ payloadWidthSupported first.file elements bytes exchange

noncomputable def supportedPayloadTotal (front : Front) (operation : Op) : Option Nat := by
  classical
  exact match storeDescriptor operation with
  | none => none
  | some (first, elements, bytes, exchange) =>
      if payloadLayoutSupported front first elements bytes exchange then some bytes.val else none

theorem supported_payload_total_bound (front : Front) (operation : Op) (total : Nat)
    (hsupported : supportedPayloadTotal front operation = some total) : total ≤ 255 := by
  unfold supportedPayloadTotal at hsupported
  split at hsupported
  · contradiction
  · split at hsupported
    · cases hsupported
      scalar_tac
    · contradiction

end Instruction
