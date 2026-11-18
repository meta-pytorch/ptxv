/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Code.Funs

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 4000000

namespace MemoryWrite.MachineControl

theorem source_set_pc {S : Type} (state : machine_control.ThreadCore S) (pc : U32) :
    machine_control.set_pc state pc = .ok {state with pc := pc} := rfl

theorem source_set_status {S : Type} (state : machine_control.ThreadCore S) (status : S) :
    machine_control.set_status state status = .ok {state with st := status} := rfl

@[step] theorem source_advance {S : Type} (state : machine_control.ThreadCore S)
    (room : state.pc.val + 1 ≤ 4294967295) :
    machine_control.advance state ⦃ next =>
      next.pc.val = state.pc.val + 1 ∧ next.st = state.st ∧
      next.registers = state.registers ∧ next.acq_bar = state.acq_bar ⦄ := by
  unfold machine_control.advance
  step as ⟨pc, value⟩
  simp [value]

@[step] theorem source_resume {S : Type} (state : machine_control.ThreadCore S) (running : S)
    (room : state.pc.val + 1 ≤ 4294967295) :
    machine_control.resume state running ⦃ next =>
      next.pc.val = state.pc.val + 1 ∧ next.st = running ∧
      next.registers = state.registers ∧ next.acq_bar = state.acq_bar ⦄ := by
  unfold machine_control.resume
  step as ⟨pc, value⟩
  simp [value]

@[step] theorem source_release {S : Type} (state : machine_control.ThreadCore S)
    (running : S) (key : U64) (room : state.pc.val + 1 ≤ 4294967295) :
    machine_control.release state running key ⦃ next =>
      next.pc.val = state.pc.val + 1 ∧ next.st = running ∧
      next.registers = state.registers ∧ next.acq_bar = some key ⦄ := by
  unfold machine_control.release
  step with source_resume state running room as ⟨next, pc, status, registers, acquire⟩
  simp [pc, status, registers]

theorem source_take_acquire {S : Type} (state : machine_control.ThreadCore S) :
    machine_control.take_acquire state = .ok (state.acq_bar, {state with acq_bar := none}) := by
  simp [machine_control.take_acquire, core.option.Option.take]

theorem take_acquire_is_single_use {S : Type} (state : machine_control.ThreadCore S) :
    (do
      let (first, next) ← machine_control.take_acquire state
      let (second, final) ← machine_control.take_acquire next
      .ok (first, second, final)) =
        .ok (state.acq_bar, none, {state with acq_bar := none}) := by
  simp [source_take_acquire]

theorem apply_flow_stay {S : Type} (eq : core.cmp.PartialEq S S)
    (state : machine_control.ThreadCore S) (pc : U32) (running : S) :
    machine_control.apply_flow eq state pc .Stay running = .ok state := rfl

theorem apply_flow_jump {S : Type} (eq : core.cmp.PartialEq S S)
    (state : machine_control.ThreadCore S) (pc target : U32) (running : S) :
    machine_control.apply_flow eq state pc (.Jump target) running =
      .ok {state with pc := target} := rfl

end MemoryWrite.MachineControl
