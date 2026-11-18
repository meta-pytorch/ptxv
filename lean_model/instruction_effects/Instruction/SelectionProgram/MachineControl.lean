/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionProgram.Code.Funs

open Aeneas Aeneas.Std Result PtxvSim

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 4000000

namespace Instruction.MachineControl

deriving instance DecidableEq for machine.St

abbrev ThreadState := ptxv_memory_write.machine_control.ThreadCore machine.St
abbrev Flow := ptxv_memory_write.machine_control.Flow

def nextPC (state : ThreadState) (pc : U32) (flow : Flow) : Result U32 :=
  match flow with
  | .Next => match state.st with
    | .Run => pc + 1#u32
    | _ => .ok state.pc
  | .Stay => .ok state.pc
  | .Jump target => .ok target

theorem source_status_equality (left right : machine.St) :
    machine.St.Insts.CoreCmpPartialEqSt.eq left right = .ok (decide (left = right)) := by
  cases left <;> cases right <;>
    simp [machine.St.Insts.CoreCmpPartialEqSt.eq, machine.St.read_discriminant]

theorem source_apply_thread_flow (state : ThreadState) (pc : U32) (flow : Flow) :
    machine.apply_thread_flow state pc flow =
      (do let next ← nextPC state pc flow; .ok {state with pc := next}) := by
  cases state with
  | mk current status registers acquire =>
    cases flow <;> cases status <;>
      simp [machine.apply_thread_flow, ptxv_memory_write.machine_control.apply_flow,
      source_status_equality, nextPC]

@[step] theorem source_next_running (state : ThreadState) (pc : U32)
    (running : state.st = .Run) (room : pc.val + 1 ≤ 4294967295) :
    machine.apply_thread_flow state pc .Next ⦃ next =>
      next.pc.val = pc.val + 1 ∧ next.st = .Run ∧
      next.registers = state.registers ∧ next.acq_bar = state.acq_bar ⦄ := by
  rw [source_apply_thread_flow]
  simp only [nextPC, running]
  step as ⟨next, value⟩
  simp [value]

theorem source_next_blocked (state : ThreadState) (pc : U32)
    (blocked : state.st ≠ .Run) :
    machine.apply_thread_flow state pc .Next = .ok state := by
  rw [source_apply_thread_flow]
  cases state with
  | mk current status registers acquire =>
    cases status <;> simp_all [nextPC]

theorem source_stay (state : ThreadState) (pc : U32) :
    machine.apply_thread_flow state pc .Stay = .ok state := by
  rw [source_apply_thread_flow]
  simp [nextPC]

theorem source_jump (state : ThreadState) (pc target : U32) :
    machine.apply_thread_flow state pc (.Jump target) = .ok {state with pc := target} := by
  rw [source_apply_thread_flow]
  simp [nextPC]

@[step] theorem source_flow_frame (state : ThreadState) (pc : U32) (flow : Flow)
    (room : pc.val + 1 ≤ 4294967295) :
    machine.apply_thread_flow state pc flow ⦃ next =>
      next.st = state.st ∧ next.registers = state.registers ∧
      next.acq_bar = state.acq_bar ⦄ := by
  cases flow with
  | Next =>
    by_cases running : state.st = .Run
    · step with source_next_running state pc running room as ⟨next, pcValue, status, registers, acquire⟩
      exact ⟨status.trans running.symm, registers, acquire⟩
    · rw [source_next_blocked state pc running]
      simp
  | Stay =>
    rw [source_stay]
    simp
  | Jump target =>
    rw [source_jump]
    simp

end Instruction.MachineControl
