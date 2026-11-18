/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.EffectsConstructors

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 4000000

namespace MemoryWrite.MachineWarp

def maskAccess (warp : U32) (barrier : U8) : machine_recorder.Access :=
  { resource :=
      { space := .ConvergenceBarrier, owner := warp,
        start := ⟨barrier.val, by scalar_tac⟩,
        «end» := ⟨barrier.val + 1, by scalar_tac⟩ },
    kind := .Write, visibility := .All }

def maskEvent (warp : U32) (barrier : U8) : effects.Effect :=
  let access := maskAccess warp barrier
  .Access access.resource access.kind access.visibility

def maskValue (old : U32) (op : machine_warp.MaskOp) : U32 :=
  match op with
  | .Insert lanes => old ||| lanes
  | .Replace lanes => lanes
  | .Remove lanes => old &&& ~~~lanes

@[step] theorem rust_mask_access (warp : U32) (barrier : U8) :
    machine_warp.mask_access warp barrier ⦃ access => access = maskAccess warp barrier ⦄ := by
  unfold machine_warp.mask_access
  step with MemoryWrite.ActionEffects.source_convergence_barrier warp barrier as ⟨resource, spaceEq, ownerEq, low, high⟩
  rcases resource with ⟨space, owner, start, finish⟩
  change space = .ConvergenceBarrier at spaceEq
  change owner = warp at ownerEq
  change start.val = barrier.val at low
  change finish.val = barrier.val + 1 at high
  subst space owner
  simp only [maskAccess]
  congr <;> apply UScalar.eq_of_val_eq <;> assumption

@[step] theorem rust_mask_effect (warp : U32) (barrier : U8) :
    machine_warp.mask_effect warp barrier ⦃ event => event = maskEvent warp barrier ⦄ := by
  unfold machine_warp.mask_effect
  step as ⟨access, haccess⟩
  simp [haccess, maskEvent]

@[step] theorem rust_write_mask (state : machine_warp.WarpCore)
    (write : machine_warp.MaskWrite) (bounded : write.barrier.val < 16) :
    machine_warp.write_mask state write ⦃ event updated =>
      event = some (maskAccess write.warp write.barrier) ∧
      updated.registers = state.registers ∧ updated.cp_seq = state.cp_seq ∧
      updated.bmask.val = state.bmask.val.set write.barrier.val
        (maskValue state.bmask.val[write.barrier.val]! write.op) ⦄ := by
  unfold machine_warp.write_mask
  simp only [lift, bind_ok]
  generalize hindex : core.convert.num.FromUsizeU8.from write.barrier = index
  have indexValue : index.val = write.barrier.val := by scalar_tac
  have bound : index.val < state.bmask.length := by
    simpa [Array.length_eq, indexValue] using bounded
  cases op : write.op with
  | Insert lanes =>
      simp only
      step with Array.index_usize_spec state.bmask index bound as ⟨old, hold⟩
      step with Array.update_spec state.bmask index _ bound as ⟨masks, hmasks⟩
      step as ⟨access, haccess⟩
      simp_all [maskValue]
  | Replace lanes =>
      simp only
      step with Array.update_spec state.bmask index lanes bound as ⟨masks, hmasks⟩
      step as ⟨access, haccess⟩
      simp_all [maskValue]
  | Remove lanes =>
      simp only
      step with Array.index_usize_spec state.bmask index bound as ⟨old, hold⟩
      step with Array.update_spec state.bmask index _ bound as ⟨masks, hmasks⟩
      step as ⟨access, haccess⟩
      simp_all [maskValue]

@[step] theorem rust_commit_async (state : machine_warp.WarpCore)
    (room : state.cp_seq.val < U32.max) :
    machine_warp.commit_async state ⦃ updated =>
      updated.cp_seq.val = state.cp_seq.val + 1 ∧
      updated.registers = state.registers ∧ updated.bmask = state.bmask ⦄ := by
  unfold machine_warp.commit_async
  step
  scalar_tac

theorem actual_uniform_mask_events_are_disjoint (uniformWarp barrierWarp : U32) (barrier : U8) :
    machine_registers.uniform_effect uniformWarp ⦃ uniform =>
      uniform ≠ maskEvent barrierWarp barrier ⦄ := by
  simp [machine_registers.uniform_effect, machine_registers.uniform_access, effects.uniform_register,
    maskEvent, maskAccess]

end MemoryWrite.MachineWarp
