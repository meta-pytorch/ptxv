/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.EffectsNamespaces

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false

namespace MemoryWrite.ActionEffects

theorem source_uniform_register (warp : U32) :
    effects.uniform_register warp = .ok
      { space := .UniformRegisters, owner := warp, start := 0#u64, «end» := 1#u64 } := by
  rfl

theorem source_uniform_register_range (warp : U32) :
    (do let resource ← effects.uniform_register warp
        effects.resource_range resource) = .ok (17#u8, warp, 0#u64, 1#u64) := by
  simp [source_uniform_register, source_resource_range, resourceTag]

theorem source_convergence_barrier (warp : U32) (barrier : U8) :
    effects.convergence_barrier warp barrier ⦃ resource =>
      resource.space = .ConvergenceBarrier ∧ resource.owner = warp ∧
      resource.start.val = barrier.val ∧ resource.end.val = barrier.val + 1 ⦄ := by
  unfold effects.convergence_barrier
  simp only [lift, bind_ok]
  step
  · scalar_tac

theorem actual_uniform_convergence_disjoint (warp : U32) (barrier : U8) :
    effects.convergence_barrier warp barrier ⦃ resource =>
      resourceTag resource.space ≠ resourceTag effects.ResourceSpace.UniformRegisters ⦄ := by
  apply WP.spec_mono (source_convergence_barrier warp barrier)
  intro resource h
  simp [h.1, resourceTag]

theorem source_tag_injective (left right : effects.ResourceSpace)
    (same : effects.resource_tag left = effects.resource_tag right) : left = right := by
  rw [source_resource_tag, source_resource_tag] at same
  exact resource_tag_injective left right (Result.ok_injective same)

end MemoryWrite.ActionEffects
