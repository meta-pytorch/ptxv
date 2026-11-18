/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MachineRegisters
import MemoryWrite.EffectsFold

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 4000000

namespace MemoryWrite.MachineRegisters

def uniformEvent (warp : U32) : effects.Effect :=
  .Access { space := .UniformRegisters, owner := warp, start := 0#u64, «end» := 1#u64 }
    .Write .All

def uniformAccess (warp : U32) : machine_recorder.Access :=
  ⟨{ space := .UniformRegisters, owner := warp, start := 0#u64, «end» := 1#u64 }, .Write, .All⟩

def accessEvent (event : machine_recorder.Access) : effects.Effect :=
  .Access event.resource event.kind event.visibility

theorem source_uniform_access (warp : U32) :
    machine_registers.uniform_access warp = .ok (uniformAccess warp) := by
  simp [machine_registers.uniform_access, effects.uniform_register, uniformAccess]

theorem source_uniform_effect (warp : U32) :
    machine_registers.uniform_effect warp = .ok (uniformEvent warp) := by
  simp [machine_registers.uniform_effect, source_uniform_access, uniformAccess, uniformEvent]

theorem uniform_access_event (warp : U32) : accessEvent (uniformAccess warp) = uniformEvent warp := rfl

def predicateBit (old : U8) (index : U16) (enabled : Bool) : U8 :=
  let bit : BitVec 8 := (1 : BitVec 8) <<< index.val
  ⟨(old.bv &&& ~~~bit) ||| if enabled then bit else 0⟩

theorem source_write_uniform_predicate
    (registers : machine_registers.WarpRegisters) (warp : U32)
    (write : machine_registers.PredicateWrite) (valid : write.index.val < 8) :
    machine_registers.write_uniform_predicate registers warp write ⦃ emitted updated =>
      emitted = some (uniformAccess warp) ∧
      updated = { registers with
        up := predicateBit registers.up write.index write.value,
        up_def := predicateBit registers.up_def write.index write.defined } ⦄ := by
  unfold machine_registers.write_uniform_predicate
  have tested : write.index < 8#u16 := by scalar_tac
  simp only [tested, massert_True, bind_ok]
  step as ⟨bit, bitValue, bitBits⟩
  have bitAt : bit = (⟨(1 : BitVec 8) <<< write.index.val⟩ : U8) :=
    (UScalar.eq_equiv_bv_eq _ _).2 bitBits
  cases hValue : write.value <;> cases hDefined : write.defined
  all_goals
    simp only [predicateBit, Bool.false_eq_true, ↓reduceIte,
      lift, bind_ok, source_uniform_access, WP.spec_ok]
    exact ⟨rfl, by rw [bitAt]; rfl⟩

theorem source_poison_uniform_predicate
    (registers : machine_registers.WarpRegisters) (warp : U32) (index : U16)
    (valid : index.val < 8) :
    machine_registers.poison_uniform_predicate registers warp index ⦃ emitted updated =>
      emitted = some (uniformAccess warp) ∧
      updated = { registers with up_def := predicateBit registers.up_def index false } ⦄ := by
  unfold machine_registers.poison_uniform_predicate
  have tested : index < 8#u16 := by scalar_tac
  simp only [tested, massert_True, bind_ok]
  step as ⟨bit, bitValue, bitBits⟩
  have bitAt : bit = (⟨(1 : BitVec 8) <<< index.val⟩ : U8) :=
    (UScalar.eq_equiv_bv_eq _ _).2 bitBits
  simp only [predicateBit, Bool.false_eq_true, ↓reduceIte, lift, bind_ok,
    source_uniform_access, WP.spec_ok]
  refine ⟨rfl, ?_⟩
  rw [bitAt]
  congr 1
  apply (UScalar.eq_equiv_bv_eq _ _).2
  simp only [HAnd.hAnd, HOr.hOr, Complement.complement, UScalar.and, UScalar.not]
  exact BitVec.or_zero.symm

theorem source_uniform_constant_frame
    (registers : machine_registers.WarpRegisters) (warp : U32)
    (write : machine_registers.WordWrite) (constant : 255 ≤ write.index.val) :
    machine_registers.write_uniform registers warp write = .ok (none, registers) := by
  have tested : write.index ≥ 255#u16 := by scalar_tac
  simp [machine_registers.write_uniform, tested]

theorem source_uniform_poison_constant_frame
    (registers : machine_registers.WarpRegisters) (warp : U32)
    (index : U16) (constant : 255 ≤ index.val) :
    machine_registers.poison_uniform registers warp index = .ok (none, registers) := by
  have tested : index ≥ 255#u16 := by scalar_tac
  simp [machine_registers.poison_uniform, tested]

theorem uniform_included (mode : effects.FoldMode) :
    effects.included mode .All .UniformRegisters = .ok true := by
  cases mode <;>
    simp [effects.included, effects.FoldMode.Insts.CoreCmpPartialEqFoldMode.eq,
      effects.Visibility.Insts.CoreCmpPartialEqVisibility.eq,
      effects.FoldMode.read_discriminant, effects.Visibility.read_discriminant]

theorem uniform_not_poisoning :
    effects.poisons_arrivals .UniformRegisters .Write = .ok false := by
  simp [effects.poisons_arrivals,
    effects.ResourceSpace.Insts.CoreCmpPartialEqResourceSpace.eq,
    effects.ResourceSpace.read_discriminant]

theorem source_uniform_apply (state : effects.FoldState) (warp : U32)
    (room : state.summary.writes.val.length < Usize.max) :
    effects.apply state (uniformEvent warp) ⦃ updated =>
      updated.mode = state.mode ∧ updated.invalidated = state.invalidated ∧
      updated.arrivals = state.arrivals ∧ updated.summary.all = state.summary.all ∧
      updated.summary.reads = state.summary.reads ∧
      updated.summary.releases = state.summary.releases ∧
      updated.summary.acquires = state.summary.acquires ∧
      updated.summary.writes.val = state.summary.writes.val ++
        [(17#u8, warp, 0#u64, 1#u64)] ⦄ := by
  simp only [uniformEvent, effects.apply, uniform_not_poisoning, uniform_included,
    bind_ok, ite_true, ActionEffects.source_resource_range, ActionEffects.resourceTag]
  step with alloc.vec.Vec.push_spec state.summary.writes (17#u8, warp, 0#u64, 1#u64)
    room as ⟨writes, appended⟩
  simpa using appended

theorem source_uniform_after_prefix (state : effects.FoldState)
    (prior recorded : Slice effects.Effect) (warp : U32)
    (events : recorded.val = prior.val ++ [uniformEvent warp]) :
    effects.fold_from state recorded = (do
      let before ← effects.fold_from state prior
      effects.apply before (uniformEvent warp)) := by
  simp [ActionEffects.source_fold_sequence, events,
    ActionEffects.sequence_append, ActionEffects.sequence]

theorem source_uniform_predicate_transition
    (registers : machine_registers.WarpRegisters) (warp : U32)
    (write : machine_registers.PredicateWrite) (valid : write.index.val < 8) :
    machine_registers.write_uniform_predicate registers warp write ⦃ emitted updated =>
      updated = { registers with
        up := predicateBit registers.up write.index write.value,
        up_def := predicateBit registers.up_def write.index write.defined } ∧
      emitted.map accessEvent = some (uniformEvent warp) ⦄ := by
  step with source_write_uniform_predicate registers warp write valid as
    ⟨emitted, updated, event, mutation⟩
  exact ⟨mutation, by simp [event, uniform_access_event]⟩

end MemoryWrite.MachineRegisters
