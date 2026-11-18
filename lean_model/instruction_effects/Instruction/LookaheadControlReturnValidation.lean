/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.Code.Funs
import Instruction.LookaheadControlFalseGuard

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace LookaheadControl

private theorem contains_usize (targets : Slice Usize) (target : Usize) :
    core.slice.Slice.contains core.cmp.PartialEqUsize targets target =
      .ok (decide (target ∈ targets.val)) := by
  unfold core.slice.Slice.contains
  have eqfun : core.cmp.PartialEqUsize.eq target =
      (fun other => pure (decide (target = other))) := by
    funext other
    rfl
  rw [eqfun, List.anyM_pure]
  congr 1
  apply Bool.eq_iff_iff.mpr
  simp only [List.any_eq_true, decide_eq_true_eq]
  constructor
  · rintro ⟨other, present, rfl⟩
    exact present
  · intro present
    exact ⟨target, present, rfl⟩

/-- This concrete source case is also the empty-table regression witness. -/
theorem source_empty_target_rejected :
    control_transfer.return_target_allowed
      (some (Slice.from ([] : List Usize) (by simp))) 0#usize = .ok false := by
  rfl

/-- A present return table validates membership even when it is empty. -/
theorem source_return_target_allowed (targets : Slice Usize) (target : Usize) :
    control_transfer.return_target_allowed (some targets) target =
      .ok (decide (target ∈ targets.val)) := by
  exact contains_usize targets target

/-- No global return table preserves the existing unconstrained-target behavior. -/
theorem source_return_target_unrestricted (target : Usize) :
    control_transfer.return_target_allowed none target = .ok true := by
  rfl

/-- Checked address decoding is explicit; the table validation itself is total. -/
theorem source_known_return_validates {R : Type}
    (readerInst : control_transfer.ControlReader R)
    (context : control_transfer.Context) (addr : Val) (reader afterRead : R)
    (address shifted : U64) (target : Usize) (targets : Slice Usize)
    (valueRead : readerInst.value reader addr = .ok (.Known address, afterRead))
    (table : context.returns = some targets)
    (shift : address >>> 4#i32 = .ok shifted)
    (offset : (UScalar.cast .Usize shifted + context.entry : Result Usize) = .ok target) :
    control_transfer.taken_operation readerInst context (.Ret addr) reader =
      .ok (some (if target ∈ targets.val then
        .Jump (UScalar.cast .U32 target) else .Wild .InvalidTarget), afterRead) := by
  simp only [control_transfer.taken_operation, valueRead, bind_ok, lift,
    table, source_return_target_allowed]
  split <;> simp_all

/-- In particular, an empty per-PC target set cannot accept a known RET. -/
theorem source_known_return_empty_rejected {R : Type}
    (readerInst : control_transfer.ControlReader R)
    (context : control_transfer.Context) (addr : Val) (reader afterRead : R)
    (address shifted : U64) (target : Usize) (targets : Slice Usize)
    (valueRead : readerInst.value reader addr = .ok (.Known address, afterRead))
    (table : context.returns = some targets) (empty : targets.val = [])
    (shift : address >>> 4#i32 = .ok shifted)
    (offset : (UScalar.cast .Usize shifted + context.entry : Result Usize) = .ok target) :
    control_transfer.taken_operation readerInst context (.Ret addr) reader =
      .ok (some (.Wild .InvalidTarget), afterRead) := by
  simpa [empty] using source_known_return_validates readerInst context addr reader
    afterRead address shifted target targets valueRead table shift offset

/-- An unknown RET does not invent a fork when the present target set is empty. -/
theorem source_unknown_return_empty_rejected (targets : Slice Usize)
    (empty : targets.val = []) :
    control_transfer.unknown_return (some targets) = .ok (.Wild .UnknownReturn) := by
  simp [control_transfer.unknown_return, core.slice.Slice.is_empty, Slice.length,
    empty, bind_ok]

/-- A successful known jump cannot bypass a present table's membership check. -/
theorem source_known_return_jump_mem {R : Type}
    (readerInst : control_transfer.ControlReader R)
    (context : control_transfer.Context) (addr : Val) (reader afterRead : R)
    (address shifted : U64) (target : Usize) (targets : Slice Usize) (jump : U32)
    (valueRead : readerInst.value reader addr = .ok (.Known address, afterRead))
    (table : context.returns = some targets)
    (shift : address >>> 4#i32 = .ok shifted)
    (offset : (UScalar.cast .Usize shifted + context.entry : Result Usize) = .ok target)
    (outcome : control_transfer.taken_operation readerInst context (.Ret addr) reader =
      .ok (some (.Jump jump), afterRead)) : target ∈ targets.val := by
  rw [source_known_return_validates readerInst context addr reader afterRead address
    shifted target targets valueRead table shift offset] at outcome
  split at outcome <;> simp_all

/-- The production dispatcher carries empty-table rejection through a true guard. -/
theorem source_guarded_known_return_empty_rejected {R : Type}
    (readerInst : control_transfer.ControlReader R)
    (context : control_transfer.Context) (predicate : Pred) (addr : Val)
    (reader afterGuard afterRead : R) (address shifted : U64)
    (target : Usize) (targets : Slice Usize)
    (predicateRead : readerInst.predicate reader predicate = .ok (.Known 1#u64, afterGuard))
    (valueRead : readerInst.value afterGuard addr = .ok (.Known address, afterRead))
    (table : context.returns = some targets) (empty : targets.val = [])
    (shift : address >>> 4#i32 = .ok shifted)
    (offset : (UScalar.cast .Usize shifted + context.entry : Result Usize) = .ok target) :
    control_transfer.from_operation readerInst context predicate (.Ret addr) reader =
      .ok (some (.Wild .InvalidTarget), afterRead) := by
  have taken := source_known_return_empty_rejected readerInst context addr afterGuard
    afterRead address shifted target targets valueRead table empty shift offset
  have nonzero : (1#u64) ≠ (0#u64) := by decide
  simp [control_transfer.from_operation, predicateRead, taken, bind_ok]
  split
  · rename_i isZero
    exact False.elim (nonzero isZero)
  · rfl

end LookaheadControl
