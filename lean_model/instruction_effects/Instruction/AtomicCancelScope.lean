/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.AtomicCancelCaller
open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace AtomicCancel

theorem source_private_file (file : File) :
    File.Insts.CoreCmpPartialEqFile.eq file File.V = .ok true ↔ file = File.V := by
  cases file <;>
    simp [File.Insts.CoreCmpPartialEqFile.eq, read_discriminant, File.read_discriminant]

theorem source_private_value (value : Val)
    (accepted : atomic_cancel.private_value value = .ok true) :
    ∀ register count, value = .Reg register count → register.file = .V := by
  intro register count shape
  subst value
  simpa only [atomic_cancel.private_value, source_private_file] using accepted

theorem source_scalar_payload (value : Val)
    (accepted : atomic_cancel.scalar_payload value = .ok true) :
    ∀ register count, value = .Reg register count → register.file = .V ∧ count = 1#u8 := by
  intro register count shape
  subst value
  unfold atomic_cancel.scalar_payload at accepted
  obtain ⟨privateFile, hprivate, accepted⟩ := bind_success _ _ _ |>.mp accepted
  cases privateFile with
  | false => simp only [Bool.false_eq_true, ite_false, Result.ok.injEq] at accepted
  | true =>
    simp only [if_pos rfl, ite_true, Result.ok.injEq, decide_eq_true_eq] at accepted
    exact ⟨source_private_file register.file |>.mp hprivate, accepted⟩
end AtomicCancel
