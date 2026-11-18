/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RustEngine

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false

namespace MemoryWrite

theorem rust_accepted_write_requires_independent_control (cells next : Slice Cell)
    (write : checked.PreparedWrite)
    (hrun : checked.checked_write cells write = .ok (.Ok (), next)) :
    write.address_dependent = false ∧ write.guard_dependent = false := by
  cases ha : write.address_dependent with
  | true => simp [checked.checked_write, ha] at hrun
  | false =>
    cases hg : write.guard_dependent with
    | true => simp [checked.checked_write, ha, hg] at hrun
    | false => exact ⟨rfl, rfl⟩

theorem rust_captured_false_guard_still_rejected (cells : Slice Cell)
    (write : checked.PreparedWrite) (haddress : write.address_dependent = false)
    (hguard : write.guard_dependent = true) :
    checked.checked_write cells write = .ok (.Err .InputDependentGuard, cells) := by
  simp [checked.checked_write, haddress, hguard]

theorem rust_engine_accepted_write_requires_independent_control
    (state next : engine.Engine) (write : checked.PreparedWrite)
    (hrun : engine.apply state write = .ok (.Ok (), next)) :
    write.address_dependent = false ∧ write.guard_dependent = false := by
  cases ha : write.address_dependent with
  | true => simp [engine.apply, checked.checked_write, alloc.vec.Vec.deref_mut, ha, lift, bind_ok]
      at hrun
  | false =>
    cases hg : write.guard_dependent with
    | true => simp [engine.apply, checked.checked_write, alloc.vec.Vec.deref_mut, ha, hg, lift,
        bind_ok] at hrun
    | false => exact ⟨rfl, rfl⟩

end MemoryWrite
