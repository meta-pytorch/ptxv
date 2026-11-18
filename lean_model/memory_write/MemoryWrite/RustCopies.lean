/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RustCapture
import MemoryWrite.RustEngine

open Aeneas Aeneas.Std PtxvMemoryWrite

set_option autoImplicit false

namespace MemoryWrite

/-- An actual Rust source capture followed by an actual checked engine write, with derived metadata soundness. -/
theorem rust_copy_progress_and_preserves_representation {Input : Type}
    (sem : Semantics Input) (issueMemory completionMemory : Memory Input)
    (source : Slice Cell) (start : Key) (width : Usize)
    (before : Storage) (state : engine.Engine) (template : checked.PreparedWrite)
    (hsource : Represents sem issueMemory source)
    (hsourceSpan : ValidWriteSpan source start width.val)
    (hstate : Storage.Represents sem completionMemory before)
    (hcells : state.cells.slice = before.toCells)
    (hnumeric : state.numeric_written.val.length + width.val ≤ Usize.max)
    (hhalf : state.half_written.val.length + width.val ≤ Usize.max)
    (haddress : template.address_dependent = false)
    (hguard : template.guard_dependent = false) (henabled : template.enabled = true)
    (hfamily : (rustFamily template.class).allows template.destination.space = true)
    (hdestination : Storage.SpanFits before template.destination width) :
    ∃ payload next,
      checked.capture_span source start width = .ok (some payload) ∧
      engine.apply state { template with payload := payload, words := alloc.vec.Vec.new _ } = .ok (.Ok (), next) ∧
      RustEngineRefines before state { template with payload := payload, words := alloc.vec.Vec.new _ } (.Ok (), next) ∧
      Represents sem (storeBytes completionMemory template.destination width.val
        (snapshotValues issueMemory start)) next.cells.slice := by
  obtain ⟨payload, hcapture, hlength, hpayload⟩ :=
    rust_capture_span_sound sem issueMemory source start width hsource hsourceSpan
  have hwidth : alloc.vec.Vec.len payload = width := by
    apply UScalar.eq_of_val_eq
    simpa using hlength
  let write := { template with payload := payload, words := alloc.vec.Vec.new _ }
  obtain ⟨next, hrun, hrefines⟩ := rust_engine_apply_progress before state write hcells
    (by simpa [write, hlength] using hnumeric) (by simpa [write, hlength] using hhalf)
    haddress hguard henabled hfamily (by simpa [write, hwidth] using hdestination)
  refine ⟨payload, next, hcapture, hrun, hrefines, ?_⟩
  obtain ⟨after, hmodel, hafter⟩ := hrefines.1
  dsimp only [Prod.fst, Prod.snd] at hafter
  have hderef : alloc.vec.Vec.deref payload = payload.slice := by
    apply Slice.ext
    simp [alloc.vec.Vec.deref, alloc.vec.Vec.val]
  have hsound : EffectSound sem completionMemory (rustEffect write)
      (snapshotValues issueMemory start) := by
    refine ⟨?_, by simp [rustEffect, write, alloc.vec.Vec.new]⟩
    simpa only [rustEffect, write, hderef] using hpayload
  have hrep := (applyEffect_preserves_representation sem completionMemory before after
    (rustEffect write) (snapshotValues issueMemory start) hstate hsound hmodel).1
  rw [hafter]
  simpa [Storage.Represents, effectMemory, rustEffect, write, henabled,
    alloc.vec.Vec.deref, hlength] using hrep

end MemoryWrite
