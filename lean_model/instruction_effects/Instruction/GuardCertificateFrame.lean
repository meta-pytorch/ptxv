/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GuardCertificateSpan
import Instruction.WriteRefinement

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 6000000
namespace Instruction.GuardCertificate

def fallbackFrame : Op → Option (List Key)
  | .Mov destination operand => some (consecutive destination (moveCount destination operand))
  | .Setp destination _ _ _ _ _ _ _ none => some (consecutive destination 1)
  | .Nop | .Bra .. | .Brx .. | .Exit | .Trap .. | .WarpSync .. | .Bssy .. | .Bsync ..
  | .Break .. | .BarSync .. | .ClusterArrive .. | .ClusterWait | .Fence .. | .MbarInit ..
  | .MbarArrive .. | .MbarInval .. | .TmaLoad .. | .TmaStore .. | .BulkCommit | .BulkWait ..
  | .CpAsync .. | .CpAsyncCommit | .CpAsyncWait .. | .CpAsyncMbarArrive ..
  | .TmemWaitLd | .TmemWaitSt => some []
  | _ => none

def frameWrites (operation : Op) : Option (List Key) :=
  match memoryWrites operation with
  | some writes => some writes
  | none => fallbackFrame operation

def FrameFits (operation : Op) : Prop :=
  optionBounded (frameWrites operation) ∧
    ((frameWrites operation).getD []).length ≤ Usize.max

theorem memory_frame_bounded (operation : Op) (bounded : optionBounded (frameWrites operation)) :
    optionBounded (memoryWrites operation) := by
  cases present : memoryWrites operation with
  | none => trivial
  | some writes => simpa only [frameWrites, present] using bounded

@[step]
theorem source_frame (operation : Op) (fits : FrameFits operation) :
    guard_certificate.frame operation
      ⦃ result => registerObservation result = frameWrites operation ⦄ := by
  unfold guard_certificate.frame
  step with memory_writes_spec operation (memory_frame_bounded operation fits.1)
    as ⟨selected, selectedAt⟩
  cases selected with
  | some writes =>
    simp only [registerObservation] at selectedAt
    simp only [WP.spec_ok, frameWrites, ←selectedAt, registerObservation, Option.map_some]
  | none =>
    have absent : memoryWrites operation = none := by simpa [registerObservation] using selectedAt.symm
    have bounded : optionBounded (fallbackFrame operation) := by
      simpa only [frameWrites, absent] using fits.1
    have sized : ((fallbackFrame operation).getD []).length ≤ Usize.max := by
      simpa only [frameWrites, absent] using fits.2
    cases operation <;> try (solve | simp [fallbackFrame, frameWrites, absent, registerObservation])
    case Mov destination operand =>
      step with move_count_spec destination operand as ⟨count, countAt⟩
      step with span_spec destination count
        (by rw [countAt]; exact consecutive_bounded destination _ bounded)
        (by simpa [fallbackFrame, countAt] using sized)
        as ⟨result, meaning, _⟩
      simp only [registerObservation, frameWrites, absent, fallbackFrame, Option.map_some, Option.some.injEq]
      simpa only [countAt] using meaning
    case Setp destination comparison bits left right and or xor extra =>
      cases extra with
      | some _ => simp [fallbackFrame, frameWrites, absent, registerObservation]
      | none =>
        step with span_spec destination 1#u16
          (consecutive_bounded destination _ bounded)
          (by simpa [fallbackFrame] using sized)
          as ⟨result, meaning, _⟩
        simpa only [registerObservation, frameWrites, absent, fallbackFrame, Option.map_some, Option.some.injEq] using meaning

end Instruction.GuardCertificate
