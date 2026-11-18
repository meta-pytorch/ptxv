/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MachineRecorderRefinement

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 100000

namespace MemoryWrite.MachineRecorder

theorem source_finish_without_footprint (state : machine_recorder.Recorder) (forceWrite : Bool) :
    machine_recorder.finish state ⟨none⟩ forceWrite = .ok (none, ⟨none⟩) := by
  cases state with
  | mk trace => cases trace <;> rfl

theorem source_finish_idle (context : machine_recorder.FinishContext) (forceWrite : Bool) :
    machine_recorder.finish ⟨none⟩ context forceWrite = .ok (none, ⟨none⟩) := rfl

theorem source_finish (state : machine_recorder.Recorder)
    (context : machine_recorder.FinishContext) (forceWrite : Bool) (room : TraceRoom state) :
    (do
      let (summary, next) ← machine_recorder.finish state context forceWrite
      .ok (summary, stateModel next)) =
        MachineRecorderModel.finishAction (stateModel state) (finishContextModel context) forceWrite := by
  cases context with
  | mk footprint =>
    cases footprint with
    | none =>
      cases state with
      | mk trace =>
        rw [source_finish_without_footprint]
        cases trace <;> simp [stateModel, finishContextModel,
          MachineRecorderModel.finishAction, MachineRecorderModel.FinishContext.lower,
          MachineRecorderModel.finish, MachineRecorderModel.empty]
    | some footprint =>
      cases footprint with
      | mk mode membership =>
        cases state with
        | mk trace =>
          cases trace with
          | none =>
            rw [source_finish_idle]
            simp [stateModel, finishContextModel, MachineRecorderModel.finishAction,
              MachineRecorderModel.FinishContext.lower, MachineRecorderModel.finish,
              MachineRecorderModel.empty]
          | some trace =>
            have capacity : trace.val.length + 1 ≤ Usize.max := by
              have bounded := room trace rfl
              omega
            let updated : alloc.vec.Vec effects.Effect := alloc.vec.Vec.from
              (trace.val ++ [.Finding]) (by simpa using capacity)
            have pushed : alloc.vec.Vec.push trace .Finding = .ok updated := by
              simp [alloc.vec.Vec.push, capacity, updated]
            have appended : (alloc.vec.Vec.deref updated).val = trace.val ++ [.Finding] := by
              simp [updated, alloc.vec.Vec.deref, alloc.vec.Vec.val, alloc.vec.Vec.from, Slice.from_val]
            cases forceWrite <;> cases membership
            · simp only [machine_recorder.finish, core.mem.replace, Bool.false_eq_true,
                ↓reduceIte, bind_ok]
              conv_lhs =>
                arg 1
                change (do
                  let summary ← effects.fold mode (alloc.vec.Vec.deref trace)
                  .ok (some summary, (⟨none⟩ : machine_recorder.Recorder)))
              simp only [Std.bind_assoc, bind_ok]
              change (do
                let summary ← effects.fold mode (alloc.vec.Vec.deref trace)
                .ok (some summary, MachineRecorderModel.empty)) = (do
                let summary ← MachineRecorderModel.summarize mode trace.val
                .ok (some summary, MachineRecorderModel.empty))
              rw [MachineRecorderModel.summarize_source]
              simp only [alloc.vec.Vec.deref, alloc.vec.Vec.val, Slice.from_val]
              rfl
            all_goals
              simp only [machine_recorder.finish, core.mem.replace, Bool.false_eq_true,
                ↓reduceIte]
              conv_lhs =>
                arg 1
                change (do
                  let payload ← (do
                    let events ← alloc.vec.Vec.push trace effects.Effect.Finding
                    .ok ((none : Option (alloc.vec.Vec effects.Effect)), events,
                      ({ mode := mode, has_membership := _ } : machine_recorder.FinishFootprint)))
                  let summary ← effects.fold payload.2.2.mode (alloc.vec.Vec.deref payload.2.1)
                  .ok (some summary, ({ trace := payload.1 } : machine_recorder.Recorder)))
                simp only [pushed, bind_ok]
              simp only [Std.bind_assoc, bind_ok]
              change (do
                let summary ← effects.fold mode (alloc.vec.Vec.deref updated)
                .ok (some summary, MachineRecorderModel.empty)) = (do
                let summary ← MachineRecorderModel.summarize mode (trace.val ++ [.Finding])
                .ok (some summary, MachineRecorderModel.empty))
              rw [MachineRecorderModel.summarize_source]
              rw [appended]
              rfl

end MemoryWrite.MachineRecorder
