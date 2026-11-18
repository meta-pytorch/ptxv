/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.Core

open Aeneas Aeneas.Std Result PtxvMemorySafety

set_option autoImplicit false
set_option maxHeartbeats 3000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

abbrev ExecutionMachine := machine_memory_safety.ExecutionMachine
abbrev StrictMachine := machine_memory_safety.StrictMachine
abbrev RustResult (α β : Type) := core.result.Result α β

abbrev StepDictionary (L F : Type) :=
  core.ops.function.FnOnce F (L × Usize) ((RustResult Event Fault) × L)

abbrev EnabledDictionary (L F : Type) :=
  core.ops.function.FnOnce F (L × Usize) Bool

abbrev OutputDictionary (L F : Type) :=
  core.ops.function.FnOnce F L (RustResult (alloc.vec.Vec U8) Fault)

inductive NativeReachable {L : Type} (initial : ExecutionMachine L) : ExecutionMachine L → Prop where
  | initial : NativeReachable initial initial
  | step {before after : ExecutionMachine L} {event : Event} {actor : Usize} {F E : Type}
      (stepDictionary : StepDictionary L F) (stepCallback : F)
      (enabledDictionary : EnabledDictionary L E) (enabledCallback : E) :
      NativeReachable initial before →
      machine_memory_safety.ExecutionMachine.enabled enabledDictionary before actor enabledCallback = .ok true →
      machine_memory_safety.ExecutionMachine.step stepDictionary before actor stepCallback = .ok (.Ok event, after) →
      NativeReachable initial after

end MemorySafetyExecution.Native
