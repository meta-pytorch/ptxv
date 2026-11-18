/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.InputState

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

theorem input_running_flags (shadow : value_flow_dependencies.State) (left right : CoreState)
    (related : StateInputsRelated shadow left right) :
    left.threads.val.map execution_core.ThreadState.running=right.threads.val.map execution_core.ThreadState.running := by
  apply List.ext_getElem?
  intro index
  simp only [List.getElem?_map]
  by_cases inside : index<left.threads.val.length
  · have rightInside : index<right.threads.val.length := by have := related.threadsLength; omega
    have shadowInside : index<shadow.threads.val.length := by have := related.threadMetadataLength; omega
    have leftAt := List.getElem?_eq_getElem inside
    have rightAt := List.getElem?_eq_getElem rightInside
    have shadowAt := List.getElem?_eq_getElem shadowInside
    have threads := related.threads index left.threads.val[index] right.threads.val[index] shadow.threads.val[index]
      leftAt rightAt shadowAt
    simpa only [leftAt,rightAt,Option.map_some] using congrArg some threads.running
  · have rightOutside : right.threads.val.length ≤ index := by have := related.threadsLength; omega
    have noLeft : left.threads.val[index]?=none := List.getElem?_eq_none (by omega)
    have noRight : right.threads.val[index]?=none := List.getElem?_eq_none rightOutside
    simp only [noLeft,noRight,Option.map_none]

theorem input_terminal_equal (shadow : value_flow_dependencies.State) (left right : CoreState)
    (related : StateInputsRelated shadow left right) : execution_core.terminal left=execution_core.terminal right := by
  rw [source_terminal,source_terminal]
  have flags := input_running_flags shadow left right related
  have both := congrArg (fun values:List Bool => values.all (fun value=>!value)) flags
  simpa only [List.all_map,Function.comp_def] using congrArg Result.ok both

end MemorySafetyExecution.Native
