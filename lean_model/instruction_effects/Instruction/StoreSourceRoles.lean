/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.StoreWindowModel
import Instruction.ReadRefinement

open Aeneas Aeneas.Std PtxvIr
set_option autoImplicit false
namespace Instruction

def storeControlReads (operation : Op) : List Key := (controlReads operation).getD []

def deferredStoreData (front : Front) (operation : Op) : List Key :=
  ((storeWindows front operation).map ByteWindow.source).filter
    (fun reg => decide (reg ∉ storeControlReads operation))

def narrowStoreDescriptor (operation : Op) : Option (Key × Nat) :=
  match operation with
  | .St _ bytes _ (.Reg source _) _ =>
      if bytes.val = 1 then some (key source, 255)
      else if bytes.val = 2 then some (key source, 65535) else none
  | _ => none

def narrowStoreSink (operation : Op) : Option (Key × Nat) :=
  match narrowStoreDescriptor operation with
  | none => none
  | some (source, mask) =>
      if source.file = .V ∨ source ∈ storeControlReads operation then none else some (source, mask)

theorem deferred_store_data_excludes_control (front : Front) (operation : Op) (reg : Key)
    (hcontrol : reg ∈ storeControlReads operation) : reg ∉ deferredStoreData front operation := by
  intro hdeferred
  exact (List.mem_filter.mp hdeferred).2 |> of_decide_eq_true |> fun h => h hcontrol

theorem narrow_store_sink_excludes_control (operation : Op) (source : Key) (mask : Nat)
    (hnarrow : narrowStoreSink operation = some (source, mask)) :
    source ∉ storeControlReads operation := by
  unfold narrowStoreSink at hnarrow
  cases hd : narrowStoreDescriptor operation with
  | none => simp [hd] at hnarrow
  | some descriptor =>
    rcases descriptor with ⟨reg, bits⟩
    simp only [hd] at hnarrow
    split at hnarrow
    · contradiction
    · rename_i hcontrol
      cases hnarrow
      exact fun h => hcontrol (Or.inr h)

def isUndefinedSink (front : Front) (operation : Op) (reg : Key) : Prop :=
  reg ∈ (memoryReads operation).getD [] ∧ reg ∉ deferredStoreData front operation

theorem control_reads_are_undefined_sinks (front : Front) (operation : Op) (reg : Key)
    (hcontrol : reg ∈ storeControlReads operation) : isUndefinedSink front operation reg := by
  refine ⟨?_, deferred_store_data_excludes_control front operation reg hcontrol⟩
  cases operation <;> simp_all [storeControlReads, controlReads, memoryReads, List.mem_append] <;> tauto

theorem layout_count_bound (first : Reg) (elements bytes : U8) (exchange : Bool) :
    layoutCount first.file elements bytes exchange ≤ 255 := by
  cases exchange
  · simpa [layoutCount, storeValueReads] using
      storeValueReads_length false bytes (.Reg first elements)
  · simpa [layoutCount, operandReads] using operandReads_length (.Reg first elements)

theorem store_windows_length_bound (front : Front) (operation : Op) :
    (storeWindows front operation).length ≤ 255 := by
  unfold storeWindows
  cases hd : storeDescriptor operation with
  | none => simp
  | some descriptor =>
    rcases descriptor with ⟨first, elements, bytes, exchange⟩
    have hb := layout_count_bound first elements bytes exchange
    by_cases hf : first.file = .V
    · cases front with
      | Ptx => simpa [hf, layoutWindows] using hb
      | Sass =>
        simp only [hf, ite_true]
        split
        · simp only [repeatedWindows, List.length_map, List.length_range]
          scalar_tac
        · simp
    · simpa [hf, layoutWindows] using hb

end Instruction
