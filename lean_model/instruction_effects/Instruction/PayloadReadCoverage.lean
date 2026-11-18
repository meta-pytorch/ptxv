/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.ExecutionCoverage
import Instruction.StoreWindowModel

open Aeneas Aeneas.Std PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace Instruction

theorem store_descriptor_read_coverage (operation : Op) (first : Reg) (elements bytes : U8)
    (exchange : Bool) (hd : storeDescriptor operation = some (first, elements, bytes, exchange))
    (reg : Key) (hreg : reg ∈ consecutive first (layoutCount first.file elements bytes exchange)) :
    reg ∈ (executionReads operation).getD [] := by
  cases operation <;> try (solve | simp [storeDescriptor] at hd)
  case St space actualBytes address value sync =>
    cases value <;> cases space <;> simp only [storeDescriptor] at hd
    all_goals try contradiction
    all_goals cases hd
    all_goals simpa [executionReads, layoutCount, storeValueReads] using
      List.mem_append_right (addressReads address) hreg
  case Atom atomicOp space actualBytes address value destination sync predicate replacement =>
    cases atomicOp <;> try (solve | simp [storeDescriptor] at hd)
    cases value <;> cases space <;> simp only [storeDescriptor] at hd
    all_goals try contradiction
    all_goals cases hd
    all_goals simpa [executionReads, executionAtomicValueReads, executionReplacementReads,
      layoutCount, operandReads] using List.mem_append_right (addressReads address) hreg

theorem store_window_read_coverage (front : Front) (operation : Op) (window : ByteWindow)
    (hwindow : window ∈ storeWindows front operation) :
    window.source ∈ (executionReads operation).getD [] := by
  have hkeys : window.source ∈ (storeWindows front operation).map ByteWindow.source :=
    List.mem_map.mpr ⟨window, hwindow, rfl⟩
  unfold storeWindows at hkeys
  cases hd : storeDescriptor operation with
  | none => simp [hd] at hkeys
  | some descriptor =>
    rcases descriptor with ⟨first, elements, bytes, exchange⟩
    have hselected : window.source ∈ consecutive first (layoutCount first.file elements bytes exchange) := by
      by_cases hf : first.file = .V
      · cases front with
        | Ptx => simpa [hd, hf, layoutWindows, source_window_keys] using hkeys
        | Sass =>
          simp only [hd, hf, ite_true] at hkeys
          split at hkeys
          · rcases List.mem_map.mp hkeys with ⟨selected, hselected, hsource⟩
            rcases List.mem_map.mp hselected with ⟨position, hposition, rfl⟩
            simp only [ByteWindow.source] at hsource
            rw [← hsource]
            have hcount : 0 < layoutCount first.file elements bytes exchange := by
              cases exchange <;> simp [layoutCount, storeSlots, scalarSlots, hf]
              all_goals split_ifs <;> omega
            rw [consecutive_mem]
            simp only [key, true_and, le_refl]
            omega
          · simp at hkeys
      · simpa [hd, hf, layoutWindows, source_window_keys] using hkeys
    exact store_descriptor_read_coverage operation first elements bytes exchange hd window.source hselected

theorem store_window_declared_read_coverage (front : Front) (operation : Op) (window : ByteWindow)
    (hwindow : window ∈ storeWindows front operation) :
    window.source ∈ (memoryReads operation).getD [] :=
  execution_reads_covered operation window.source (store_window_read_coverage front operation window hwindow)

end Instruction
