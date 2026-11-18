/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.StoreWindowRefinement

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 8000000
namespace Instruction

@[step] theorem store_descriptor_spec (operation : Op) :
    store_window.descriptor operation ⦃ result => result = storeDescriptor operation ⦄ := by
  cases operation <;> try (solve | simp [store_window.descriptor, storeDescriptor])
  case St space bytes address value sync =>
    cases value <;> cases space <;> simp [store_window.descriptor, storeDescriptor]
  case Atom operation space bytes address value destination sync predicate replacement =>
    cases operation <;> cases value <;> cases space <;>
      simp [store_window.descriptor, storeDescriptor]

@[step] theorem store_stride_spec (file : File) (bytes elements : U8) (exchange : Bool) :
    store_window.stride file bytes elements exchange ⦃ result =>
      result.val = layoutStride file elements bytes exchange ∧ result.val ≤ 255 ⦄ := by
  have heq : elements ≥ 2#u8 ↔ elements.val ≥ 2 := by scalar_tac
  have hb8eq : bytes > 8#u8 ↔ bytes.val > 8 := by scalar_tac
  cases file <;> cases exchange <;> by_cases he : elements.val ≥ 2 <;>
    by_cases hb : bytes.val > 8
  all_goals simp [store_window.stride, core.cmp.PartialEq.ne.trait_default,
    File.Insts.CoreCmpPartialEqFile.eq, File.read_discriminant, heq, hb8eq,
    he, hb, layoutStride, lift, bind_ok]
  all_goals first
    | scalar_tac
    | (constructor <;> scalar_tac)
    | (step as ⟨quotient, hquotient⟩ <;> scalar_tac)

@[step] theorem assemble_store_windows_spec (first : Reg) (elements bytes : U8) (exchange : Bool)
    (hindex : Bounded (consecutive first (layoutCount first.file elements bytes exchange))) :
    store_window.assemble first elements bytes exchange ⦃ result =>
      result.val.map byteWindow = layoutWindows first elements bytes exchange ⦄ := by
  have hcount : (if exchange then register_effect.scalar_count first.file elements
      else register_effect.store_count first.file bytes elements false) ⦃ result =>
        result.val = layoutCount first.file elements bytes exchange ∧ result.val ≤ 255 ⦄ := by
    cases exchange with
    | false => simpa [layoutCount] using store_count_spec first.file bytes elements false
    | true =>
      simp only [ite_true]
      step with scalar_count_spec first.file elements as ⟨count, hcount, hbound⟩
      simp [layoutCount, hcount]
      scalar_tac
  unfold store_window.assemble
  step with hcount as ⟨count, hcountval, hcountbound⟩
  step as ⟨stride, hstride, hstridebound⟩
  simp only [lift, bind_ok]
  step with append_windows_spec first count stride (UScalar.cast .U32 bytes)
    (by rw [hcountval]; exact consecutive_bounded first _ hindex) hcountbound hstridebound
    as ⟨registers, hmap, hlength⟩
  simpa [layoutWindows, hcountval, hstride] using hmap

@[step] theorem store_windows_spec (front : Front) (operation : Op)
    (hindex : windowsBounded (storeWindows front operation)) :
    store_window.sources front operation ⦃ result =>
      result.val.map byteWindow = storeWindows front operation ⦄ := by
  unfold store_window.sources
  step with store_descriptor_spec operation as ⟨descriptor, hdescriptor⟩
  rw [hdescriptor]
  cases hd : storeDescriptor operation with
  | none => simp [hd, storeWindows, empty_window_vector]
  | some descriptor =>
    rcases descriptor with ⟨first, elements, bytes, exchange⟩
    simp only [hd]
    step as ⟨virtual, hvirtual⟩
    rw [hvirtual]
    simp only [decide_eq_true_eq]
    split
    · rename_i hf
      cases front with
      | Sass =>
        cases exchange with
        | true => simp [storeWindows, hd, hf, empty_window_vector]
        | false =>
          simp only [Bool.false_eq_true, ite_false]
          step with repeated_virtual_spec .Sass first.file elements bytes as ⟨repeated, hrepeated⟩
          have hrepeat : repeated = decide (elements.val = 1 ∧ bytes.val > 8) := by
            simpa [hf] using hrepeated
          rw [hrepeat]
          by_cases he : elements.val = 1 <;> by_cases hb : bytes.val > 8
          all_goals simp only [he, hb, and_true, and_false, decide_true, decide_false,
            Bool.true_and, Bool.false_and, Bool.false_eq_true, Bool.true_eq_false,
            ite_true, ite_false]
          all_goals simp [storeWindows, hd, hf, he, hb, empty_window_vector]
          all_goals exact repeated_windows_spec first bytes
      | Ptx =>
        have hbounded : Bounded (consecutive first (layoutCount first.file elements bytes exchange)) := by
          simpa [windowsBounded, storeWindows, hd, hf, layoutWindows, source_window_keys] using hindex
        step with assemble_store_windows_spec first elements bytes exchange hbounded as ⟨registers, hmap⟩
        simpa [storeWindows, hd, hf] using hmap
    · rename_i hf
      have hbounded : Bounded (consecutive first (layoutCount first.file elements bytes exchange)) := by
        simpa [windowsBounded, storeWindows, hd, hf, layoutWindows, source_window_keys] using hindex
      step with assemble_store_windows_spec first elements bytes exchange hbounded as ⟨registers, hmap⟩
      simpa [storeWindows, hd, hf] using hmap


end Instruction
