/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.MemoryRefinement

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace Instruction

@[step] theorem file_ne (left right : File) :
    core.cmp.PartialEq.ne.trait_default File.Insts.CoreCmpPartialEqFile left right
      ⦃ result => result = decide (left ≠ right) ⦄ := by
  cases left <;> cases right <;>
    simp [core.cmp.PartialEq.ne.trait_default, core.cmp.PartialEq.ne.default, File.Insts.CoreCmpPartialEqFile,
      File.Insts.CoreCmpPartialEqFile.eq, File.read_discriminant]

@[step] theorem cas_count_spec (file : File) (bytes : U8) :
    register_effect.cas_count file bytes ⦃ result =>
      result.val = (if bytes.val = 8 ∧ file ≠ .V then 2 else 1) ∧ result.val ≤ 2 ⦄ := by
  unfold register_effect.cas_count
  split
  · rename_i hb
    have hbval : bytes.val = 8 := by scalar_tac
    step as ⟨different, hdifferent⟩
    rw [hdifferent]
    simp only [decide_eq_true_eq]
    split <;> simp_all
  · rename_i hb
    have hbval : bytes.val ≠ 8 := by scalar_tac
    simp [hbval]

@[step] theorem floating_sources_spec (initial : alloc.vec.Vec Reg) (value : Val) (bytes : U8)
    (hindex : Bounded (atomicValueReads .FAddFtz bytes value))
    (hsize : initial.val.length + (atomicValueReads .FAddFtz bytes value).length ≤ Usize.max) :
    register_effect.floating_sources initial value bytes ⦃ result =>
      result.val.map key = initial.val.map key ++ atomicValueReads .FAddFtz bytes value ∧
      result.val.length = initial.val.length + (atomicValueReads .FAddFtz bytes value).length ⦄ := by
  unfold register_effect.floating_sources
  split
  · rename_i hb
    have hbval : bytes.val > 4 := by scalar_tac
    cases value with
    | Reg first _ =>
      simp only [atomicValueReads, hbval, ite_true] at hindex hsize ⊢
      step as ⟨words, hwords⟩
      simp only [lift, bind_ok]
      step with append_spec initial first (UScalar.cast .U16 words)
        (by have h := consecutive_bounded first _ hindex; scalar_tac)
        (by have hs := hsize; simp only [consecutive_length] at hs; scalar_tac)
        as ⟨result, hmap, hlength⟩
      constructor
      · convert hmap using 1 <;> congr 1 <;> scalar_tac
      · convert hlength using 1 <;> simp only [consecutive_length] <;> scalar_tac
    | Imm _ | Const _ _ | SReg _ => simp [register_effect.slots, atomicValueReads, operandReads]
  · rename_i hb
    have hbval : ¬bytes.val > 4 := by scalar_tac
    step with slots_spec initial value
      (by cases value <;> simpa [atomicValueReads, hbval] using hindex)
      (by cases value <;> simpa [atomicValueReads, hbval] using hsize)
      as ⟨result, hmap, hlength⟩
    cases value <;> simpa [atomicValueReads, hbval] using And.intro hmap hlength

@[step] theorem implicit_cas_spec (initial : alloc.vec.Vec Reg) (bytes : U8) (compare : Val)
    (hindex : Bounded (casReplacement bytes compare none))
    (hsize : initial.val.length + (casReplacement bytes compare none).length ≤ Usize.max) :
    register_effect.implicit_cas_sources initial bytes compare ⦃ result =>
      result.val.map key = initial.val.map key ++ casReplacement bytes compare none ∧
      result.val.length = initial.val.length + (casReplacement bytes compare none).length ⦄ := by
  unfold register_effect.implicit_cas_sources
  cases compare with
  | Reg first slots =>
    step as ⟨count, hcount, hsmall⟩
    let width := if bytes.val = 8 ∧ first.file ≠ .V then 2 else 1
    have hwidth : count.val = width := hcount
    have hpos : 0 < width := by dsimp only [width]; split <;> omega
    have hbound : first.idx.val + width + width ≤ 65536 := by
      apply offset_sequence_bound first.file (first.idx.val + width) width hpos
      simpa only [casReplacement, width] using hindex
    step as ⟨next, hnext⟩; try scalar_tac
    have hnextval : next.val = first.idx.val + width := by scalar_tac
    have hspan : consecutive { first with idx := next } count.val =
        casReplacement bytes (.Reg first slots) none := by
      simp only [consecutive, casReplacement, hwidth, width, hnextval]
    step with append_spec initial { first with idx := next } count
      (by dsimp only; scalar_tac)
      (by have hs := hsize; rw [←hspan, consecutive_length] at hs; exact hs)
      as ⟨result, hmap, hlength⟩
    constructor
    · simpa only [hspan] using hmap
    · have hlen : (casReplacement bytes (.Reg first slots) none).length = count.val := by
        rw [←hspan, consecutive_length]
      simpa only [hlen] using hlength
  | Imm _ | Const _ _ | SReg _ => simp [casReplacement]

end Instruction
