/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.RustClockCover

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite.SparseClock

@[step] theorem rust_base_covers_linear_loop (left right : Slice (U32 × U32))
    (covered : Bool) (i j : Usize)
    (hi : i.val ≤ left.val.length) (hj : j.val ≤ right.val.length) :
    clock.base_covers_linear_loop left right covered i j ⦃ result =>
      result = (covered && scanCovers (baseModel (left.val.drop i.val))
        (baseModel (right.val.drop j.val))) ⦄ := by
  let expected := covered && scanCovers (baseModel (left.val.drop i.val))
    (baseModel (right.val.drop j.val))
  change clock.base_covers_linear_loop left right covered i j ⦃ result => result = expected ⦄
  unfold clock.base_covers_linear_loop
  apply loop.spec_decr_nat
    (fun current : Bool × Usize × Usize =>
      if current.1 then (left.val.length - current.2.1.val) +
        (right.val.length - current.2.2.val) + 1 else 0)
    (fun current : Bool × Usize × Usize =>
      current.2.1.val ≤ left.val.length ∧ current.2.2.val ≤ right.val.length ∧
      (current.1 && scanCovers (baseModel (left.val.drop current.2.1.val))
        (baseModel (right.val.drop current.2.2.val))) = expected)
  · intro current hc
    rcases current with ⟨active, x, y⟩
    obtain ⟨hx, hy, he⟩ := hc
    change x.val ≤ left.val.length at hx
    change y.val ≤ right.val.length at hy
    change (active && scanCovers (baseModel (left.val.drop x.val))
      (baseModel (right.val.drop y.val))) = expected at he
    unfold clock.base_covers_linear_loop.body
    cases active with
    | false => simpa using he
    | true =>
      simp only [Bool.true_and] at he
      simp only [ite_true]
      by_cases hyr : y.val < right.val.length
      · have hyt : y < Slice.len right := by scalar_tac
        simp only [hyt, ite_true]
        by_cases hxl : x.val < left.val.length
        · have hxt : x < Slice.len left := by scalar_tac
          simp only [hxt, ite_true]
          step with Slice.index_usize_spec left x hxl as ⟨a, ha⟩
          step with Slice.index_usize_spec right y hyr as ⟨b, hb⟩
          rcases a with ⟨ak, av⟩
          rcases b with ⟨bk, bv⟩
          change (if ak < bk then do
              let nx ← x + 1#usize
              pure (ControlFlow.cont (true, nx, y))
            else do
              let known ← (if ak = bk then pure (av >= bv) else pure false : Result Bool)
              let nx ← x + 1#usize
              let ny ← y + 1#usize
              pure (ControlFlow.cont (known, nx, ny))) ⦃ _ ⦄
          have hdl : left.val.drop x.val = (ak,av) :: left.val.drop (x.val+1) := by
            rw [List.drop_eq_getElem_cons hxl, ha]
          have hdr : right.val.drop y.val = (bk,bv) :: right.val.drop (y.val+1) := by
            rw [List.drop_eq_getElem_cons hyr, hb]
          by_cases hlt : ak < bk
          · simp only [hlt, ite_true]
            step as ⟨next, hn⟩; try scalar_tac
            have hnext : next.val = x.val + 1 := by scalar_tac
            have hltv : ak.val < bk.val := by scalar_tac
            refine ⟨by omega, hy, ?_, ?_⟩
            · simpa [hnext, hdl, hdr, baseModel, entryModel, scanCovers, hltv] using he
            · simp only [ite_true]; omega
          · simp only [hlt, ite_false]
            by_cases heq : ak = bk
            · simp only [heq, ite_true, pure_tc_eq, bind_ok]
              step as ⟨nx, hnx⟩; try scalar_tac
              step as ⟨ny, hny⟩; try scalar_tac
              have hnxv : nx.val = x.val+1 := by scalar_tac
              have hnyv : ny.val = y.val+1 := by scalar_tac
              have hnot : ¬ak.val < bk.val := by scalar_tac
              have heqv : ak.val = bk.val := by scalar_tac
              have hv : decide (av >= bv) = decide (bv.val ≤ av.val) := by scalar_tac
              refine ⟨by omega, by omega, ?_, ?_⟩
              · simpa [hnxv, hnyv, hv, hdl, hdr, baseModel, entryModel,
                  scanCovers, hnot, heqv] using he
              · split <;> omega
            · simp only [heq, ite_false, pure_tc_eq, bind_ok]
              step as ⟨nx, hnx⟩; try scalar_tac
              step as ⟨ny, hny⟩; try scalar_tac
              have hnot : ¬ak.val < bk.val := by scalar_tac
              have hneq : ak.val ≠ bk.val := by scalar_tac
              refine ⟨by scalar_tac, by scalar_tac, ?_, ?_⟩
              · simpa [hdl, hdr, baseModel, entryModel, scanCovers, hnot, hneq] using he
              · simp only [Bool.false_eq_true, ite_false]; omega
        · have hxt : ¬x < Slice.len left := by scalar_tac
          simp only [hxt, ite_false, WP.spec_ok]
          have hxe : x.val = left.val.length := by omega
          have hfalse : expected = false := by
            rw [←he, hxe, List.drop_length]
            change scanCovers [] (baseModel (right.val.drop y.val)) = false
            rw [List.drop_eq_getElem_cons hyr]
            simp only [baseModel, List.map_cons, scanCovers]
          exact ⟨⟨hx, hy, by simp [hfalse]⟩, by simp⟩
      · have hyt : ¬y < Slice.len right := by scalar_tac
        simp only [hyt, ite_false, WP.spec_ok]
        have hye : y.val = right.val.length := by omega
        simpa [hye, baseModel, scanCovers] using he
  · exact ⟨hi, hj, rfl⟩

@[step] theorem rust_base_covers_linear (left right : Slice (U32 × U32)) :
    clock.base_covers_linear left right ⦃ result =>
      result = scanCovers (baseModel left.val) (baseModel right.val) ⦄ := by
  unfold clock.base_covers_linear
  step with rust_base_covers_linear_loop left right true 0#usize 0#usize
    (by simp) (by simp) as ⟨result, hr⟩
  simpa using hr

@[step] theorem rust_base_covers (left right : Slice (U32 × U32))
    (hl : Sorted (baseModel left.val)) (hr : Sorted (baseModel right.val)) :
    clock.base_covers left right ⦃ result =>
      result = baseCovers (baseModel left.val) (baseModel right.val) ⦄ := by
  unfold clock.base_covers
  step as ⟨threshold, hthreshold⟩; try scalar_tac
  split
  · exact rust_base_covers_sparse left right hl
  · step with rust_base_covers_linear left right as ⟨result, hresult⟩
    rw [hresult, scan_covers_eq _ _ hl hr]

end MemoryWrite.SparseClock
