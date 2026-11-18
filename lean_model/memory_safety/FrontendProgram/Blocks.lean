/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.Scans

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety
open MemorySafety.Source

set_option autoImplicit false
set_option maxHeartbeats 12000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false
set_option backward.isDefEq.respectTransparency false
set_option backward.split false

open PtxFrontend

namespace PtxFrontendProgram

theorem no_closer_first (source : List U8) (index : Nat) (byte : U8)
    (read : source[index]? = some byte) (different : byte.val ≠ 42) : ¬CloserAt source index := by
  rintro ⟨first, second, firstRead, secondRead, firstValue, secondValue⟩
  rw [read] at firstRead
  have same := Option.some.inj firstRead
  subst first
  exact different firstValue

theorem no_closer_second (source : List U8) (index : Nat) (byte : U8)
    (read : source[index + 1]? = some byte) (different : byte.val ≠ 47) : ¬CloserAt source index := by
  rintro ⟨first, second, firstRead, secondRead, firstValue, secondValue⟩
  rw [read] at secondRead
  have same := Option.some.inj secondRead
  subst second
  exact different secondValue

theorem no_closer_last (source : List U8) (index : Nat)
    (outside : source.length ≤ index + 1) : ¬CloserAt source index := by
  rintro ⟨first, second, firstRead, secondRead, firstValue, secondValue⟩
  have absent : source[index + 1]? = none := by simp [outside]
  rw [absent] at secondRead
  cases secondRead

theorem block_prepend (source : List U8) (index finish : Nat)
    (notEnd : ¬CloserAt source index) (tail : BlockBody source (index + 1) finish) :
    BlockBody source index finish := by
  rcases tail with ⟨lower, upper, ending, previous⟩
  refine ⟨by omega, upper, ending, ?_⟩
  intro earlier after before
  by_cases same : earlier = index
  · simpa only [same] using notEnd
  · exact previous earlier (by omega) before

theorem source_usize_add_two (index : Usize) (bound : index.val + 2 < 2 ^ UScalarTy.Usize.numBits) :
    (index + 2#usize : Result Usize) = .ok (Usize.ofNatCore (index.val + 2) bound) := by
  have platformBound : index.val + 2 < 2 ^ System.Platform.numBits := bound
  simp [HAdd.hAdd, UScalar.add, UScalar.tryMk, UScalar.tryMkOpt,
    UScalar.check_bounds, platformBound, Result.ofOption]
  rfl

theorem source_scan_block_closed (source : Slice U8) (position : Usize) :
    strict_memory_frontend.scan_block_loop source position true = .ok (position, true) := by
  unfold strict_memory_frontend.scan_block_loop
  rw [loop.eq_def]
  simp only [strict_memory_frontend.scan_block_loop.body, ite_true, ite_self, bind_ok]

theorem source_scan_block_loop (source : Slice U8) (start finish : Usize)
    (bound : start.val ≤ source.val.length)
    (computed : strict_memory_frontend.scan_block_loop source start false = .ok (finish, true)) :
    BlockBody source.val start.val finish.val := by
  generalize fuelEq : source.val.length - start.val = fuel
  induction fuel using Nat.strong_induction_on generalizing start with
  | h fuel ih =>
    have unroll : strict_memory_frontend.scan_block_loop source start false = (do
        let result ← strict_memory_frontend.scan_block_loop.body source start false
        match result with
        | .cont pair => strict_memory_frontend.scan_block_loop source pair.1 pair.2
        | .done result => .ok result) := by
      unfold strict_memory_frontend.scan_block_loop
      rw [loop.eq_def]
      congr 1
      funext result
      cases result with
      | cont pair => cases pair; rfl
      | done result => rfl
    rw [unroll] at computed
    by_cases inside : start.val < source.val.length
    · have test : start < Slice.len source := by scalar_tac
      let byte := source.val[start.val]
      have selected : source.val[start.val]? = some byte := by
        simp only [List.getElem?_eq_getElem inside, byte]
      have read : Slice.index_usize source start = .ok byte := by
        simp [Slice.index_usize, List.getElem?_eq_getElem inside, byte]
      have incrementBound : start.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := Slice.length_ineq source
        scalar_tac
      let after : Usize := Usize.ofNatCore (start.val + 1) incrementBound
      have afterVal : after.val = start.val + 1 := by simp [after]
      have increment := source_usize_increment start incrementBound
      simp only [strict_memory_frontend.scan_block_loop.body, test, ite_true,
        Bool.false_eq_true, ite_false, read, bind_ok] at computed
      by_cases star : byte = 42#u8
      · simp only [star, ite_true, increment, bind_ok] at computed
        by_cases secondInside : after.val < source.val.length
        · have secondTest : after < Slice.len source := by scalar_tac
          let second := source.val[after.val]
          have secondRead : Slice.index_usize source after = .ok second := by
            simp [Slice.index_usize, List.getElem?_eq_getElem secondInside, second]
          have secondSelected : source.val[start.val + 1]? = some second := by
            simpa only [←afterVal, List.getElem?_eq_getElem secondInside] using
              (show some source.val[after.val] = some second from rfl)
          simp only [after] at secondTest secondRead
          simp only [secondTest, ite_true, secondRead, bind_ok] at computed
          by_cases slash : second = 47#u8
          · have advanceBound : start.val + 2 < 2 ^ UScalarTy.Usize.numBits := by
              have := Slice.length_ineq source
              scalar_tac
            let advance : Usize := Usize.ofNatCore (start.val + 2) advanceBound
            have advanceVal : advance.val = start.val + 2 := by simp [advance]
            have addTwo := source_usize_add_two start advanceBound
            simp only [slash, ite_true, addTwo, bind_ok, source_scan_block_closed,
              ok.injEq, Prod.mk.injEq, and_true] at computed
            subst finish
            change BlockBody source.val start.val advance.val
            refine ⟨by omega, by omega, ?_, ?_⟩
            · have endpoint : advance.val - 2 = start.val := by omega
              rw [endpoint]
              exact ⟨byte, second, selected, secondSelected, by scalar_tac, by scalar_tac⟩
            · intro earlier lower upper
              omega
          · simp only [slash, ite_false, bind_ok] at computed
            have tail := ih (source.val.length - after.val) (by omega) after (by omega) computed rfl
            have noEnd : ¬CloserAt source.val start.val :=
              no_closer_second source.val start.val second secondSelected (by scalar_tac)
            apply block_prepend source.val start.val finish.val noEnd
            simpa only [afterVal] using tail
        · have secondTest : ¬after < Slice.len source := by scalar_tac
          simp only [after] at secondTest
          simp only [secondTest, ite_false, bind_ok] at computed
          have tail := ih (source.val.length - after.val) (by omega) after (by omega) computed rfl
          have noEnd := no_closer_last source.val start.val (by omega)
          apply block_prepend source.val start.val finish.val noEnd
          simpa only [afterVal] using tail
      · simp only [star, ite_false, increment, bind_ok] at computed
        have tail := ih (source.val.length - after.val) (by omega) after (by omega) computed rfl
        have noEnd := no_closer_first source.val start.val byte selected (by scalar_tac)
        apply block_prepend source.val start.val finish.val noEnd
        simpa only [afterVal] using tail
    · have test : ¬start < Slice.len source := by scalar_tac
      simp only [strict_memory_frontend.scan_block_loop.body, test, ite_false, bind_ok,
        ok.injEq, Prod.mk.injEq, Bool.false_eq_true, and_false] at computed

theorem source_scan_block (source : Slice U8) (start finish : Usize)
    (bound : start.val ≤ source.val.length)
    (computed : strict_memory_frontend.scan_block source start = .ok (some finish)) :
    BlockBody source.val start.val finish.val := by
  unfold strict_memory_frontend.scan_block at computed
  obtain ⟨pair, searched, computed⟩ := source_bind_success _ _ _ computed
  rcases pair with ⟨position, closed⟩
  cases closed with
  | false => simp at computed
  | true =>
    change (ok (some position) : Result (Option Usize)) = ok (some finish) at computed
    simp only [ok.injEq, Option.some.injEq] at computed
    subst position
    exact source_scan_block_loop source start finish bound searched

end PtxFrontendProgram
