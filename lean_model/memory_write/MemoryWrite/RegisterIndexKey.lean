/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Code.Funs

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.RegisterIndex

def keyPage (key : symbolic.RegisterKey) : Nat :=
  key.thread.val * 262144 + key.file.val * 1024 + key.index.val / 64

def keyOffset (key : symbolic.RegisterKey) : Nat := key.index.val % 64

theorem page_offset_injective (left right : symbolic.RegisterKey)
    (samePage : keyPage left = keyPage right) (sameOffset : keyOffset left = keyOffset right) :
    left = right := by
  have leftFile : left.file.val < 256 := by scalar_tac
  have rightFile : right.file.val < 256 := by scalar_tac
  have leftIndex : left.index.val < 65536 := by scalar_tac
  have rightIndex : right.index.val < 65536 := by scalar_tac
  dsimp only [keyPage] at samePage
  dsimp only [keyOffset] at sameOffset
  have threadEqual : left.thread.val = right.thread.val := by omega
  have fileEqual : left.file.val = right.file.val := by omega
  have indexEqual : left.index.val = right.index.val := by omega
  have thread := UScalar.eq_of_val_eq threadEqual
  have file := UScalar.eq_of_val_eq fileEqual
  have index := UScalar.eq_of_val_eq indexEqual
  cases left
  cases right
  simp only [symbolic.RegisterKey.mk.injEq]
  exact ⟨thread, file, index⟩

@[step]
theorem rust_page_key (key : symbolic.RegisterKey) :
    register_index.page_key key ⦃ result => result.val = keyPage key ⦄ := by
  unfold register_index.page_key
  simp only [lift, bind_ok]
  step as ⟨threadPart, threadValue⟩
  step as ⟨filePart, fileValue⟩
  step as ⟨prefixPart, prefixValue⟩
  step as ⟨page, pageValue⟩
  step as ⟨result, resultValue⟩
  dsimp only [keyPage]
  scalar_tac

end MemoryWrite.RegisterIndex
