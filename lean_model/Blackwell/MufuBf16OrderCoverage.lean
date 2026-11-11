/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MufuBf16Order

namespace GB300Math.MufuBf16Order

set_option maxRecDepth 100000

theorem finite_domain_covers (word : Nat) (hw : word < 65536)
    (hf : magnitude word < 32640) :
    ∃ index < 65280, code "finite" index = word := by
  have hm : word % 32768 < 32640 := hf
  by_cases hs : word < 32768
  · refine ⟨word+32640, by omega, ?_⟩
    change (if word+32640 < 32640 then 65407-(word+32640)
      else word+32640-32640) = word
    split <;> omega
  · refine ⟨65407-word, by omega, ?_⟩
    change (if 65407-word < 32640 then 65407-(65407-word)
      else 65407-word-32640) = word
    split <;> omega

theorem block_partition (index count : Nat) (h : index < count) :
    index / 256 * 256 < count ∧ index % 256 < 256 ∧
    index / 256 * 256 + index % 256 = index := by omega

#print axioms finite_domain_covers
#print axioms block_partition

end GB300Math.MufuBf16Order
