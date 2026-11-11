/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureFormatsCore

namespace GB300Math.MultisampleNarrowFormats

def lane (channels format component value : Nat) : Nat :=
  if channels = 1 ∧ component > 0 then
    if component = 3 then 0x3f800000 else 0
  else
    TextureFormatsCore.evaluate format 0 value

theorem selected_channel (channels format component value : Nat)
    (h : ¬ (channels = 1 ∧ component > 0)) :
    lane channels format component value = TextureFormatsCore.evaluate format 0 value := by
  simp [lane,h]

theorem half_payload : lane 4 4 2 0xfc01 = 0xff802000 := by decide +kernel

theorem signed_minimum : lane 4 2 0 0x80 = 0xbf800000 := by decide +kernel

end GB300Math.MultisampleNarrowFormats
