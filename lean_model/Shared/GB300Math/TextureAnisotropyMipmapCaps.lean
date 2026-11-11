/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureAnisotropyMipmap

namespace GB300Math.TextureAnisotropyMipmapCaps
open TextureAnisotropyMipmap
set_option maxHeartbeats 0
set_option maxRecDepth 100000

theorem descriptor_classes :
    ((List.range 16).map fun i => cap (i+1)) =
      [0,256,256,512,512,656,656,768,768,848,848,912,912,912,912,1024] := by
  decide +kernel

theorem base_tap_properties : ∀ a : Fin 16,
    let (groups,taps) := TextureAnisotropyTaps.taps (cap (a.val+1)) false
    taps.length≤a.val+1 ∧ (taps.map TextureAnisotropyTaps.Tap.weight).sum=groups*256 ∧
      (groups=1 ∨ groups=2 ∨ groups=4) := by decide +kernel

theorem effective_cap_brackets : ∀ a : Fin 16,
    (2*(a.val+1))^256 ≤ 3^256*2^(cap (a.val+1)) ∧
    2^(cap (a.val+1)) ≤ (a.val+1)^256 := by decide +kernel

theorem cap3_loss_attained : cap 3=256 ∧ 2^256*3^256=(2*3)^256 := by decide +kernel

#print axioms descriptor_classes
#print axioms base_tap_properties
#print axioms effective_cap_brackets
#print axioms cap3_loss_attained
end GB300Math.TextureAnisotropyMipmapCaps
