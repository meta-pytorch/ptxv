/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureGradientAxis

/- This candidate fails the retained non-power-of-two anisotropic hardware corpus.
Its theorems describe the proposed function, not established sampler semantics. -/
namespace GB300Math.TextureAnisotropyAxis

def effective (a b logAniso : Nat) : Nat :=
  max (min a b) (max a b - logAniso * 2^23)

def sample (a b logAniso : Nat) : Nat :=
  TextureGradientAxis.sample (effective a b logAniso)

theorem lod_bounded (a b logAniso : Nat) :
    TextureGradientAxis.position (effective a b logAniso) ≤ 2048 :=
  TextureGradientAxis.position_bounded _

theorem isotropic (a b : Nat) : effective a b 0 = max a b := by
  unfold effective
  simp
  omega

end GB300Math.TextureAnisotropyAxis
