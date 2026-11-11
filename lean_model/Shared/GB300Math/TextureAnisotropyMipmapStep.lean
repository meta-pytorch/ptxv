/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureAnisotropyMipmapWeights

namespace GB300Math.TextureAnisotropyMipmapStep

set_option maxRecDepth 1000000
set_option maxHeartbeats 0

def table : Array Nat := #[
  64, 64, 64, 64, 64, 65, 65, 65, 65, 65, 66, 66, 66, 66, 66, 66,
  67, 67, 67, 67, 67, 67, 68, 68, 68, 68, 68, 69, 69, 69, 69, 69,
  70, 70, 70, 70, 70, 70, 71, 71, 71, 71, 71, 72, 72, 72, 72, 72,
  73, 73, 73, 73, 73, 74, 74, 74, 74, 74, 75, 75, 75, 75, 75, 76,
  76, 76, 76, 76, 77, 77, 77, 77, 78, 78, 78, 78, 78, 79, 79, 79,
  79, 79, 80, 80, 80, 80, 81, 81, 81, 81, 81, 82, 82, 82, 82, 83,
  83, 83, 83, 83, 84, 84, 84, 84, 85, 85, 85, 85, 85, 86, 86, 86,
  86, 87, 87, 87, 87, 88, 88, 88, 88, 89, 89, 89, 89, 90, 90, 90,
  90, 91, 91, 91, 91, 91, 92, 92, 92, 92, 93, 93, 93, 94, 94, 94,
  94, 95, 95, 95, 95, 96, 96, 96, 96, 97, 97, 97, 97, 98, 98, 98,
  98, 99, 99, 99, 100, 100, 100, 100, 101, 101, 101, 101, 102, 102, 102, 103,
  103, 103, 103, 104, 104, 104, 105, 105, 105, 105, 106, 106, 106, 107, 107, 107,
  107, 108, 108, 108, 109, 109, 109, 109, 110, 110, 110, 111, 111, 111, 112, 112,
  112, 112, 113, 113, 113, 114, 114, 114, 115, 115, 115, 116, 116, 116, 116, 117,
  117, 117, 118, 118, 118, 119, 119, 119, 120, 120, 120, 121, 121, 121, 122, 122,
  122, 123, 123, 123, 124, 124, 124, 125, 125, 125, 126, 126, 126, 127, 127, 127,
  128]

def step (fraction : Nat) : Nat := table[fraction]!

theorem exact_root_brackets : ∀ f : Fin 257,
    (4*step f.val-1)^256 ≤ 256^256*2^f.val ∧
    256^256*2^f.val < (4*step f.val+3)^256 := by decide +kernel

theorem step_bounds : ∀ f : Fin 257, 64≤step f.val ∧ step f.val≤128 := by
  decide +kernel

theorem retained_bucket_equivalence : ∀ k : Fin 256,
    step (TextureGradientAxis.logTable[k.val]!) =
      TextureAnisotropyMipmapWeights.halfStep (256+k.val) := by decide +kernel

#print axioms exact_root_brackets
#print axioms step_bounds
#print axioms retained_bucket_equivalence

end GB300Math.TextureAnisotropyMipmapStep
