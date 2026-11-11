/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import TextureCubeSeamless
import GB300Math.TextureSeamless

namespace GB300Math.TextureSeamlessBridge

theorem weights_match (x y : Nat) :
    TextureSeamless.weights x y = TextureCubeSeamless.weights x y := by rfl

theorem encoder_matches (value scale : Int) :
    TextureSeamless.encode value scale = TextureCubeSeamless.encode value scale := by rfl

theorem accumulator_matches (weights geometric values : Array Nat) :
    TextureSeamless.weighted weights geometric values =
      TextureCubeSeamless.interpolate weights geometric values := by rfl

end GB300Math.TextureSeamlessBridge
#print axioms GB300Math.TextureSeamlessBridge.weights_match
#print axioms GB300Math.TextureSeamlessBridge.encoder_matches
#print axioms GB300Math.TextureSeamlessBridge.accumulator_matches
