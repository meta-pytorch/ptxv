/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureDepthPCF

namespace GB300Math.VulkanDepthFilter

open TextureDepthPCF

def fetch (depths : Array Nat) (mode function reference : Nat) (x y : Int) : Bool :=
  let depth := match index x mode, index y mode with
    | some a, some b => depths[b*16+a]!
    | _, _ => 0
  TextureDepthCompare.compare function reference depth

def sample (depths : Array Nat) (x y mode function reference kind : Nat) (dx dy : Int) : Array Nat :=
  let f := fetch depths mode function reference
  if kind = 0 then
    let px := position x mode
    let py := position y mode
    #[encode (blend (px%256).toNat (py%256).toNat
      (f (px/256) (py/256)) (f (px/256+1) (py/256))
      (f (px/256) (py/256+1)) (f (px/256+1) (py/256+1))), 0, 0, 0x3f800000]
  else
    let px := coordinate x / 256 + dx
    let py := coordinate y / 256 + dy
    #[f px (py+1), f (px+1) (py+1), f (px+1) py, f px py].map
      fun b => if b then 0x3f800000 else 0

theorem border_zero (depths : Array Nat) (function reference : Nat) :
    fetch depths 3 function reference (-1) 0 =
      TextureDepthCompare.compare function reference 0 := by
  simp [fetch, index]

theorem output_size (depths : Array Nat) (x y mode function reference kind : Nat) (dx dy : Int) :
    (sample depths x y mode function reference kind dx dy).size = 4 := by
  simp [sample]; split <;> simp

end GB300Math.VulkanDepthFilter
