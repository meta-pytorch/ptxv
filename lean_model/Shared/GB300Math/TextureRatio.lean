/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.TextureRatio

def ratioBits (n d : Nat) : Nat := Id.run do
  if n = 0 then return 0
  let tentative := (n.log2 : Int) - (d.log2 : Int)
  let below := if tentative ≥ 0 then n < d*2^tentative.toNat
    else n*2^(-tentative).toNat < d
  let e := if below then tentative-1 else tentative
  let scaled := n*2^(23-e).toNat
  let q := scaled/d
  let rem := scaled%d
  let m := q + if 2*rem > d ∨ (2*rem = d ∧ q%2 = 1) then 1 else 0
  let (m,e) := if m ≥ 2^24 then (m/2,e+1) else (m,e)
  return (e+127).toNat*2^23 + m%2^23

end GB300Math.TextureRatio
