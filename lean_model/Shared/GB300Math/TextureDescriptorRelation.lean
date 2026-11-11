/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.TextureDescriptorRelation

def word (bytes : ByteArray) (index : Nat) : Nat :=
  (List.range 4).foldl (fun v i => v + bytes[index*4+i]!.toNat * 2^(8*i)) 0

def direct (alpha : Option Nat) (lane value : Nat) : Nat :=
  if lane % 4 = 3 then alpha.getD value else value

theorem rgb_preserved (alpha : Option Nat) (lane value : Nat) (h : lane % 4 ≠ 3) :
    direct alpha lane value = value := by simp [direct, h]
theorem alpha_replaced (alpha lane value : Nat) (h : lane % 4 = 3) :
    direct (some alpha) lane value = alpha := by simp [direct, h]
theorem absent_override (lane value : Nat) : direct none lane value = value := by
  simp [direct]
#print axioms rgb_preserved
#print axioms alpha_replaced
#print axioms absent_override
end GB300Math.TextureDescriptorRelation
open GB300Math.TextureDescriptorRelation

def main (args : List String) : IO UInt32 := do
  let [replacement, reference, actual] := args |
    throw (IO.userError "Expected replacement reference actual")
  unless replacement ∈ ["none","1065353216","15360"] do
    throw (IO.userError "Unsupported descriptor relation")
  let xs ← IO.FS.readBinFile reference
  let ys ← IO.FS.readBinFile actual
  unless xs.size > 0 && xs.size % 16 = 0 && xs.size = ys.size do
    throw (IO.userError "Invalid descriptor relation extent")
  let mut failures := 0
  for i in [:xs.size/4] do
    let expected := direct replacement.toNat? i (word xs i)
    let got := word ys i
    if expected != got then
      if failures < 5 then IO.println s!"witness word={i} expected={expected} actual={got}"
      failures := failures+1
  IO.println s!"result count={xs.size/4} mismatches={failures}"
  return if failures = 0 then 0 else 1
