/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TanhSmall

def readWord (data : ByteArray) (i : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:4] do
    word := word + data[i + j]!.toNat * 2 ^ (8 * j)
  return word

def main (args : List String) : IO UInt32 := do
  let [input, output] := args | throw (IO.userError "Expected identity.u32 tanh_approx.u32")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size = ys.size ∧ xs.size % 4 = 0 do
    throw (IO.userError "Input/output sizes differ")
  for i in [:xs.size / 4] do
    let x := readWord xs (4 * i)
    let y := readWord ys (4 * i)
    unless GB300Math.TanhSmall.firstBinade x = some y do
      throw (IO.userError s!"TANH model mismatch: input={x}, hardware={y}")
  IO.println s!"Matched TANH first-binade model on {xs.size / 4} inputs"
  return 0
