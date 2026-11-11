/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.Fp64Seed

def wordAt (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do
    word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

def main (args : List String) : IO UInt32 := do
  let [input, rcp, rsq] := args | throw (IO.userError "Expected input.u64 rcp.u64 rsq.u64")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile rcp
  let zs ← IO.FS.readBinFile rsq
  unless xs.size % 8 = 0 ∧ xs.size = ys.size ∧ xs.size = zs.size do
    throw (IO.userError "Input/output byte lengths differ")
  for i in [:xs.size / 8] do
    let x := wordAt xs (8 * i)
    for (mode, output) in [(false, ys), (true, zs)] do
      let actual := wordAt output (8 * i)
      let expected := GB300Math.Fp64Seed.evaluate mode x
      unless actual = expected do
        throw (IO.userError s!"Mismatch row={i} rsq={mode} input={x} hardware={actual} model={expected}")
  IO.println s!"Matched both FP64 seed models on {xs.size / 8} inputs"
  return 0
