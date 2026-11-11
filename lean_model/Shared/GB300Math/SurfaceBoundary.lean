/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std
namespace GB300Math.SurfaceBoundary

def edge (pixel : Nat) : Bool := pixel % 256 = 0 || pixel % 256 = 255

def expectedByte (mode : Bool) (pixel initial baseline : Nat) : Nat :=
  if mode && edge pixel then initial else baseline

theorem zero_left (row initial baseline : Nat) :
    expectedByte true (row*256) initial baseline = initial := by simp [expectedByte,edge]

theorem clamp_uses_baseline (pixel initial baseline : Nat) :
    expectedByte false pixel initial baseline = baseline := by simp [expectedByte]
#print axioms zero_left
#print axioms clamp_uses_baseline
end GB300Math.SurfaceBoundary
open GB300Math.SurfaceBoundary

def main (args : List String) : IO UInt32 := do
  let [mode, pixelBytes, initial, baseline, output] := args |
    throw (IO.userError "Expected mode pixelBytes initial baseline output")
  let some mode := mode.toNat? | throw (IO.userError "Invalid mode")
  let some pixelBytes := pixelBytes.toNat? | throw (IO.userError "Invalid pixel width")
  unless mode < 2 && pixelBytes ∈ [4,8,16] do throw (IO.userError "Unsupported configuration")
  let xs ← IO.FS.readBinFile initial
  let bs ← IO.FS.readBinFile baseline
  let ys ← IO.FS.readBinFile output
  unless bs.size = 65536*pixelBytes && ys.size = bs.size && xs.size ≥ bs.size do
    throw (IO.userError "Invalid capture extent")
  let mut failures := 0
  for i in [:ys.size] do
    let expected := expectedByte (mode = 1) (i/pixelBytes) xs[i]!.toNat bs[i]!.toNat
    if expected != ys[i]!.toNat then
      if failures < 5 then IO.println s!"witness byte={i} expected={expected} actual={ys[i]!.toNat}"
      failures := failures+1
  IO.println s!"result count={ys.size} mismatches={failures}"
  return if failures = 0 then 0 else 1
