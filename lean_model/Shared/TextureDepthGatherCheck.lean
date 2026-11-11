/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureDepthGather

open GB300Math.TextureDepthGather

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut n := 0
  for j in [:4] do n := n+data[offset+j]!.toNat*2^(8*j)
  return n

def main (args : List String) : IO UInt32 := do
  let [depthPath,inputPath,outputPath,modeText,functionText,referenceText,dxText,dyText,expectedText] := args |
    throw (IO.userError "Expected depths inputs outputs mode function reference")
  let some expectedErrors := expectedText.toNat? | throw (IO.userError "Invalid expected errors")
  let some dx := dxText.toInt? | throw (IO.userError "Invalid offset x")
  let some dy := dyText.toInt? | throw (IO.userError "Invalid offset y")
  unless -8 ≤ dx ∧ dx ≤ 7 ∧ -8 ≤ dy ∧ dy ≤ 7 do throw (IO.userError "Offset out of range")
  let some mode := modeText.toNat? | throw (IO.userError "Invalid mode")
  let some function := functionText.toNat? | throw (IO.userError "Invalid function")
  let some reference := referenceText.toNat? | throw (IO.userError "Invalid reference")
  unless mode < 4 ∧ (function = 1 ∨ function = 2 ∨ function = 5) ∧ reference ≤ 0xffffffff do
    throw (IO.userError "Unsupported descriptor")
  let ds ← IO.FS.readBinFile depthPath
  let xs ← IO.FS.readBinFile inputPath
  let ys ← IO.FS.readBinFile outputPath
  unless ds.size = 1024 ∧ xs.size > 0 ∧ xs.size%8 = 0 ∧ ys.size = xs.size+256 do
    throw (IO.userError "Invalid PCF capture sizes")
  let depths := (Array.range 256).map fun i => readWord ds (4*i)
  let mut bad := 0
  for i in [:xs.size/8] do
    let expected := sample depths (readWord xs (8*i)) (readWord xs (8*i+4)) mode function reference dx dy
    for lane in [:2] do
      let actual := readWord ys (8*i+4*lane)
      if expected ≠ actual then
        if bad < 3 then IO.println s!"Mismatch row={i} lane={lane} model={expected} hardware={actual}"
        bad := bad+1
  for i in [:64] do
    if readWord ys (xs.size+4*i) ≠ 0xdeadbeef then bad := bad+1
  IO.println s!"result count={xs.size/4} mismatches={bad} expected={expectedErrors}"
  return if bad = expectedErrors then 0 else 1
