/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureMipmap
open GB300Math.TextureMipmap

def word (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut n := 0
  for j in [:4] do n := n + data[offset+j]!.toNat * 2^(8*j)
  return n

def main (args : List String) : IO UInt32 := do
  let [ls, input, output] := args | throw (IO.userError "Expected levels inputs outputs")
  let some levels := ls.toNat? | throw (IO.userError "Invalid levels")
  unless levels = 6 ∨ levels = 9 do throw (IO.userError "Unsupported level count")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 ∧ xs.size % 64 = 0 ∧ ys.size = xs.size/2 do
    throw (IO.userError "Invalid capture sizes")
  let mut bad := 0
  for i in [:xs.size/64] do
    let lod := word xs (i*64+48)
    let value := sample lod levels
    for lane in [:8] do
      let expected := if lane < 4 then value else if lane = 4 then 1 else 0
      let actual := word ys (i*32+lane*4)
      if expected ≠ actual then
        if bad < 3 then IO.println s!"Mismatch row={i} lane={lane} model={expected} hardware={actual}"
        bad := bad+1
  IO.println s!"result count={xs.size/8} mismatches={bad}"
  return if bad = 0 then 0 else 1
