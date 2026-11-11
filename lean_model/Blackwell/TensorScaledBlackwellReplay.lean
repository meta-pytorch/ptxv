/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TensorScaledBlackwell

def word (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat + data[4*i+1]!.toNat*256 +
    data[4*i+2]!.toNat*65536 + data[4*i+3]!.toNat*16777216

def main (args : List String) : IO UInt32 := do
  let [nvText, kText, floorText, inputPath, outputPath, expectedText] := args |
    throw (IO.userError "Expected nv k minimum-step input output expected-mismatches")
  let some minimumStep := floorText.toInt? | throw (IO.userError "Invalid minimum step")
  let some k := kText.toNat? | throw (IO.userError "Invalid K")
  let some expected := expectedText.toNat? | throw (IO.userError "Invalid mismatch count")
  unless (nvText = "0" ∨ nvText = "1") ∧ (k = 64 ∨ k = 96) do
    throw (IO.userError "Invalid kind or K")
  let input ← IO.FS.readBinFile inputPath
  let output ← IO.FS.readBinFile outputPath
  let stride := 136*k/8 + 1024 + 4096
  unless input.size > 0 ∧ input.size % (4*stride) = 0 ∧
      output.size = input.size/(4*stride)*4096 do
    throw (IO.userError "Invalid input/output extent")
  let nv := nvText = "1"
  let blocks := k / (if nv then 16 else 32)
  let mut mismatches := 0
  for matrix in [:input.size/(4*stride)] do
    let base := matrix*stride
    let co := base+136*k/8
    let so := co+1024
    for side in [:2] do
      for row in [:128] do
        for col in [:16] do
          unless word input (so+side*2048+row*16+col) =
              word input (so+side*2048+(row%32)*16+col) do
            throw (IO.userError "Scale physical partitions must repeat identically")
    let bs := (List.range 8).toArray.map fun col =>
      (List.range k).toArray.map fun j =>
        word input (base+128*k/8+col*k/8+j/8) / 2^(4*(j%8)) % 16
    let sb := (List.range 8).toArray.map fun col =>
      (List.range blocks).toArray.map fun j =>
        word input (so+2048+(col%32)*16+
          GB300Math.TensorScaledIdeal.physicalColumn col j) / 2^(8*(j%4)) % 256
    for row in [:128] do
      let a := (List.range k).toArray.map fun j =>
        word input (base+row*k/8+j/8) / 2^(4*(j%8)) % 16
      let sa := (List.range blocks).toArray.map fun j =>
        word input (so+(row%32)*16+GB300Math.TensorScaledIdeal.physicalColumn row j) /
          2^(8*(j%4)) % 256
      for col in [:8] do
        let cell := row*8+col
        let some predicted := GB300Math.TensorScaledBlackwell.evaluate nv a bs[col]! sa sb[col]!
            (word input (co+cell)) minimumStep |
          throw (IO.userError s!"Rejected operands: matrix {matrix}, cell {cell}")
        let observed := word output (matrix*1024+cell)
        if predicted != observed then
          if mismatches < 8 then
            IO.println s!"difference matrix={matrix} cell={cell} predicted={predicted} observed={observed}"
          mismatches := mismatches+1
  IO.println s!"cells={output.size/4} mismatches={mismatches} expected={expected}"
  return if mismatches = expected then 0 else 1
