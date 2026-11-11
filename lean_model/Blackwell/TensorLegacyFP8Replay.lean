/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TensorLegacyFP8

def word (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat + data[4*i+1]!.toNat*256 +
    data[4*i+2]!.toNat*65536 + data[4*i+3]!.toNat*16777216

def main (args : List String) : IO UInt32 := do
  let [afmt, bfmt, kText, halfText, inputPath, outputPath, expectedText] := args |
    throw (IO.userError "Expected afmt bfmt k half input output expected-mismatches")
  let some k := kText.toNat? | throw (IO.userError "Invalid K")
  unless (k = 16 ∨ k = 32) ∧ (halfText = "0" ∨ halfText = "1") do
    throw (IO.userError "Invalid K or output format")
  let some expected := expectedText.toNat? | throw (IO.userError "Invalid mismatch count")
  let input ← IO.FS.readBinFile inputPath
  let output ← IO.FS.readBinFile outputPath
  let stride := 24*k+128
  unless input.size > 0 ∧ input.size % (4*stride) = 0 ∧
      output.size = input.size/(4*stride)*512 do
    throw (IO.userError "Invalid input/output extent")
  let mut mismatches := 0
  for matrix in [:input.size/(4*stride)] do
    let base := matrix*stride
    let bs := (List.range 8).toArray.map fun col =>
      (List.range k).toArray.map fun j =>
        let v := word input (base+16*k+col*k+j)
        v
    for row in [:16] do
      let a := (List.range k).toArray.map fun j =>
        let v := word input (base+row*k+j)
        v
      for col in [:8] do
        let cell := row*8+col
        let c := word input (base+24*k+cell)
        let some predicted := GB300Math.TensorLegacyFP8.dot afmt bfmt (halfText = "1") a bs[col]! c |
          throw (IO.userError s!"Rejected operands: matrix {matrix}, cell {cell}")
        let observed := word output (matrix*128+cell)
        if predicted != observed then
          if mismatches < 8 then
            IO.println s!"difference matrix={matrix} cell={cell} predicted={predicted} observed={observed}"
          mismatches := mismatches+1
  IO.println s!"cells={output.size/4} mismatches={mismatches} expected={expected}"
  return if mismatches = expected then 0 else 1
