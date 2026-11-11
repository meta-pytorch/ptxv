/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TensorDecompression

def readWord (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat + data[4*i+1]!.toNat*256 +
    data[4*i+2]!.toNat*65536 + data[4*i+3]!.toNat*16777216

def main (args : List String) : IO UInt32 := do
  let [shapeText,widthText,inputPath,outputPath,expectedText] := args |
    throw (IO.userError "Expected shape width input output expected-mismatches")
  let some shape := shapeText.toNat? | throw (IO.userError "Invalid shape")
  let some width := widthText.toNat? | throw (IO.userError "Invalid width")
  let some expected := expectedText.toNat? | throw (IO.userError "Invalid mismatch count")
  unless shape < 6 ∧ (width = 4 ∨ width = 6 ∨ width = 8) do
    throw (IO.userError "Unsupported copy form")
  let input ← IO.FS.readBinFile inputPath
  let output ← IO.FS.readBinFile outputPath
  unless input.size > 0 ∧ input.size % 4096 = 0 ∧ output.size = input.size do
    throw (IO.userError "Invalid input/output extent")
  let mut mismatches := 0
  for sample in [:input.size/4096] do
    let words := (Array.range 1024).map fun j => readWord input (sample*1024+j)
    for cell in [:1024] do
      let predicted := GB300Math.TensorDecompression.word shape width words 0xdeadbeef (cell/8) (cell%8)
      let observed := readWord output (sample*1024+cell)
      if predicted != observed then
        if mismatches < 8 then
          IO.println s!"difference sample={sample} cell={cell} predicted={predicted} observed={observed}"
        mismatches := mismatches+1
  IO.println s!"cells={output.size/4} mismatches={mismatches} expected={expected}"
  return if mismatches = expected then 0 else 1
