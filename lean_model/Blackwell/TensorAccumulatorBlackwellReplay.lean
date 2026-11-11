/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TensorAccumulatorBlackwell

def word (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat + data[4*i+1]!.toNat*256 +
    data[4*i+2]!.toNat*65536 + data[4*i+3]!.toNat*16777216

def main (args : List String) : IO UInt32 := do
  let [fmt, shiftText, modeText, maskText, inputPath, outputPath, expectedText] := args |
    throw (IO.userError "Expected format shift mode mask input output expected-mismatches")
  let some shift := shiftText.toNat? | throw (IO.userError "Invalid shift")
  let some mode := modeText.toNat? | throw (IO.userError "Invalid mode")
  let some expected := expectedText.toNat? | throw (IO.userError "Invalid mismatch count")
  let some masks := (maskText.splitOn ",").mapM String.toNat? | throw (IO.userError "Invalid mask")
  unless masks.length = 4 ∧ masks.all (· < 2^32) ∧ shift ≤ 15 ∧ mode ≤ 2 do
    throw (IO.userError "Invalid mode, shift or masks")
  let k ← match fmt with
    | "f16" | "bf16" => pure 16
    | "tf32" => pure 8
    | _ => throw (IO.userError "Unsupported format")
  let input ← IO.FS.readBinFile inputPath
  let output ← IO.FS.readBinFile outputPath
  let stride := 136*k+1024
  unless input.size > 0 ∧ input.size % (4*stride) = 0 ∧
      output.size = input.size/(4*stride)*4096 do
    throw (IO.userError "Invalid input/output extent")
  let mut mismatches := 0
  for matrix in [:input.size/(4*stride)] do
    let base := matrix*stride
    let bs := (List.range 8).toArray.map fun col =>
      (List.range k).toArray.map fun j => word input (base+128*k+col*k+j)
    for row in [:128] do
      let a := (List.range k).toArray.map fun j => word input (base+row*k+j)
      for col in [:8] do
        let cell := row*8+col
        let c := word input (base+136*k+cell)
        let some computed := GB300Math.TensorAccumulatorBlackwell.evaluate fmt a bs[col]! c shift mode |
          throw (IO.userError s!"Rejected operands: matrix {matrix}, cell {cell}")
        let predicted := if masks[row/32]! / 2^(row%32) % 2 = 1 then c else computed
        let observed := word output (matrix*1024+cell)
        if predicted != observed then
          if mismatches < 8 then
            IO.println s!"difference matrix={matrix} cell={cell} predicted={predicted} observed={observed}"
          mismatches := mismatches+1
  IO.println s!"cells={output.size/4} mismatches={mismatches} expected={expected}"
  return if mismatches = expected then 0 else 1
