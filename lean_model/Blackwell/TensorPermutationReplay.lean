/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TensorSmallBlackwell
import GB300Math.TensorBlackwellUnderflow

def word (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat + data[4*i+1]!.toNat*256 +
    data[4*i+2]!.toNat*65536 + data[4*i+3]!.toNat*16777216

def main (args : List String) : IO UInt32 := do
  let [fmt, inputPath, outputPath, expectedText] := args |
    throw (IO.userError "Expected format input output expected-mismatches")
  let predicting := expectedText = "predict"
  let some expected := if predicting then some 0 else expectedText.toNat? | throw (IO.userError "Invalid mismatch count")
  let k := if fmt = "f16" ∨ fmt = "bf16" then 16 else if fmt = "tf32" then 8 else 32
  let input ← IO.FS.readBinFile inputPath
  let output ← if predicting then pure ByteArray.empty else IO.FS.readBinFile outputPath
  let stride := 136*k+1024
  unless input.size > 0 ∧ input.size % (4*stride) = 0 ∧
      (predicting ∨ output.size = input.size/(4*stride)*4096) do
    throw (IO.userError "Invalid input/output extent")
  let mut mismatches := 0
  let mut predictions := ByteArray.empty
  for matrix in [:input.size/(4*stride)] do
    let base := matrix*stride
    let bs := (List.range 8).toArray.map fun col =>
      (List.range k).toArray.map fun j => word input (base+128*k+col*k+j)
    for row in [:128] do
      let a := (List.range k).toArray.map fun j => word input (base+row*k+j)
      for col in [:8] do
        let cell := row*8+col
        let c := word input (base+136*k+cell)
        let some predicted := (if k = 32 then GB300Math.TensorSmallBlackwell.dot fmt fmt a bs[col]! c
          else GB300Math.TensorBlackwellUnderflow.evaluate fmt a bs[col]! c true) |
          throw (IO.userError s!"Rejected operands: matrix {matrix}, cell {cell}")
        if predicting then
          for shift in [:4] do
            predictions := predictions.push (predicted / 2^(8*shift)).toUInt8
        let observed := if predicting then predicted else word output (matrix*1024+cell)
        if predicted != observed then
          if mismatches < 8 then
            IO.println s!"difference matrix={matrix} cell={cell} predicted={predicted} observed={observed}"
          mismatches := mismatches+1
  if predicting then IO.FS.writeBinFile outputPath predictions
  IO.println s!"cells={if predicting then predictions.size/4 else output.size/4} mismatches={mismatches} expected={expected}"
  return if mismatches = expected then 0 else 1
