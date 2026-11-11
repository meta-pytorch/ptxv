/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GpuMath

open GB300Math

def readWord (bytes : ByteArray) (index : Nat) : Word32 :=
  let offset := index * 4
  BitVec.ofNat 32 (bytes[offset]!.toNat + bytes[offset + 1]!.toNat * 256 +
    bytes[offset + 2]!.toNat * 65536 + bytes[offset + 3]!.toNat * 16777216)

def checkFile (directory : System.FilePath) (inputs : ByteArray)
    (operation : String) (model : Word32 → Option Word32) : IO Nat := do
  let outputs ← IO.FS.readBinFile (directory / s!"{operation}.u32")
  unless outputs.size = inputs.size do
    throw (IO.userError s!"Input/output size mismatch: {operation}")
  let mut covered := 0
  let mut mismatches := 0
  for index in [:inputs.size / 4] do
    let input := readWord inputs index
    if let some expected := model input then
      covered := covered + 1
      let actual := readWord outputs index
      if expected != actual then
        if mismatches < 8 then
          IO.eprintln s!"{operation}: input={input.toNat} model={expected.toNat} device={actual.toNat}"
        mismatches := mismatches + 1
  IO.println s!"{operation} covered={covered} unmodeled={inputs.size / 4 - covered} mismatches={mismatches}"
  return mismatches

def main (args : List String) : IO UInt32 := do
  let [directory] := args | throw (IO.userError "Usage: check <capture-directory>")
  let directory : System.FilePath := directory
  let inputs ← IO.FS.readBinFile (directory / "inputs.u32")
  unless inputs.size > 0 ∧ inputs.size % 4 = 0 do
    throw (IO.userError "Expected nonempty little-endian uint32 input words")
  let models : List (String × (Word32 → Option Word32)) := [
    ("ex2_approx", ex2Special), ("ex2_approx_ftz", ex2Special),
    ("lg2_approx", lg2Special false), ("lg2_approx_ftz", lg2Special true),
    ("tanh_approx", tanhSpecial)]
  let mut failures := 0
  for (name, model) in models do
    failures := failures + (← checkFile directory inputs name model)
  return if failures = 0 then 0 else 1
