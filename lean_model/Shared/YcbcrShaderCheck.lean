/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.YcbcrShader

private def word (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat + data[4*i+1]!.toNat*256 +
    data[4*i+2]!.toNat*65536 + data[4*i+3]!.toNat*16777216

def main (args : List String) : IO UInt32 := do
  let [path,inputPath,modeText,rawText,expectedText] := args |
    throw (IO.userError "Expected output inputs mode raw expected_mismatches")
  let [some mode,some raw,some expected] := [modeText,rawText,expectedText].map String.toNat? |
    throw (IO.userError "Invalid numeric argument")
  unless mode < 5 && raw < 2 do throw (IO.userError "Invalid mode")
  let input ← IO.FS.readBinFile inputPath
  unless input.size = 24 do throw (IO.userError "Expected16luma+8chroma bytes")
  let data ← IO.FS.readBinFile path
  unless data.size = 512 do throw (IO.userError "Invalid extent")
  let mut mismatches := 0
  for i in [:16] do
    let chroma := (i/8)*2+(i%4)/2
    let predicted ← if raw = 1 then pure [GB300Math.YcbcrShader.normalized input[i]!.toNat,0,0,0x3f800000]
      else do
        let some values := GB300Math.YcbcrShader.convert mode input[i]!.toNat input[16+2*chroma]!.toNat input[17+2*chroma]!.toNat |
          throw (IO.userError "Invalid fixture")
        pure values
    for j in [:4] do
      if word data (4*i+j) != predicted[j]! then
        if mismatches < 8 then IO.println s!"row={i} lane={j} predicted={predicted[j]!} observed={word data (4*i+j)}"
        mismatches := mismatches+1
  for i in [64:128] do
    if word data i != 0xdeadbeef then mismatches := mismatches+1
  IO.println s!"records=64 guards=64 mismatches={mismatches} expected={expected}"
  return if mismatches = expected then 0 else 1
