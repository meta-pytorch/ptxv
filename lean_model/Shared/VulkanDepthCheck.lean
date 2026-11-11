/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureDepth24

open GB300Math

def word (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat + data[4*i+1]!.toNat*256 +
    data[4*i+2]!.toNat*65536 + data[4*i+3]!.toNat*16777216

def center (index : Nat) : Nat :=
  TextureDepthCompare.encodeSixtyFourths (4*index+2)

def main (args : List String) : IO UInt32 := do
  let [dataPath,queryPath,outputPath,formatText,functionText,expectedText] := args |
    throw (IO.userError "Expected data queries output format function expected")
  let [some format,some function,some expected] :=
      [formatText,functionText,expectedText].map String.toNat? |
    throw (IO.userError "Invalid integer")
  unless (format = 16 || format = 24 || format = 32) && function < 8 do
    throw (IO.userError "Invalid format/comparison")
  let data ← IO.FS.readBinFile dataPath
  let queries ← IO.FS.readBinFile queryPath
  let output ← IO.FS.readBinFile outputPath
  unless data.size = (if format = 16 then 512 else 1024) && queries.size % 16 = 0 &&
      output.size = queries.size+256 do throw (IO.userError "Invalid extent")
  let mut mismatches := 0
  for i in [:queries.size/16] do
    let location := word queries (4*i+3)
    unless location < 256 && word queries (4*i) = center (location%16) &&
        word queries (4*i+1) = center (location/16) do
      throw (IO.userError "Query is not the stated texel center")
    let reference := word queries (4*i+2)
    let depth := if format = 16 then data[2*location]!.toNat + data[2*location+1]!.toNat*256
      else word data location
    if format = 24 && depth ≥ 2^24 then throw (IO.userError "Invalid D24 storage")
    let predicted := if format = 16 then TextureDepthNormalized.compare function reference depth
      else if format = 24 then TextureDepth24.compare function reference depth
      else if TextureDepthCompare.compare function reference depth then 0x3f800000 else 0
    for component in [:4] do
      let value := if component = 0 then predicted else if component = 3 then 0x3f800000 else 0
      if word output (4*i+component) != value then
        if mismatches < 8 then IO.println s!"difference row={i} component={component} predicted={value} actual={word output (4*i+component)}"
        mismatches := mismatches+1
  for i in [:64] do
    if word output (queries.size/4+i) != 0xdeadbeef then mismatches := mismatches+1
  IO.println s!"records={queries.size/4} guards=64 mismatches={mismatches} expected={expected}"
  return if mismatches = expected then 0 else 1
