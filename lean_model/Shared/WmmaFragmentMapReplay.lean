/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.WmmaFragmentMap

open GB300Math.WmmaFragmentMap

def readWord (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat + data[4*i+1]!.toNat*256 +
    data[4*i+2]!.toNat*65536 + data[4*i+3]!.toNat*16777216

def main (args : List String) : IO UInt32 := do
  let [mapPath,inputPath,outputPath,expectedText] := args |
    throw (IO.userError "Expected map input output expected/predict")
  let mapping ← IO.FS.readBinFile mapPath
  unless mapping.size % 4 = 0 do throw (IO.userError "Invalid map extent")
  let mut index := 0
  let mut layouts : Array Layout := #[]
  for _ in [:1024] do
    unless index+2 ≤ mapping.size/4 do throw (IO.userError "Truncated map")
    let constant := readWord mapping index
    let count := readWord mapping (index+1)
    index := index+2
    unless count ≤ 32 ∧ index+3*count ≤ mapping.size/4 do
      throw (IO.userError "Invalid map count")
    let mut segments : Array Segment := #[]
    let mut bits := constant
    for _ in [:count] do
      let s : Segment := ⟨readWord mapping index,readWord mapping (index+1),readWord mapping (index+2)⟩
      unless s.width > 0 ∧ s.width ≤ 32 ∧ s.source+s.width ≤ 32768 ∧ s.target+s.width ≤ 32 do
        throw (IO.userError "Invalid map segment")
      let mask := (2^s.width-1)*2^s.target
      unless Nat.land bits mask = 0 do throw (IO.userError "Overlapping map segments")
      bits := Nat.lor bits mask
      segments := segments.push s
      index := index+3
    layouts := layouts.push ⟨constant,segments⟩
  unless index = mapping.size/4 do throw (IO.userError "Trailing map data")
  let input ← IO.FS.readBinFile inputPath
  unless input.size > 0 ∧ input.size % 4096 = 0 do throw (IO.userError "Invalid input extent")
  let predict := expectedText = "predict"
  let expected := expectedText.toNat?.getD 0
  unless predict ∨ expectedText.toNat?.isSome do throw (IO.userError "Invalid mismatch count")
  let output ← if predict then pure ByteArray.empty else IO.FS.readBinFile outputPath
  unless predict ∨ output.size = input.size do throw (IO.userError "Invalid output extent")
  let mut result := ByteArray.empty
  let mut mismatches := 0
  for sample in [:input.size/4096] do
    let words := (Array.range 1024).map fun j => readWord input (sample*1024+j)
    for cell in [:1024] do
      let predicted := word words layouts[cell]!
      if predict then
        for i in [:4] do result := result.push (UInt8.ofNat (predicted / 2^(8*i) % 256))
      else
        let observed := readWord output (sample*1024+cell)
        if predicted != observed then
          if mismatches < 8 then IO.println s!"difference sample={sample} cell={cell} predicted={predicted} observed={observed}"
          mismatches := mismatches+1
  if predict then IO.FS.writeBinFile outputPath result
  IO.println s!"cells={input.size/4} mismatches={mismatches} expected={expected}"
  return if mismatches = expected then 0 else 1
