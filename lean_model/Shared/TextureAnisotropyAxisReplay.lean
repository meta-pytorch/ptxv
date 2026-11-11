/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureAnisotropyAxis

open GB300Math

def readWord (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat + data[4*i+1]!.toNat*256 +
    data[4*i+2]!.toNat*65536 + data[4*i+3]!.toNat*16777216

def main (args : List String) : IO UInt32 := do
  let [anisoText,pattern,inputPath,outputPath,expectedText] := args |
    throw (IO.userError "Expected anisotropy pattern input output expected_mismatches")
  let [some aniso,some expected] := [anisoText,expectedText].map String.toNat? |
    throw (IO.userError "Invalid numeric arguments")
  unless [1,2,4,8,16].contains aniso ∧ ["levels","checker"].contains pattern do
    throw (IO.userError "Invalid descriptor parameters")
  let input ← IO.FS.readBinFile inputPath
  let output ← IO.FS.readBinFile outputPath
  unless input.size=8192*64 ∧ output.size=8192*32+64 do
    throw (IO.userError "Unexpected corpus extent")
  let mut mismatches := 0
  let mut exactChecks := 0
  let mut bounds := 0
  for row in [:8192] do
    let dx0 := readWord input (row*16+4)
    let dx1 := readWord input (row*16+5)
    let dy0 := readWord input (row*16+8)
    let dy1 := readWord input (row*16+9)
    let first := max dx0 dx1
    let second := max dy0 dy1
    unless ((dx0>0 ∧ dx1=0 ∧ dy0=0 ∧ dy1>0) ∨
        (dx0=0 ∧ dx1>0 ∧ dy0>0 ∧ dy1=0)) ∧
        first ≥ 0x00800000 ∧ second ≥ 0x00800000 ∧
        first < 0x7f800000 ∧ second < 0x7f800000 ∧
        max first second / 2^23 > aniso.log2 do
      throw (IO.userError "Invalid positive normal axis fixture")
    for channel in [:8] do
      let observed := readWord output (row*8+channel)
      let predicted : Option Nat :=
        if channel≥5 then some 0xa5a5a5a5
        else if channel=4 then some 1
        else if pattern="levels" ∨ channel=0 then
          some (TextureAnisotropyAxis.sample first second aniso.log2)
        else none
      match predicted with
      | some value =>
        exactChecks := exactChecks+1
        if observed != value then
          if mismatches<8 then IO.println s!"difference row={row} channel={channel} predicted={value} observed={observed}"
          mismatches := mismatches+1
      | none =>
        bounds := bounds+1
        let valid := if pattern="levels" ∨ channel=0 then 0x3f800000≤observed ∧ observed≤0x41100000
          else observed≤0x3f800000
        if !valid then mismatches := mismatches+1
  for i in [:16] do
    if readWord output (8192*8+i) != 0xa5a5a5a5 then mismatches := mismatches+1
  IO.println s!"exact_checks={exactChecks} bounded_checks={bounds} guard_words=16 mismatches={mismatches} expected={expected}"
  return if mismatches=expected then 0 else 1
