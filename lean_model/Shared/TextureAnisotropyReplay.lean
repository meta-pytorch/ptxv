/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureAnisotropy

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
  unless input.size=4836*64 ∧ output.size=4836*32+64 do
    throw (IO.userError "Unexpected corpus extent")
  let mut mismatches := 0
  let mut exactChecks := 0
  let mut bounds := 0
  for row in [:4836] do
    let dx0 := readWord input (row*16+4)
    let dx1 := readWord input (row*16+5)
    let dy0 := readWord input (row*16+8)
    let dy1 := readWord input (row*16+9)
    if row<484 ∧ row%4<2 then
      let shape := if row%4=0 then dx0>0 ∧ dx1=0 ∧ dy0=0 ∧ dy1>0
        else dx0=0 ∧ dx1>0 ∧ dy0>0 ∧ dy1=0
      if !shape ∨ max dx0 dx1 % 2^23 != 0 ∨ max dy0 dy1 % 2^23 != 0 ∨
          max dx0 dx1 ≥ 0x7f800000 ∨ max dy0 dy1 ≥ 0x7f800000 then
        throw (IO.userError "Invalid axis fixture")
    if 484≤row ∧ row<740 then
      let x := (row-484)/16
      let y := (row-484)%16
      unless readWord input (row*16)=MultisampleFetch.encode (2*x+1)+7*2^23 ∧
          readWord input (row*16+1)=MultisampleFetch.encode (2*y+1)+7*2^23 ∧
          dx0=0 ∧ dx1=0 ∧ dy0=0 ∧ dy1=0 do
        throw (IO.userError "Invalid point fixture")
    for channel in [:8] do
      let observed := readWord output (row*8+channel)
      let predicted : Option Nat :=
        if channel≥5 then some 0xa5a5a5a5
        else if channel=4 then some 1
        else if row<484 ∧ row%4<2 ∧ (pattern="levels" ∨ channel=0) then
          some (TextureAnisotropy.levelWord (((max dx0 dx1 / 2^23 % 256 : Nat) : Int)-119)
            (((max dy0 dy1 / 2^23 % 256 : Nat) : Int)-119) (aniso.log2 : Int))
        else if 484≤row ∧ row<740 then
          some (TextureAnisotropy.pointWord (pattern="checker") ((row-484)/16) ((row-484)%16) channel)
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
    if readWord output (4836*8+i) != 0xa5a5a5a5 then mismatches := mismatches+1
  IO.println s!"exact_checks={exactChecks} bounded_checks={bounds} guard_words=16 mismatches={mismatches} expected={expected}"
  return if mismatches=expected then 0 else 1
