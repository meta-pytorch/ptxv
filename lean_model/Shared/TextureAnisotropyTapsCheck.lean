/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureAnisotropyTaps
open GB300Math

def word (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat + data[4*i+1]!.toNat*256 +
    data[4*i+2]!.toNat*65536 + data[4*i+3]!.toNat*16777216

def dyadicBits (n : Int) (shift : Nat) : Nat :=
  let a := n.natAbs
  if a = 0 then 0 else
    let e := a.log2
    (if n < 0 then 2^31 else 0)+(127+e-shift)*2^23+
      (a-2^e)*2^(23-e)

def integerValue (v : Nat) : Int :=
  let e := v/2^23%256
  let n : Nat := (v%2^23+2^23)/2^(150-e)
  if v ≥ 2^31 then -(n:Int) else n

def coordinate (v : Nat) : Int :=
  let e := v/2^23%256
  ((v%2^23+2^23)/2^(134-e):Nat)-128

def main (args : List String) : IO UInt32 := do
  let [inputPath,imagePath,outputPath] := args |
    throw (IO.userError "Expected inputs image outputs")
  let inputs ← IO.FS.readBinFile inputPath
  let image ← IO.FS.readBinFile imagePath
  let outputs ← IO.FS.readBinFile outputPath
  let count := inputs.size/64
  unless count>0 ∧ inputs.size%64=0 ∧ image.size=256*256*16 ∧ outputs.size=count*32+64 do
    throw (IO.userError "Invalid capture extent")
  let mut values : Array Int := #[]
  for i in [:256*256*4] do
    let bits := word image i
    let n := integerValue bits
    unless n.natAbs ≤ 255 ∧ dyadicBits n 0 = bits do
      throw (IO.userError s!"Image outside exact integer domain at {i}")
    values := values.push n
  let mut bad := 0
  for row in [:count] do
    let dx0 := word inputs (row*16+4)
    let dx1 := word inputs (row*16+5)
    let dy0 := word inputs (row*16+8)
    let dy1 := word inputs (row*16+9)
    let vertical := dy0=0x3b000000 ∧ dy1=0 ∧ dx0=0
    unless vertical ∨ (dy0=0 ∧ dy1=0x3b000000 ∧ dx1=0) do
      throw (IO.userError s!"Outside axis domain at {row}")
    let gradient := if vertical then dx1 else dx0
    unless gradient%2^31 ≤ 0x3d800000 do
      throw (IO.userError s!"Major width exceeds16 at {row}")
    let majorCoord := word inputs (row*16+(if vertical then 1 else 0))
    let minorCoord := word inputs (row*16+(if vertical then 0 else 1))
    let q := coordinate majorCoord
    let minorQ := coordinate minorCoord
    unless q ≥ 2560 ∧ q ≤ 245*256 ∧ minorQ ≥ 0 ∧ minorQ ≤ 255*256 ∧ minorQ%256=0 ∧
        dyadicBits (q+128) 16 = majorCoord ∧ dyadicBits (minorQ+128) 16 = minorCoord do
      throw (IO.userError s!"Outside exact coordinate domain at {row}")
    let log := TextureAnisotropyTaps.logarithm gradient
    let (groups,taps) := TextureAnisotropyTaps.taps log (gradient ≥ 2^31)
    for channel in [:4] do
      let mut numerator : Int := 0
      for tap in taps do
        let position := q+tap.offset
        let index := position/256
        let fraction := (position%256).toNat
        let other := (minorQ/256).toNat
        let (x,y) := if vertical then (other,index.toNat) else (index.toNat,other)
        let low := values[(y*256+x)*4+channel]!
        let (xh,yh) := if vertical then (x,y+1) else (x+1,y)
        let high := values[(yh*256+xh)*4+channel]!
        numerator := numerator+TextureAnisotropyTaps.contribution tap.weight fraction vertical low high
      let expected := dyadicBits numerator (8+groups.log2)
      let actual := word outputs (row*8+channel)
      if expected != actual then
        if bad<8 then IO.println s!"row={row} channel={channel} gradient={gradient} q={q} expected={expected} actual={actual}"
        bad := bad+1
    if word outputs (row*8+4) != 1 then bad := bad+1
    for k in [5:8] do
      if word outputs (row*8+k) != 0xa5a5a5a5 then bad := bad+1
  for k in [:16] do
    if word outputs (count*8+k) != 0xa5a5a5a5 then bad := bad+1
  IO.println s!"rows={count} numeric_words={count*4} state_words={count*4} guard_words=16 differences={bad}"
  return if bad=0 then 0 else 1
