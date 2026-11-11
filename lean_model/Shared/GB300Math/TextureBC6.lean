/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureBC6Tables

namespace GB300Math.TextureBC6
open TextureBC6Tables

def readBits (block offset count : Nat) : Nat := block / 2^offset % 2^count
def reverseBits (value width : Nat) : Nat :=
  (List.range width).foldl (fun out bit => out*2 + readBits value bit 1) 0

def signExtend (value width : Nat) : Int :=
  if value ≥ 2^(width-1) then (value : Int) - 2^width else value

def inverse (value base : Int) (width : Nat) (signed : Bool) : Int :=
  let bits := ((value+base) % (2^width : Nat)).toNat
  if signed then signExtend bits width else bits

def unquantize (value : Int) (width : Nat) (signed : Bool) : Int :=
  if signed then
    if width ≥ 16 then value else
      let magnitude := value.natAbs
      let out := if magnitude = 0 then 0 else if magnitude ≥ 2^(width-1)-1 then 32767
        else (magnitude*32768+16384)/2^(width-1)
      if value < 0 then -(out : Int) else out
  else
    if width ≥ 15 then value else if value = 0 then 0 else
      if value = (2^width : Nat)-1 then 65535 else (value*65536+32768)/(2^width : Nat)

def interpolate (a b : Int) (weight : Nat) : Int :=
  (a*(64-(weight : Int))+b*weight+32)/64

def finish (value : Int) (signed : Bool) : Nat :=
  if signed then
    let magnitude := value.natAbs*31/32
    magnitude + if value < 0 && magnitude != 0 then 32768 else 0
  else (value*31/64).toNat

def halfToFloat (bits : Nat) : Nat :=
  let sign := bits/32768*2147483648
  let exponent := bits/1024%32
  let fraction := bits%1024
  sign + if exponent = 0 then
    if fraction = 0 then 0 else
      let top := (List.range 10).foldl (fun n i => if fraction ≥ 2^i then i else n) 0
      (top+103)*8388608+(fraction-2^top)*2^(23-top)
  else if exponent = 31 then 2139095040+fraction*8192
  else (exponent+112)*8388608+fraction*8192

def decode (block : Nat) (signed : Bool) : Option (Array Nat) := do
  let tag :=  readBits block 0 (if readBits block 0 2 ≤ 1 then 2 else 5)
  let mode ← (List.range 14).find? (fun i => prefixes[i]! = tag)
  Id.run do
    let mut fields := Array.replicate 13 0
    for field in layouts[mode]! do
      let value := readBits block field.offset field.width
      let value := if field.reversed then reverseBits value field.width else value
      fields := fields.set! field.destination (fields[field.destination]! + value*2^field.shift)
    let partition := fields[12]!
    let endpoints := if mode < 10 then 4 else 2
    let width := bitWidths[mode]!
    let mut values := Array.replicate 12 (0 : Int)
    for channel in [:3] do
      for endpoint in [:endpoints] do
        let raw := fields[channel*4+endpoint]!
        let value := if endpoint = 0 then
          if signed then signExtend raw width else raw
        else if (mode != 9 && mode != 10) || signed then
          signExtend raw bitWidths[(channel+1)*14+mode]! else raw
        values := values.set! (channel*4+endpoint) value
      if mode != 9 && mode != 10 then
        for endpoint in [1:endpoints] do
          values := values.set! (channel*4+endpoint)
            (inverse values[channel*4+endpoint]! values[channel*4]! width signed)
      for endpoint in [:endpoints] do
        values := values.set! (channel*4+endpoint) (unquantize values[channel*4+endpoint]! width signed)
    let mut offset := if mode < 10 then 82 else 65
    let mut output := Array.replicate 64 0
    for pixel in [:16] do
      let part := if mode < 10 then partitions[partition*16+pixel]! else if pixel = 0 then 128 else 0
      let indexBits := (if mode < 10 then 3 else 4) - if part ≥ 128 then 1 else 0
      let index := readBits block offset indexBits
      offset := offset+indexBits
      let weight := (if mode < 10 then weights3 else weights4)[index]!
      let endpoint := part%2*2
      for channel in [:3] do
        let value := interpolate values[channel*4+endpoint]! values[channel*4+endpoint+1]! weight
        output := output.set! (pixel*4+channel) (halfToFloat (finish value signed))
    return if offset = 128 then some output else none

theorem equal_endpoints (a : Int) (w : Nat) : interpolate a a w = a := by
  simp only [interpolate]
  have h : a*(64-(w : Int))+a*w+32 = a*64+32 := by
    rw [Int.mul_sub]
    omega
  rw [h]
  omega

theorem interpolation_error (a b : Int) (w : Nat) :
    let idealNumerator := a*(64-(w : Int))+b*w
    64*interpolate a b w ≤ idealNumerator+32 ∧
      idealNumerator ≤ 64*interpolate a b w+32 := by
  dsimp only [interpolate]
  omega

theorem half_one : halfToFloat 15360 = 1065353216 := by decide

theorem half_smallest : halfToFloat 1 = 864026624 := by decide

def word (data : ByteArray) (offset size : Nat) : Nat :=
  (List.range size).foldl (fun n i => n+data[offset+i]!.toNat*2^(8*i)) 0
end GB300Math.TextureBC6

#print axioms GB300Math.TextureBC6.equal_endpoints
#print axioms GB300Math.TextureBC6.interpolation_error
#print axioms GB300Math.TextureBC6.half_one
#print axioms GB300Math.TextureBC6.half_smallest

def main (args : List String) : IO UInt32 := do
  let [format,input,output] := args | throw (IO.userError "Expected signed/unsigned BC6H format, blocks and output")
  unless format = "bc6hu" || format = "bc6hs" do throw (IO.userError "Invalid BC6H format")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size = 4096*16 && ys.size = 65536*16 do throw (IO.userError "Invalid capture sizes")
  let mut bad := 0
  for block in [:4096] do
    let bits := GB300Math.TextureBC6.word xs (block*16) 16
    let some decoded := GB300Math.TextureBC6.decode bits (format = "bc6hs") | throw (IO.userError s!"Invalid BC6H block {block}")
    unless decoded.size = 64 do throw (IO.userError "Invalid decoder output size")
    for pixel in [:16] do
      let i := (block/64*4+pixel/4)*256+block%64*4+pixel%4
      for component in [:4] do
        let expected := decoded[pixel*4+component]!
        let actual := GB300Math.TextureBC6.word ys ((i*4+component)*4) 4
        if actual != expected then
          if bad < 4 then IO.println s!"witness block={block} pixel={pixel} component={component} expected={expected} actual={actual}"
          bad := bad+1
  IO.println s!"result count=262144 mismatches={bad}"
  return if bad = 0 then 0 else 1
