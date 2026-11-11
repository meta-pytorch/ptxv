/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureBC7Tables
import GB300Math.TextureRatio

namespace GB300Math.TextureBC7
open TextureBC7Tables

def readBits (block offset count : Nat) : Nat := block / 2^offset % 2^count
def dequant (value width : Nat) : Nat := value*2^(8-width) + value/2^(2*width-8)
def weight (width index : Nat) : Nat :=
  (if width = 2 then weights2 else if width = 3 then weights3 else weights4)[index]!
def interpolate (a b width index : Nat) : Nat :=
  let w := weight width index
  (a*(64-w)+b*w+32)/64

def decodePartitioned (mode block : Nat) : Option (Array Nat) := Id.run do
  let subsets := if mode = 0 || mode = 2 then 3 else if mode = 6 then 1 else 2
  let endpoints := 2*subsets
  let comps := if mode = 6 || mode = 7 then 4 else 3
  let endpointBits := [4,6,5,7,0,0,7,5][mode]!
  let weightBits := if mode = 0 || mode = 1 then 3 else if mode = 6 then 4 else 2
  let partBits := if mode = 0 then 4 else if mode = 6 then 0 else 6
  let pCount := if mode = 2 then 0 else if mode = 1 then 2 else endpoints
  let mut offset := mode+1
  let part := readBits block offset partBits
  offset := offset+partBits
  let mut values := Array.replicate (endpoints*4) 255
  for component in [:comps] do
    for endpoint in [:endpoints] do
      values := values.set! (endpoint*4+component) (readBits block offset endpointBits)
      offset := offset+endpointBits
  let mut pbits := Array.replicate pCount 0
  for i in [:pCount] do
    pbits := pbits.set! i (readBits block offset 1)
    offset := offset+1
  for endpoint in [:endpoints] do
    for component in [:comps] do
      let value := values[endpoint*4+component]!
      let value := if pCount = 0 then dequant value endpointBits else
        dequant (2*value+pbits[if mode = 1 then endpoint/2 else endpoint]!) (endpointBits+1)
      values := values.set! (endpoint*4+component) value
  let mut output := Array.replicate 64 0
  for pixel in [:16] do
    let anchor := pixel = 0 || (subsets = 2 && pixel = anchor2[part]!) ||
      (subsets = 3 && (pixel = anchor31[part]! || pixel = anchor32[part]!))
    let bits := weightBits - if anchor then 1 else 0
    let index := readBits block offset bits
    offset := offset+bits
    let subset := if subsets = 1 then 0 else
      (if subsets = 2 then partition2 else partition3)[part*16+pixel]!
    for component in [:4] do
      let value := interpolate values[subset*8+component]! values[subset*8+4+component]! weightBits index
      output := output.set! (pixel*4+component) value
  return if offset = 128 then some output else none

def decodeDual (mode block : Nat) : Option (Array Nat) := Id.run do
  let mut offset := mode+1
  let rotation := readBits block offset 2
  offset := offset+2
  let indexMode := if mode = 4 then readBits block offset 1 else 0
  offset := offset + if mode = 4 then 1 else 0
  let colorBits := if mode = 4 then 5 else 7
  let alphaBits := if mode = 4 then 6 else 8
  let secondBits := if mode = 4 then 3 else 2
  let mut endpoints := Array.replicate 8 0
  for component in [:4] do
    let bits := if component = 3 then alphaBits else colorBits
    for endpoint in [:2] do
      endpoints := endpoints.set! (endpoint*4+component) (dequant (readBits block offset bits) bits)
      offset := offset+bits
  let mut first := Array.replicate 16 0
  let mut second := Array.replicate 16 0
  for pixel in [:16] do
    let bits := 2 - if pixel = 0 then 1 else 0
    first := first.set! pixel (readBits block offset bits)
    offset := offset+bits
  for pixel in [:16] do
    let bits := secondBits - if pixel = 0 then 1 else 0
    second := second.set! pixel (readBits block offset bits)
    offset := offset+bits
  let mut output := Array.replicate 64 0
  for pixel in [:16] do
    for component in [:4] do
      let fromSecond := if component = 3 then indexMode = 0 else indexMode = 1
      let bits := if fromSecond then secondBits else 2
      let index := (if fromSecond then second else first)[pixel]!
      let value := interpolate endpoints[component]! endpoints[4+component]! bits index
      let destination := if rotation = 0 then component else
        if component = 3 then rotation-1 else if component = rotation-1 then 3 else component
      output := output.set! (pixel*4+destination) value
  return if offset = 128 then some output else none

def decode (block : Nat) : Option (Array Nat) := do
  let mode ← (List.range 8).find? (fun mode => readBits block mode 1 = 1)
  if mode = 4 || mode = 5 then decodeDual mode block else decodePartitioned mode block

theorem interpolation_bound (a b width index : Nat) (ha : a ≤ 255) (hb : b ≤ 255)
    (hw : weight width index ≤ 64) : interpolate a b width index ≤ 255 := by
  have h := Nat.add_le_add (Nat.mul_le_mul_right (64-weight width index) ha)
    (Nat.mul_le_mul_right (weight width index) hb)
  have he : 255*(64-weight width index)+255*weight width index = 16320 := by
    rw [← Nat.mul_add, Nat.sub_add_cancel hw]
  dsimp only [interpolate]
  omega

theorem exact_equal_endpoints (a width index : Nat) (hw : weight width index ≤ 64) :
    interpolate a a width index = a := by
  simp only [interpolate, ← Nat.mul_add, Nat.sub_add_cancel hw]
  omega

theorem weights_bounded :
    (∀ i : Fin 4, weight 2 i.val ≤ 64) ∧
    (∀ i : Fin 8, weight 3 i.val ≤ 64) ∧
    (∀ i : Fin 16, weight 4 i.val ≤ 64) := by decide

theorem invalid_zero : decode 0 = none := by decide

def word (data : ByteArray) (offset size : Nat) : Nat :=
  (List.range size).foldl (fun n i => n+data[offset+i]!.toNat*2^(8*i)) 0

end GB300Math.TextureBC7

#print axioms GB300Math.TextureBC7.interpolation_bound
#print axioms GB300Math.TextureBC7.exact_equal_endpoints
#print axioms GB300Math.TextureBC7.weights_bounded
#print axioms GB300Math.TextureBC7.invalid_zero

def main (args : List String) : IO UInt32 := do
  let [input,output] := args | throw (IO.userError "Expected BC7 blocks and point RGBA output")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size = 4096*16 && ys.size = 65536*16 do throw (IO.userError "Invalid capture sizes")
  let mut bad := 0
  for block in [:4096] do
    let bits := GB300Math.TextureBC7.word xs (block*16) 16
    let some decoded := GB300Math.TextureBC7.decode bits | throw (IO.userError s!"Invalid BC7 block {block}")
    unless decoded.size = 64 do throw (IO.userError "Invalid decoder output size")
    for pixel in [:16] do
      let i := (block/64*4+pixel/4)*256 + block%64*4+pixel%4
      for component in [:4] do
        let expected := GB300Math.TextureRatio.ratioBits decoded[pixel*4+component]! 255
        let actual := GB300Math.TextureBC7.word ys ((i*4+component)*4) 4
        if actual != expected then
          if bad < 4 then IO.println s!"witness block={block} pixel={pixel} component={component} expected={expected} actual={actual}"
          bad := bad+1
  IO.println s!"result count=262144 mismatches={bad}"
  return if bad = 0 then 0 else 1
