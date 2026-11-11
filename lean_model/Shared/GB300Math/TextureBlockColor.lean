/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std.Tactic

namespace GB300Math.TextureBlockColor

def expand5 (n : Nat) : Nat := n * 8 + n / 4
def expand6 (n : Nat) : Nat := n * 4 + n / 16
def third5 (a b : Nat) : Nat := (2*a+b)*22/8
def midpoint5 (a b : Nat) : Nat := (a+b)*33/8
def interpolate6 (half : Bool) (a b : Nat) : Nat :=
  let x := expand6 a
  let delta : Int := (expand6 b : Int) - x
  ((256*x + delta.tdiv 4 + 128 + delta*(if half then 128 else 80)).tdiv 256).toNat

def channel (forceFour : Bool) (a b index component : Nat) : Nat :=
  let half := !forceFour && a ≤ b
  let shift := if component = 0 then 11 else if component = 1 then 5 else 0
  let width := if component = 1 then 64 else 32
  let x := a / 2^shift % width
  let y := b / 2^shift % width
  if half && index = 3 then 0
  else if index < 2 then
    if component = 1 then expand6 (if index = 0 then x else y)
    else expand5 (if index = 0 then x else y)
  else if component = 1 then
    if index = 2 then interpolate6 half x y else interpolate6 false y x
  else if half then midpoint5 x y
  else if index = 2 then third5 x y else third5 y x

def ratioBits (n d : Nat) : Nat := Id.run do
  if n = 0 then return 0
  let tentative := (n.log2 : Int) - (d.log2 : Int)
  let below := if tentative ≥ 0 then n < d*2^tentative.toNat
    else n*2^(-tentative).toNat < d
  let e := if below then tentative-1 else tentative
  let scaled := n*2^(23-e).toNat
  let q := scaled/d
  let rem := scaled%d
  let m := q + if 2*rem > d ∨ (2*rem = d ∧ q%2 = 1) then 1 else 0
  let (m,e) := if m ≥ 2^24 then (m/2,e+1) else (m,e)
  return (e+127).toNat*2^23 + m%2^23

def pixel (bc2 : Bool) (color alpha location component : Nat) : Nat :=
  let a := color % 65536
  let b := color / 65536 % 65536
  let index := color / 2^(32+2*location) % 4
  let value := if component < 3 then channel bc2 a b index component
    else if bc2 then alpha / 2^(4*location) % 16 * 17
    else if a ≤ b && index = 3 then 0 else 255
  ratioBits value 255

def coefficient (eight : Bool) (index : Nat) : Nat :=
  if eight then [0, 257, 36, 72, 113, 144, 185, 221][index]!
  else [0, 257, 48, 96, 161, 209, 0, 257][index]!

def blend16 (a b weight : Nat) : Nat := (257-weight)*a + weight*b

def alpha16 (block location : Nat) : Nat :=
  let a := block%256
  let b := block/256%256
  let index := block/2^(16+3*location)%8
  if a ≤ b && index = 6 then 0
  else if a ≤ b && index = 7 then 65535
  else blend16 a b (coefficient (a > b) index)

def alphaPixel (bc5 : Bool) (low high location component : Nat) : Nat :=
  if component = 0 then ratioBits (alpha16 low location) 65535
  else if component = 1 && bc5 then ratioBits (alpha16 high location) 65535
  else 0

def packedPixel (bits component : Nat) : Nat :=
  if component = 3 then ratioBits (bits/2^30%4) 3
  else ratioBits (bits/2^(10*component)%1024) 1023

theorem blend16_bound (a b weight : Nat) (ha : a ≤ 255) (hb : b ≤ 255)
    (hw : weight ≤ 257) : blend16 a b weight ≤ 65535 := by
  unfold blend16
  have h := Nat.add_le_add (Nat.mul_le_mul_left (257-weight) ha)
    (Nat.mul_le_mul_left weight hb)
  have he : (257-weight)*255 + weight*255 = 65535 := by
    rw [← Nat.add_mul, Nat.sub_add_cancel hw]
  omega

theorem coefficient_bound : ∀ eight : Bool, ∀ i : Fin 8,
    coefficient eight i.val ≤ 257 := by decide

theorem alpha_weight_deviation_witness :
    coefficient false 4 * 5 - 3 * 257 = 34 := by decide

theorem packed_white : ∀ i : Fin 4, packedPixel 0xffffffff i.val = 0x3f800000 := by decide

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem thirds5_tight_bound : ∀ a b : Fin 32,
    3 * third5 a.val b.val ≤ 2*expand5 a.val + expand5 b.val + 2 ∧
    2*expand5 a.val + expand5 b.val ≤ 3*third5 a.val b.val + 2 := by decide +kernel

set_option maxRecDepth 100000 in
set_option maxHeartbeats 0 in
theorem thirds6_tight_bound : ∀ a b : Fin 64,
    3 * interpolate6 false a.val b.val ≤ 2*expand6 a.val + expand6 b.val + 16 ∧
    2*expand6 a.val + expand6 b.val ≤ 3*interpolate6 false a.val b.val + 16 := by decide +kernel

theorem bounds_attained :
    3*third5 0 1 + 2 = 2*expand5 0 + expand5 1 ∧
    3*interpolate6 false 0 61 + 16 = 2*expand6 0 + expand6 61 := by decide

theorem captured_green_witness :
    channel false 0xc5ff 0xb0bb 2 1 = 137 ∧
    channel false 0xc5ff 0xb0bb 3 1 = 73 := by decide

def word (data : ByteArray) (offset size : Nat) : Nat :=
  (List.range size).foldl (fun n i => n + data[offset+i]!.toNat*2^(8*i)) 0

end GB300Math.TextureBlockColor

#print axioms GB300Math.TextureBlockColor.thirds5_tight_bound
#print axioms GB300Math.TextureBlockColor.thirds6_tight_bound
#print axioms GB300Math.TextureBlockColor.bounds_attained
#print axioms GB300Math.TextureBlockColor.captured_green_witness
#print axioms GB300Math.TextureBlockColor.blend16_bound
#print axioms GB300Math.TextureBlockColor.coefficient_bound
#print axioms GB300Math.TextureBlockColor.alpha_weight_deviation_witness
#print axioms GB300Math.TextureBlockColor.packed_white

def main (args : List String) : IO UInt32 := do
  let [format,input,output] := args | throw (IO.userError "Expected bc1/bc2 blocks and point output")
  unless format ∈ ["bc1", "bc2", "bc4u", "bc5u", "rgb10a2"] do
    throw (IO.userError "Unsupported block format")
  let bc2 := format == "bc2"
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  let size := if bc2 || format = "bc5u" then 16 else 8
  let expectedInput := if format = "rgb10a2" then 65536*4 else 4096*size
  unless xs.size = expectedInput && ys.size = 65536*16 do throw (IO.userError "Invalid block capture sizes")
  let mut bad := 0
  for i in [:65536] do
    let x := i%256
    let y := i/256
    let offset := (y/4*64+x/4)*size
    let alpha := GB300Math.TextureBlockColor.word xs offset 8
    let color := GB300Math.TextureBlockColor.word xs (offset + if bc2 then 8 else 0) 8
    let high := if format = "bc5u" then GB300Math.TextureBlockColor.word xs (offset+8) 8 else 0
    let packed := if format = "rgb10a2" then GB300Math.TextureBlockColor.word xs (i*4) 4 else 0
    let location := y%4*4+x%4
    for component in [:4] do
      let expected := if format = "rgb10a2" then GB300Math.TextureBlockColor.packedPixel packed component
        else if format = "bc4u" || format = "bc5u" then
          GB300Math.TextureBlockColor.alphaPixel (format == "bc5u") alpha high location component
        else GB300Math.TextureBlockColor.pixel bc2 color alpha location component
      let actual := GB300Math.TextureBlockColor.word ys ((i*4+component)*4) 4
      if actual != expected then
        if bad < 4 then IO.println s!"witness pixel={i} component={component} expected={expected} actual={actual}"
        bad := bad+1
  IO.println s!"result count=262144 mismatches={bad}"
  return if bad = 0 then 0 else 1
