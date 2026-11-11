/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.TextureOffsets

def word (bytes : ByteArray) (index : Nat) : Nat :=
  (List.range 4).foldl (fun v i => v + bytes[index*4+i]!.toNat * 2^(8*i)) 0

def signed (n : Nat) : Int := if n < 2^31 then n else (n : Int)-2^32

def floatFloor (bits extra : Nat) : Int :=
  let e := bits / 2^23 % 256
  let m := bits % 2^23 + if e = 0 then 0 else 2^23
  let shift := (if e = 0 then 1 else e) + extra
  let numerator := if shift ≥ 150 then m * 2^(shift-150) else m
  let denominator := if shift ≥ 150 then 1 else 2^(150-shift)
  if bits / 2^31 = 1 then -((numerator+denominator-1)/denominator : Nat)
  else numerator/denominator

def clampIndex (extent : Nat) (i : Int) : Nat := min (extent-1) i.toNat

theorem clamp_bounded (extent : Nat) (i : Int) (h : 0 < extent) :
    clampIndex extent i < extent := by
  have := Nat.min_le_left (extent-1) i.toNat
  simp only [clampIndex]
  omega

theorem negative_clamps (extent : Nat) (i : Int) (h : i < 0) :
    clampIndex extent i = 0 := by
  have hi : i.toNat = 0 := Int.toNat_eq_zero.mpr (by omega)
  simp [clampIndex, hi]

def fetch (data : ByteArray) (integer : Bool) (w h d layer : Nat)
    (x y z : Int) (component : Nat) : Nat :=
  if integer && (x < 0 || y < 0 || z < 0 || x ≥ w || y ≥ h || z ≥ d) then 0
  else
    let index := clampIndex w x + w*(clampIndex h y + h*(clampIndex d z + d*layer))
    word data (index*4+component)

end GB300Math.TextureOffsets

#print axioms GB300Math.TextureOffsets.clamp_bounded
#print axioms GB300Math.TextureOffsets.negative_clamps

open GB300Math.TextureOffsets

def main (args : List String) : IO UInt32 := do
  let [geometry, ctype, component, input, texels, output] := args |
    throw (IO.userError "Expected geometry, coordinate type, component, inputs, texels, outputs")
  unless geometry ∈ ["1d","2d","3d","a1d","a2d"] && ctype ∈ ["s32","f32"] do
    throw (IO.userError "Unsupported geometry or coordinate type")
  let some component := component.toInt? | throw (IO.userError "Invalid component")
  unless component ≥ -1 && component < 4 && (component < 0 || (ctype = "f32" && geometry ∈ ["2d","a2d"])) do
    throw (IO.userError "Unsupported gather")
  let w := 7
  let h := if geometry ∈ ["1d","a1d"] then 1 else 5
  let d := if geometry = "3d" then 3 else 1
  let layers := if geometry ∈ ["a1d","a2d"] then 3 else 1
  let xs ← IO.FS.readBinFile input
  let ts ← IO.FS.readBinFile texels
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 && xs.size % 32 = 0 && xs.size = ys.size && ts.size = w*h*d*layers*16 do
    throw (IO.userError "Invalid capture extent")
  let mut failures := 0
  for i in [:xs.size/32] do
    let layer := word xs (8*i+3)
    unless layer < layers do throw (IO.userError "Layer outside tested domain")
    let mut coordinates : Array Int := #[]
    for axis in [:3] do
      let coordinate := word xs (8*i+axis)
      let offset := signed (word xs (8*i+4+axis))
      unless -8 ≤ offset && offset ≤ 7 do throw (IO.userError "Offset outside PTX domain")
      unless ctype = "s32" || coordinate / 2^23 % 256 < 255 do
        throw (IO.userError "Nonfinite coordinate outside this fixture")
      let pos := if ctype = "s32" then signed coordinate else
        if component ≥ 0 then (floatFloor coordinate 1 - 1)/2 else floatFloor coordinate 0
      coordinates := coordinates.push (pos + if geometry = "1d" && ctype = "s32" then 0 else offset)
    for lane in [:8] do
      let expected := if lane ≥ 4 then if lane = 4 then 1 else 0 else
        let dx := if component ≥ 0 && lane ∈ [1,2] then 1 else 0
        let dy := if component ≥ 0 && lane ∈ [0,1] then 1 else 0
        fetch ts (ctype = "s32") w h d layer
          (coordinates[0]!+dx) (if h = 1 then 0 else coordinates[1]!+dy)
          (if d = 1 then 0 else coordinates[2]!) (if component < 0 then lane else component.toNat)
      let actual := word ys (8*i+lane)
      if expected != actual then
        if failures < 5 then IO.println s!"witness row={i} lane={lane} expected={expected} actual={actual}"
        failures := failures+1
  IO.println s!"result count={xs.size/4} mismatches={failures}"
  return if failures = 0 then 0 else 1
