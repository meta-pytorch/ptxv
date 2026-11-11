/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureDepthCompare

namespace GB300Math.TextureDepthGeometry

open TextureDepthCompare

def interior (word : Nat) : Bool :=
  [0x3e000000, 0x3ec00000, 0x3f200000, 0x3f600000].contains word

def layer (word : Nat) : Option Nat :=
  if word = 0 then some 0 else if word = 0x3f800000 then some 1 else none

def face (x y z : Nat) : Option Nat :=
  let a := magnitude x
  let b := magnitude y
  let c := magnitude z
  if a = 0x3f800000 && b < a && c < a then some (x/2^31)
  else if b = 0x3f800000 && a < b && c < b then some (2+y/2^31)
  else if c = 0x3f800000 && a < c && b < c then some (4+z/2^31)
  else none

def payload (geometry x y z w : Nat) : Option Nat := do
  match geometry with
  | 0 => if interior x then some 0x3e000000 else none
  | 1 =>
    if !interior x then none else do
      let slot ← layer y
      some (if slot = 0 then 0x3e000000 else 0x3f400000)
  | 2 =>
    if !interior x || !interior y then none else do
      let slot ← layer z
      some (if slot = 0 then 0x3e000000 else 0x3f400000)
  | 3 =>
    let slot ← face x y z
    some (encodeSixteenths (2*(slot+1)))
  | 4 =>
    let slot ← face x y z
    let index ← layer w
    some (encodeSixteenths (slot+index*6+1))
  | _ => none

def result (function reference depth : Nat) : Nat :=
  if compare function reference depth then 0x3f800000 else 0

def direction (side : Nat) (u v : Nat) : Nat × Nat × Nat :=
  match side with
  | 0 => (0x3f800000,u,v)
  | 1 => (0xbf800000,u,v)
  | 2 => (u,0x3f800000,v)
  | 3 => (u,0xbf800000,v)
  | 4 => (u,v,0x3f800000)
  | _ => (u,v,0xbf800000)

def minor (i : Fin 4) : Nat :=
  #[0xbf400000,0xbe800000,0x3e800000,0x3f400000][i.val]!

theorem face_recovery (side : Fin 6) (u v : Fin 4) :
    let (x,y,z) := direction side.val (minor u) (minor v)
    face x y z = some side.val := by
  revert side u v
  decide +kernel

theorem cube_payloads_distinct :
    ((List.range 6).map fun i => encodeSixteenths (2*(i+1))).Pairwise (· ≠ ·) := by
  decide +kernel

theorem cube_array_payloads_distinct :
    ((List.range 12).map fun i => encodeSixteenths (i+1)).Pairwise (· ≠ ·) := by
  decide +kernel

theorem comparison_binary (function reference depth : Nat) :
    result function reference depth = 0 ∨
    result function reference depth = 0x3f800000 := by
  unfold result
  split <;> simp_all

end GB300Math.TextureDepthGeometry
