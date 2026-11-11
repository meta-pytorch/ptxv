/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TextureAnisotropyBlend
open GB300Math

def word (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat + data[4*i+1]!.toNat*256 +
    data[4*i+2]!.toNat*65536 + data[4*i+3]!.toNat*16777216

def main (args : List String) : IO UInt32 := do
  let [aText, nooptPath, defaultPath] := args |
    throw (IO.userError "Expected anisotropy noopt default")
  let some a := aText.toNat? | throw (IO.userError "Invalid anisotropy")
  unless [1,2,4,8,16].contains a do
    throw (IO.userError "Unsupported anisotropy")
  let noopt ← IO.FS.readBinFile nooptPath
  let default ← IO.FS.readBinFile defaultPath
  unless noopt.size=8192*32+64 ∧ default.size=8192*32+64 do
    throw (IO.userError "Invalid capture extent")
  let mut bad := 0
  for row in [:8192] do
    for channel in [:4] do
      let x := word noopt (row*8+channel)
      let observed := word default (row*8+channel)
      let q := (TextureMipmap.q8 x).toNat
      unless 256≤q ∧ q≤2304 ∧ TextureMipmap.encodeQ8 q=x do
        throw (IO.userError "Reference output is outside the exact positive Q8 domain")
      let predicted := TextureMipmap.encodeQ8 (256+TextureAnisotropyBlend.remap (q-256) a)
      if observed != predicted then bad := bad+1
    for data in [noopt, default] do
      if word data (row*8+4) != 1 then bad := bad+1
      for k in [5:8] do
        if word data (row*8+k) != 0xa5a5a5a5 then bad := bad+1
  for data in [noopt, default] do
    for k in [:16] do
      if word data (8192*8+k) != 0xa5a5a5a5 then bad := bad+1
  IO.println s!"rows=8192 numeric_relations=32768 state_words=65536 guard_words=32 differences={bad}"
  return if bad=0 then 0 else 1
