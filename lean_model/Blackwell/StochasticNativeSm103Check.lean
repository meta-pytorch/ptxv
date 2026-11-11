/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.StochasticNativeSm103
open GB300Math.StochasticSm103

def word (data : ByteArray) (offset width : Nat) : Nat :=
  (List.range width).foldl (fun v i => v + data[offset+i]!.toNat * 2^(8*i)) 0

def main (args : List String) : IO UInt32 := do
  let [name, relu, sat, input, output] := args | throw (IO.userError "Expected format relu sat inputs outputs")
  let format : Format ← match name with
    | "e4m3" => pure .e4m3 | "e5m2" => pure .e5m2 | "e2m3" => pure .e2m3
    | "e3m2" => pure .e3m2 | "e2m1" => pure .e2m1
    | _ => throw (IO.userError "Unsupported native RS format")
  unless relu ∈ ["0","1"] && sat ∈ ["0","1"] && (sat = "1" || name ∈ ["e4m3","e5m2"]) do
    throw (IO.userError "Unsupported native RS modifiers")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 && xs.size % 16 = 0 && ys.size = xs.size / 2 do
    throw (IO.userError "Invalid capture extent")
  let mut failures := 0
  for i in [:xs.size/16] do
    let a := word xs (i*16) 4
    let b := word xs (i*16+4) 4
    let c := word xs (i*16+8) 4
    let expected := GB300Math.StochasticNativeSm103.packed format (relu = "1") (sat = "1") a b c
    let actual := word ys (i*8) 8
    if expected != actual then
      if failures < 4 then IO.println s!"witness row={i} expected={expected} actual={actual}"
      failures := failures+1
  IO.println s!"result count={xs.size/16} mismatches={failures}"
  return if failures = 0 then 0 else 1
