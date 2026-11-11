/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.MufuSm103
import GB300Math.Fp64Seed
import GB300Math.TensorAccumulatorBlackwell

namespace PtxvCheck

def evaluate (op : String) (x : Nat) : Option Nat :=
  if op = "rcp64h" then some (GB300Math.Fp64Seed.highWord false x)
  else if op = "rsq64h" then some (GB300Math.Fp64Seed.highWord true x)
  else if op.endsWith ".ftz" then
    GB300Math.MufuSm103.evaluate (op.dropEnd 4).toString x
  else if op.endsWith ".raw" then
    GB300Math.MufuSm103.evaluate (op.dropEnd 4).toString x
  else if op = "tanh" || op.endsWith ".f16" || op.endsWith ".bf16" then
    GB300Math.MufuSm103.evaluate op x
  else none

def wordAt (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:4] do word := word + data[offset + j]!.toNat * 2^(8*j)
  return word

def check (args : List String) : IO UInt32 := do
  let [op, input, output] := args | throw (IO.userError "Expected OP INPUT.u32 OUTPUT.u32")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  let stride := if op = "mma.bf16" then 132 else 4
  unless xs.size > 0 && xs.size % stride = 0 && ys.size = xs.size / stride * 4 do
    throw (IO.userError "Invalid byte lengths")
  for i in [:xs.size / stride] do
    let x := wordAt xs (i*stride)
    let expected := if op = "mma.bf16" then
        GB300Math.TensorAccumulatorBlackwell.evaluate "bf16"
          ((Array.range 16).map (fun j => wordAt xs (i*stride+4+4*j)))
          ((Array.range 16).map (fun j => wordAt xs (i*stride+68+4*j))) x 0 0
      else evaluate op x
    let some expected := expected | throw (IO.userError s!"Unsupported {op}: row={i}")
    let actual := wordAt ys (i*4)
    unless actual = expected do
      throw (IO.userError s!"Mismatch {op}: row={i} input={x} Lean={expected} Rust={actual}")
  IO.println s!"{op}: {xs.size / stride} words match"
  return 0

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem bf16_underflow_zero :
    GB300Math.TensorAccumulatorBlackwell.evaluate "bf16"
      (#[0x16b2] ++ Array.replicate 15 0)
      (#[0x96e5] ++ Array.replicate 15 0) 0 0 0 = some 0 := by
  decide +kernel

#print axioms bf16_underflow_zero

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem bf16_overflow_infinity :
    GB300Math.TensorAccumulatorBlackwell.evaluate "bf16"
      (#[0x7f00] ++ Array.replicate 15 0)
      (#[0x4000] ++ Array.replicate 15 0) 0 0 0 = some 0x7f800000 := by
  decide +kernel

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem bf16_negative_overflow_infinity :
    GB300Math.TensorAccumulatorBlackwell.evaluate "bf16"
      (#[0xff00] ++ Array.replicate 15 0)
      (#[0x4000] ++ Array.replicate 15 0) 0 0 0 = some 0xff800000 := by
  decide +kernel

set_option maxRecDepth 100000 in
set_option maxHeartbeats 1000000 in
theorem bf16_large_products_cancel :
    GB300Math.TensorAccumulatorBlackwell.evaluate "bf16"
      (#[0x7f00, 0xff00] ++ Array.replicate 14 0)
      (#[0x7f00, 0x7f00] ++ Array.replicate 14 0) 0 0 0 = some 0 := by
  decide +kernel

#print axioms bf16_overflow_infinity
#print axioms bf16_negative_overflow_infinity
#print axioms bf16_large_products_cancel

end PtxvCheck

def main := PtxvCheck.check
