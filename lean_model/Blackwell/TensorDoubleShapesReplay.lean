/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.TensorDoubleShapes
open GB300Math.TensorDoubleShapes

def main (args : List String) : IO UInt32 := do
  let [mode,kText,input,output,expected] := args | throw (IO.userError "Expected mode k inputs outputs expected-mismatches")
  unless mode ∈ ["rn","rz","rm","rp"] do throw (IO.userError "Invalid rounding mode")
  let some expected := expected.toNat? | throw (IO.userError "Invalid expected mismatch count")
  let some k := kText.toNat? | throw (IO.userError "Invalid K")
  unless k ∈ [4,8,16] do throw (IO.userError "Unsupported shape")
  let stride := 8*(24*k+128)
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 ∧ xs.size % stride = 0 ∧ ys.size = xs.size / stride * 1024 do
    throw (IO.userError "Incorrect M16 N8 capture extent")
  let mut errors := 0
  for sample in [:xs.size/stride] do
    let base := sample*stride
    for row in [:16] do
      let a := (Array.range k).map (fun j => readWord xs (base+8*(row*k+j)))
      for col in [:8] do
        let b := (Array.range k).map (fun j => readWord xs (base+8*(16*k+col*k+j)))
        let c := readWord xs (base+8*(24*k+row*8+col))
        let model := dot mode a b c
        let actual := readWord ys ((sample*128+row*8+col)*8)
        if model ≠ actual then
          if errors < 8 then IO.println s!"witness sample={sample} row={row} col={col} a={a} b={b} c={c} model={model} actual={actual}"
          errors := errors+1
  IO.println s!"result mode={mode} cells={xs.size/stride*128} mismatches={errors} expected={expected}"
  return if errors = expected then 0 else 1
