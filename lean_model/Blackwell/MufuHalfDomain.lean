/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.Mufu
import GB300Math.MufuSm103

set_option maxRecDepth 1000000
set_option maxHeartbeats 0

namespace GB300Math.MufuHalfDomain

def modify (modifier input : Nat) : Nat :=
  let word := input % 65536
  let magnitude := if modifier / 2 % 2 = 1 then word % 32768 else word
  if modifier % 2 = 1 then magnitude ^^^ 32768 else magnitude

def evaluate (precision op : String) (input : Nat) : Option Nat :=
  if precision = "f16" then Mufu.evaluate (op ++ ".f16") input
  else if precision = "bf16" then MufuSm103.evaluateBf16 op input
  else none

def controlIndex (modifier : Nat) : Nat :=
  if modifier = 3 then 0xb555 else 0x3555

def preimages (modifier : Nat) : List Nat :=
  if modifier = 0 then [0x3555]
  else if modifier = 1 then [0xb555]
  else [0x3555, 0xb555]

def expected (modifier selector : Nat) (corrupt : Bool) (low : Nat) : Nat :=
  if !corrupt then 0
  else if selector = 0 then
    if (preimages modifier).contains low then 65536 * 2^32 + low else 0
  else (preimages modifier).length * 2^32 + (preimages modifier).head! * 65536 + low

theorem split_covers (word : Nat) (h : word < 2^32) :
    word % 65536 + (word / 65536) * 65536 = word ∧
    word % 65536 < 65536 ∧ word / 65536 < 65536 := by omega

theorem selected_high (low high : Nat) (h : low < 65536) :
    (low + high * 65536) / 65536 = high := by omega

theorem control_witnesses :
    modify 0 0x3555 = controlIndex 0 ∧
    modify 1 0xb555 = controlIndex 1 ∧
    modify 2 0xb555 = controlIndex 2 ∧
    modify 3 0x3555 = controlIndex 3 := by decide +kernel

#print axioms split_covers
#print axioms selected_high
#print axioms control_witnesses

end GB300Math.MufuHalfDomain

open GB300Math.MufuHalfDomain

def readWord (data : ByteArray) (offset : Nat) : Nat :=
  (List.range 8).foldl (fun result i => result + data[offset+i]!.toNat * 2^(8*i)) 0

def main (args : List String) : IO UInt32 := do
  if let ["export", out] := args then
    for precision in ["f16", "bf16"] do
      for op in ["ex2", "rcp", "rsq", "sqrt", "lg2", "sin", "cos", "tanh"] do
        let mut data := ByteArray.empty
        for input in [:65536] do
          let some value := evaluate precision op input | throw (IO.userError "Unsupported table")
          unless value < 65536 do throw (IO.userError "Non-half output")
          data := data.push (UInt8.ofNat value)
          data := data.push (UInt8.ofNat (value / 256))
        IO.FS.writeBinFile (out ++ s!"/{precision}_{op}.bin") data
    return 0
  let [manifest] := args | throw (IO.userError "Expected export directory or replay manifest")
  let mut failures := 0
  for row in (← IO.FS.readFile manifest).splitOn "\n" |>.filter (! ·.isEmpty) do
    let [modifier, selector, corrupt, output, expectedErrors] := row.splitOn "\t" |
      throw (IO.userError "Malformed scan row")
    let some modifier := modifier.toNat? | throw (IO.userError "Invalid modifier")
    let some selector := selector.toNat? | throw (IO.userError "Invalid selector")
    let some corrupt := corrupt.toNat? | throw (IO.userError "Invalid corruption flag")
    let some expectedErrors := expectedErrors.toNat? | throw (IO.userError "Invalid error count")
    unless modifier < 4 ∧ selector < 2 ∧ corrupt < 2 do throw (IO.userError "Invalid scan form")
    let data ← IO.FS.readBinFile output
    unless data.size = 65536 * 8 do throw (IO.userError "Incomplete scan")
    let mut errors := 0
    for low in [:65536] do
      let actual := readWord data (8*low)
      let target := expected modifier selector (corrupt = 1) low
      if actual ≠ target then
        if errors < 3 then IO.println s!"witness low={low} actual={actual} expected={target}"
        errors := errors + 1
    IO.println s!"records=65536 mismatches={errors} expected={expectedErrors}"
    if errors ≠ expectedErrors then failures := failures + 1
  return if failures = 0 then 0 else 1
