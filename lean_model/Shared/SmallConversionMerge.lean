/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.SmallConversions

namespace GB300Math.SmallConversionMerge

open GB300Math.SmallConversions

def merge (upper : Bool) (converted payload : Nat) : Nat :=
  if upper then converted % 65536 * 65536 + payload % 65536
  else converted % 65536 + payload % 65536 * 65536

theorem low_conversion_preserved (converted payload : Nat) :
    merge false converted payload % 65536 = converted % 65536 := by
  simp [merge, Nat.add_mod]

theorem high_payload_preserved (converted payload : Nat) :
    merge false converted payload / 65536 = payload % 65536 := by
  have h := Nat.mod_lt converted (by decide : 0 < 65536)
  simp only [merge, Bool.false_eq_true, ↓reduceIte]
  omega

theorem output_bound (upper : Bool) (converted payload : Nat) :
    merge upper converted payload < 2^32 := by
  have hc := Nat.mod_lt converted (by decide : 0 < 65536)
  have hp := Nat.mod_lt payload (by decide : 0 < 65536)
  cases upper <;> simp only [merge, Bool.false_eq_true, ↓reduceIte] <;> omega

#print axioms low_conversion_preserved
#print axioms high_payload_preserved
#print axioms output_bound

end GB300Math.SmallConversionMerge

open GB300Math.SmallConversions GB300Math.SmallConversionMerge

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected merge manifest")
  let rows := (← IO.FS.readFile manifest).splitOn "\n" |>.filter (! ·.isEmpty)
  unless !rows.isEmpty do throw (IO.userError "Empty manifest")
  let mut failed := 0
  for row in rows do
    let [fmt, other, mode, upper, input, output, expected] := row.splitOn "\t" |
      throw (IO.userError "Malformed merge row")
    unless upper = "0" ∨ upper = "1" do throw (IO.userError "Invalid merge position")
    let some expected := expected.toNat? | throw (IO.userError "Invalid mismatch count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 16 = 0 ∧ ys.size = xs.size/2 do
      throw (IO.userError "Invalid record sizes")
    let mut errors := 0
    for i in [:xs.size/16] do
      let a := readWord xs (16*i) 4
      let b := readWord xs (16*i+4) 4
      let c := readWord xs (16*i+8) 4
      let some converted := evaluate "down" fmt other mode false true false a b 0 |
        throw (IO.userError "Unsupported conversion")
      let target := merge (upper = "1") converted c
      let actual := readWord ys (8*i) 8
      if actual ≠ target then
        if errors < 5 then IO.println s!"witness row={i} expected={target} actual={actual}"
        errors := errors+1
    IO.println s!"result {output} records={xs.size/16} mismatches={errors} expected={expected}"
    if errors ≠ expected then failed := failed+1
  return if failed = 0 then 0 else 1
