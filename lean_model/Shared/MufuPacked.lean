/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.Mufu

namespace GB300Math.MufuPacked

def modify (modifier input : Nat) : Nat :=
  let word := input % 65536
  let value := if modifier / 2 % 2 = 1 then word % 32768 else word
  if modifier % 2 = 1 then value ^^^ 32768 else value

def half (modifier input : Nat) : Nat :=
  Mufu.narrow16 (Mufu.ex2 (Mufu.widen16 (modify modifier input)))

def evaluate (modifier input : Nat) : Nat :=
  half modifier input + half modifier (input / 65536) * 65536

theorem packed_low (lo hi : Nat) (h : lo < 65536) :
    (lo + hi * 65536) % 65536 = lo := by omega

theorem packed_high (lo hi : Nat) (h : lo < 65536) :
    (lo + hi * 65536) / 65536 = hi := by omega

theorem both_zero : evaluate 0 0 = 0x3c003c00 := by decide +kernel
theorem distinct_lanes : evaluate 0 0xbc003c00 = 0x38004000 := by decide +kernel
theorem negate_both : evaluate 1 0xbc003c00 = 0x40003800 := by decide +kernel
theorem absolute_both : evaluate 2 0xbc003c00 = 0x40004000 := by decide +kernel

#print axioms packed_low
#print axioms packed_high
#print axioms both_zero
#print axioms distinct_lanes
#print axioms negate_both
#print axioms absolute_both

def readWord (data : ByteArray) (offset : Nat) : Nat :=
  (List.range 8).foldl (fun word i => word + data[offset+i]!.toNat * 2^(8*i)) 0

end GB300Math.MufuPacked

open GB300Math.MufuPacked

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected packed MUFU manifest")
  let rows := (← IO.FS.readFile manifest).splitOn "\n" |>.filter (! ·.isEmpty)
  unless !rows.isEmpty do throw (IO.userError "Empty manifest")
  let mut failed := 0
  for row in rows do
    let [modifier, input, actualOutput, expected] := row.splitOn "\t" |
      throw (IO.userError "Malformed packed MUFU row")
    let some modifier := modifier.toNat? | throw (IO.userError "Invalid modifier")
    unless modifier < 4 do throw (IO.userError "Unsupported modifier")
    let some expected := expected.toNat? | throw (IO.userError "Invalid mismatch count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile actualOutput
    unless xs.size > 0 ∧ xs.size % 8 = 0 ∧ ys.size = xs.size do
      throw (IO.userError "Invalid record sizes")
    let mut errors := 0
    for i in [:xs.size/8] do
      let a := readWord xs (8*i)
      let target := evaluate modifier (a % 2^32)
      let actual := readWord ys (8*i)
      if actual ≠ target then
        if errors < 5 then IO.println s!"witness row={i} a={a} expected={target} actual={actual}"
        errors := errors+1
    IO.println s!"result {actualOutput} records={xs.size/8} mismatches={errors} expected={expected}"
    if errors ≠ expected then failed := failed+1
  return if failed = 0 then 0 else 1
