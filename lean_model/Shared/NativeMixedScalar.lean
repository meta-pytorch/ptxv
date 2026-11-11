/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MixedScalarCore

namespace GB300Math.NativeMixedScalar

def modify (width mode word : Nat) : Nat :=
  let value := if mode % 2 = 1 then word % 2^(width-1) else word
  if mode / 2 = 1 then value ^^^ 2^(width-1) else value

def evaluate (precision op : String)
    (mode : FloatLib.Floats.Formats.BinaryInterchange.Model.IEEERoundingMode)
    (sat : Bool) (sa sb ma mb mc a b c : Nat) : Option Nat := do
  if sa ≥ 2 || sb ≥ 2 || ma ≥ 4 || mb ≥ 4 || mc ≥ 4 ||
      a ≥ 2^32 || b ≥ 2^32 || c ≥ 2^32 then none else do
    let x := modify 16 ma (a / 2^(16*sa) % 65536)
    let y := modify 16 mb (b / 2^(16*sb) % 65536)
    let z := modify 32 mc c
    MixedScalar.evaluate precision op mode sat x y z

theorem modify_identity (width word : Nat) : modify width 0 word = word := by
  simp [modify]

theorem high_half_selection (lo hi : Nat) (hl : lo < 65536) (hh : hi < 65536) :
    (hi*65536+lo)/2^(16*1)%65536 = hi := by omega

theorem absolute_negative_half (word : Nat) :
    modify 16 3 word = (word%32768) ^^^ 32768 := by rfl

#print axioms modify_identity
#print axioms high_half_selection
#print axioms absolute_negative_half

def read32 (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:4] do word := word + data[offset+j]!.toNat * 2^(8*j)
  return word

end GB300Math.NativeMixedScalar

open GB300Math.NativeMixedScalar GB300Math.MixedScalar

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected a TSV manifest")
  let lines ← IO.FS.readFile manifest
  let mut tasks := 0
  let mut failures := 0
  let mut records := 0
  for line in lines.splitOn "\n" do
    if line.isEmpty then continue
    let [name, precision, op, modeName, satName, saName, sbName, maName, mbName,
      mcName, input, ioName, output, ooName, countName, expectedName] := line.splitOn "\t" |
      throw (IO.userError s!"Malformed case: {line}")
    let some mode := rounding modeName | throw (IO.userError "Invalid rounding")
    let some params := [satName,saName,sbName,maName,mbName,mcName,ioName,ooName,countName,expectedName].mapM String.toNat? |
      throw (IO.userError "Invalid integer parameter")
    let [sat,sa,sb,ma,mb,mc,inputOffset,outputOffset,count,expected] := params |
      throw (IO.userError "Invalid parameter count")
    unless sat < 2 ∧ count > 0 do throw (IO.userError "Invalid saturation or row count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless inputOffset+count*16 ≤ xs.size ∧ outputOffset+count*16 ≤ ys.size do
      throw (IO.userError "Input/output slice out of bounds")
    let mut mismatches := 0
    for i in [:count] do
      let ix := inputOffset+16*i
      let iy := outputOffset+16*i
      let a := read32 xs ix
      let b := read32 xs (ix+4)
      let c := read32 xs (ix+8)
      unless read32 ys (iy+4) = a ∧ read32 ys (iy+8) = b ∧ read32 ys (iy+12) = c do
        throw (IO.userError s!"Input echo changed: {name} row={i}")
      let some value := GB300Math.NativeMixedScalar.evaluate precision op mode (sat=1) sa sb ma mb mc a b c |
        throw (IO.userError s!"Unsupported input/configuration: {name} row={i}")
      let actual := read32 ys iy
      if value != actual then
        if mismatches < 3 then IO.println s!"witness {name} row={i} actual={actual} expected={value}"
        mismatches := mismatches+1
    IO.println s!"result {name} cases={count} mismatches={mismatches} expected={expected}"
    if mismatches != expected then failures := failures+1
    tasks := tasks+1
    records := records+count
  unless tasks > 0 do throw (IO.userError "Empty manifest")
  IO.println s!"summary tasks={tasks} records={records} failures={failures}"
  return if failures=0 then 0 else 1
