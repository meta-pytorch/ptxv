/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.FswzaddGeneral

private def read32 (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut v := 0
  for j in [:4] do v := v + data[offset + j]!.toNat * 2 ^ (8 * j)
  return v

private def checkCase (args : List String) : IO UInt32 := do
  let [patternArg, aFile, bFile, outputFile] := args |
    throw (IO.userError "Expected pattern, A, B and output files")
  let some pattern := patternArg.toNat? |
    throw (IO.userError "Invalid pattern")
  unless pattern < 256 do
    throw (IO.userError "Unsupported FSWZADD pattern")
  let a ← IO.FS.readBinFile aFile
  let b ← IO.FS.readBinFile bFile
  let y ← IO.FS.readBinFile outputFile
  unless a.size > 0 ∧ a.size % 128 = 0 ∧ a.size = b.size ∧ a.size = y.size do
    throw (IO.userError "Expected equal nonempty arrays containing complete 32-lane warps")
  let mut failures := 0
  for i in [:a.size / 4] do
    let aa := read32 a (4 * i)
    let bb := read32 b (4 * i)
    let yy := read32 y (4 * i)
    let some expected := GB300Math.FswzaddGeneral.evaluate pattern i aa bb |
      throw (IO.userError "Unsupported pattern during replay")
    if expected ≠ yy then
      if failures < 4 then
        IO.println s!"mismatch row={i} A={aa} B={bb} expected={expected} actual={yy}"
      failures := failures + 1
  IO.println s!"records={a.size / 4} mismatches={failures}"
  return if failures = 0 then 0 else 1

def main (args : List String) : IO UInt32 := do
  match args with
  | ["--manifest", path] =>
    let text ← IO.FS.readFile path
    let mut cases := 0
    let mut failures := 0
    for line in text.splitOn "\n" do
      if line.isEmpty then continue
      cases := cases + 1
      let status ← checkCase (line.splitOn "\t")
      if status ≠ 0 then failures := failures + 1
    unless cases > 0 do throw (IO.userError "Empty manifest")
    IO.println s!"cases={cases} failed_cases={failures}"
    return if failures = 0 then 0 else 1
  | _ => checkCase args
