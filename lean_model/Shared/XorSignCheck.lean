/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.XorSign

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut value := 0
  for i in [:8] do
    value := value + data[offset+i]!.toNat*2^(8*i)
  return value

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected manifest")
  let rows ← IO.FS.readFile manifest
  let mut cases := 0
  let mut failures := 0
  for line in rows.splitOn "\n" do
    if line.isEmpty then continue
    let [fmt, op, ftz, nan, input, output, expected] := line.splitOn "\t" |
      throw (IO.userError "Invalid manifest row")
    unless [ftz,nan].all (fun x => x = "0" || x = "1") do
      throw (IO.userError "Invalid flag")
    let some expected := expected.toNat? | throw (IO.userError "Invalid mismatch count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 && xs.size % 32 = 0 && ys.size = xs.size/4 do
      throw (IO.userError "Invalid input/output extent")
    let mut differences := 0
    for i in [:xs.size/32] do
      let a := readWord xs (32*i)
      let b := readWord xs (32*i+8)
      let some value := GB300Math.XorSign.evaluate fmt op (ftz = "1") (nan = "1") a b |
        throw (IO.userError "Unsupported variant")
      let actual := readWord ys (8*i)
      if value != actual then
        if differences < 8 then IO.println s!"witness {fmt} {op} ftz={ftz} nan={nan} row={i} a={a} b={b} model={value} actual={actual}"
        differences := differences+1
    IO.println s!"result {output} records={xs.size/32} differences={differences} expected={expected}"
    if differences != expected then failures := failures+1
    cases := cases+1
  if cases = 0 then throw (IO.userError "Empty manifest")
  return if failures = 0 then 0 else 1
