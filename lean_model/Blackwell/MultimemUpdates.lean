/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import BulkReductionSpec

set_option maxRecDepth 100000
set_option maxHeartbeats 2000000
namespace GB300Math.MultimemUpdates
open FloatLib.Floats.Formats.BinaryInterchange

def width (precision : String) : Nat :=
  if precision.endsWith "x4" then 32
  else if precision.endsWith "x2" then if precision.startsWith "e" then 16 else 32
  else if precision.startsWith "e" then 8
  else if precision = "f16" || precision = "bf16" then 16
  else if precision.endsWith "128" then 128
  else if precision.endsWith "64" then 64 else 32

def base (precision : String) : String :=
  match precision with
  | "f16x2" => "f16"
  | "bf16x2" => "bf16"
  | "e4m3x2" | "e4m3x4" => "e4m3"
  | "e5m2x2" | "e5m2x4" => "e5m2"
  | _ => precision

def floatType (precision : String) : Bool := precision ∈ ["f16", "f16x2", "bf16", "bf16x2", "f32", "f64",
  "e4m3", "e4m3x2", "e4m3x4", "e5m2", "e5m2x2", "e5m2x4"]

def legal (kind op precision : String) (vec : Nat) (noftz : Bool) : Bool :=
  let total := width precision * vec
  if kind = "bulk" then total = 128 &&
    (if op = "copy" then precision = "b128" && !noftz
     else BulkReduction.legal "global" op precision noftz)
  else if noftz then false
  else if kind = "st" || kind = "red" then
    (if floatType precision then
       (total ∈ [32,64,128]) && (vec ∈ [1,2,4,8]) && (precision ≠ "f64" || vec = 1) &&
       (if kind = "st" then op = "copy" else !precision.startsWith "e" && op = "add")
     else vec = 1 &&
       (if kind = "st" then precision ∈ ["u32","s32","u64","s64","b32","b64"] && op = "copy"
        else if precision ∈ ["b32","b64"] then op ∈ ["and","or","xor"]
        else if precision = "s64" then op ∈ ["min","max"]
        else precision ∈ ["u32","s32","u64"] && op ∈ ["add","min","max"]))
  else false

def evaluate (kind op precision : String) (a b : Nat) : Option Nat := do
  if kind = "st" || op = "copy" then return b
  let scalar := base precision
  if kind = "red" && (scalar = "f32" || scalar = "f64") then
    return Atomic.add (if scalar = "f32" then FloatFormat.binary32 else FloatFormat.binary64) true a b
  BulkReduction.evaluate op scalar a b

def output (kind op precision : String) (vec lane : Nat) (a b : Nat) : Option Nat :=
  if lane < (width precision * vec) / width (base precision) then evaluate kind op precision a b else some a

theorem untouched_component (kind op precision : String) (vec lane a b : Nat)
    (h : (width precision * vec) / width (base precision) ≤ lane) :
    output kind op precision vec lane a b = some a := by simp [output, Nat.not_lt.mpr h]

theorem copy_exact (precision : String) (a b : Nat) : evaluate "st" "copy" precision a b = some b := by
  simp [evaluate]

theorem red_f32_subnormal_flushes : evaluate "red" "add" "f32" 1 1 = some 0 := by decide +kernel

theorem bulk_f32_subnormal_preserved : evaluate "bulk" "add" "f32" 1 1 = some 2 := by decide +kernel

#print axioms untouched_component
#print axioms copy_exact
#print axioms red_f32_subnormal_flushes
#print axioms bulk_f32_subnormal_preserved

def read (data : ByteArray) (offset size : Nat) : Nat := Id.run do
  let mut value := 0
  for j in [:size] do value := value + data[offset+j]!.toNat * 2^(8*j)
  return value
end GB300Math.MultimemUpdates
open GB300Math.MultimemUpdates

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected multicast-update manifest")
  let lines := (← IO.FS.readFile manifest).splitOn "\n" |>.filter (! ·.isEmpty)
  unless !lines.isEmpty do throw (IO.userError "Empty multicast-update manifest")
  let mut failed := 0
  for line in lines do
    let [kind, op, precision, vecText, noftz, input, result, expectedText] := line.splitOn "\t" |
      throw (IO.userError "Malformed multicast-update row")
    let some vec := vecText.toNat? | throw (IO.userError "Invalid vector width")
    let some expected := expectedText.toNat? | throw (IO.userError "Invalid mismatch count")
    unless noftz = "0" ∨ noftz = "1" do throw (IO.userError "Invalid noftz flag")
    unless legal kind op precision vec (noftz = "1") do throw (IO.userError "Illegal multicast-update form")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile result
    unless xs.size > 0 ∧ xs.size % 32 = 0 ∧ ys.size = xs.size/2 do
      throw (IO.userError "Invalid multicast-update record sizes")
    let bytes := width (base precision) / 8
    let mut errors := 0
    for i in [:xs.size/32] do
      let mut matched := true
      for lane in [:16/bytes] do
        let a := read xs (32*i+bytes*lane) bytes
        let b := read xs (32*i+16+bytes*lane) bytes
        let some target := output kind op precision vec lane a b |
          throw (IO.userError "Unsupported multicast scalar")
        if read ys (16*i+bytes*lane) bytes ≠ target then matched := false
      if !matched then
        if errors < 4 then IO.println s!"witness {kind} {op} {precision} row={i}"
        errors := errors+1
    IO.println s!"result {result} records={xs.size/32} mismatches={errors} expected={expected}"
    if errors ≠ expected then failed := failed+1
  return if failed = 0 then 0 else 1
