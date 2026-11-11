/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import AtomicSpec

set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

namespace GB300Math.FloatAtomicTrace

open FloatLib.Floats.Formats.BinaryInterchange

def update (fmt : FloatFormat) (packed global : Bool) (state value : Nat) : Nat :=
  if packed then
    Atomic.add fmt global (state%65536) (value%65536) +
      65536*Atomic.add fmt global (state/65536) (value/65536)
  else Atomic.add fmt global state value

def linearize (fmt : FloatFormat) (packed global : Bool) (values old : Array Nat) (final : Nat) :
    Nat → Nat → Nat → Bool
  | 0, state, used => state = final && used = 15
  | fuel+1, state, used => (List.range 4).any fun lane =>
    if used &&& 2^lane ≠ 0 ∨ old[lane]! ≠ state then false
    else linearize fmt packed global values old final fuel
      (update fmt packed global state values[lane]!) (used+2^lane)

def check (fmt : FloatFormat) (packed global : Bool) (initial : Nat)
    (values old : Array Nat) (final : Nat) : Option (Bool × Bool) := do
  let width := if packed then 32 else fmt.bitWidth
  unless values.size = 4 ∧ old.size = 4 ∧ initial < 2^width ∧ final < 2^width ∧
    values.all (· < 2^width) ∧ old.all (· < 2^width) do none
  let joint := linearize fmt packed global values old final 4 initial 0
  if !packed then return (joint,joint)
  let low := linearize fmt false global (values.map (· % 65536)) (old.map (· % 65536))
    (final%65536) 4 (initial%65536) 0
  let high := linearize fmt false global (values.map (· / 65536)) (old.map (· / 65536))
    (final/65536) 4 (initial/65536) 0
  return (low && high,joint)

theorem rounded_history_one :
    check FloatFormat.binary32 false true 0 #[0x4b800000,0x3f800000,0xcb800000,0x3f800000]
      #[0,0x4b800000,0x4b800000,0] 0x3f800000 = some (true,true) := by decide +kernel
theorem rounded_history_two :
    check FloatFormat.binary32 false true 0 #[0x4b800000,0x3f800000,0xcb800000,0x3f800000]
      #[0,0,0x4b800000,0x3f800000] 0x40000000 = some (true,true) := by decide +kernel
theorem final_alone_is_insufficient :
    check FloatFormat.binary32 false true 0 #[0x4b800000,0x3f800000,0xcb800000,0x3f800000]
      #[0,0,0,0] 0x40000000 = some (false,false) := by decide +kernel
theorem packed_components_can_have_distinct_orders :
    check FloatFormat.binary16 true true 0 #[0x68006800,0x3c003c00,0xe800e800,0x3c003c00]
      #[0,0x00006800,0x68006800,0x3c000000] 0x40003c00 = some (true,false) := by decide +kernel

#print axioms rounded_history_one
#print axioms rounded_history_two
#print axioms final_alone_is_insufficient
#print axioms packed_components_can_have_distinct_orders

def run (manifest : String) : IO UInt32 := do
  let lines := (← IO.FS.readFile manifest).splitOn "\n" |>.filter (!·.isEmpty)
  unless !lines.isEmpty do throw (IO.userError "Empty atomic history manifest")
  let mut total := 0
  let mut errors := 0
  let mut separate := 0
  for line in lines do
    let [precision,space,input,output] := line.splitOn "\t" |
      throw (IO.userError "Malformed atomic history manifest")
    unless space = "global" ∨ space = "shared" do throw (IO.userError "Invalid address space")
    unless precision ∈ ["f32","f64","f16","bf16","f16x2","bf16x2"] do
      throw (IO.userError "Invalid precision")
    let some fmt := Atomic.format precision | throw (IO.userError "Invalid precision")
    let packed := precision.endsWith "x2"
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size%40 = 0 ∧ ys.size = xs.size do
      throw (IO.userError "Invalid atomic history record sizes")
    let mut failures := 0
    for i in [:xs.size/40] do
      let initial := Atomic.readWord xs (i*40)
      let values := (Array.range 4).map (fun j => Atomic.readWord xs (i*40+8+j*8))
      let old := (Array.range 4).map (fun j => Atomic.readWord ys (i*40+j*8))
      let final := Atomic.readWord ys (i*40+32)
      let some (valid,joint) := check fmt packed (space = "global") initial values old final |
        throw (IO.userError s!"Invalid atomic history at row {i}")
      if !valid then
        if failures < 3 then IO.println s!"Invalid history row={i} initial={initial} values={values} old={old} final={final}"
        failures := failures+1
      if valid && !joint then separate := separate+1
    IO.println s!"case {precision} {space} records={xs.size/40} mismatches={failures}"
    (← IO.getStdout).flush
    total := total+xs.size/40
    errors := errors+failures
  IO.println s!"total histories={total} mismatches={errors} separate_component_orders={separate}"
  return if errors = 0 then 0 else 1

end GB300Math.FloatAtomicTrace

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected atomic history manifest")
  GB300Math.FloatAtomicTrace.run manifest
