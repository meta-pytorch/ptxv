/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std.Tactic

namespace GB300Math.MinMax3Redux

def canonicalNaN : Nat := 0x7fffffff

def prepare (ftz absolute : Bool) (bits : Nat) : Nat :=
  let x := bits % 2^32
  let magnitude := x % 2^31
  let y := if ftz && magnitude < 0x00800000 then x - magnitude else x
  if absolute then y % 2^31 else y

def isNaN (x : Nat) : Bool := x % 2^31 > 0x7f800000

def orderKey (x : Nat) : Nat :=
  if x ≥ 2^31 then 2^32 - 1 - x else x + 2^31

def fromKey (key : Nat) : Nat :=
  if key ≥ 2^31 then key - 2^31 else 2^32 - 1 - key

theorem fromKey_orderKey (x : Nat) (h : x < 2^32) :
    fromKey (orderKey x) = x := by
  unfold orderKey fromKey
  split <;> split <;> omega

def chooseKey (minimum : Bool) : Option Nat → Option Nat → Option Nat
  | none, b => b
  | a, none => a
  | some a, some b => some (if minimum then min a b else max a b)

theorem chooseKey_comm (minimum : Bool) (a b : Option Nat) :
    chooseKey minimum a b = chooseKey minimum b a := by
  cases a <;> cases b <;> cases minimum <;>
    simp [chooseKey, Nat.min_comm, Nat.max_comm]

theorem chooseKey_assoc (minimum : Bool) (a b c : Option Nat) :
    chooseKey minimum (chooseKey minimum a b) c =
      chooseKey minimum a (chooseKey minimum b c) := by
  cases a <;> cases b <;> cases c <;> cases minimum <;>
    simp [chooseKey, Nat.min_assoc, Nat.max_assoc]

structure Selected where
  key : Option Nat
  nanSeen : Bool
  deriving DecidableEq, Repr

def join (minimum : Bool) (a b : Selected) : Selected :=
  ⟨chooseKey minimum a.key b.key, a.nanSeen || b.nanSeen⟩

theorem join_comm (minimum : Bool) (a b : Selected) :
    join minimum a b = join minimum b a := by
  cases a; cases b
  simp [join, chooseKey_comm, Bool.or_comm]

theorem join_assoc (minimum : Bool) (a b c : Selected) :
    join minimum (join minimum a b) c = join minimum a (join minimum b c) := by
  cases a; cases b; cases c
  simp [join, chooseKey_assoc, Bool.or_assoc]

def wordState (ftz absolute : Bool) (bits : Nat) : Selected :=
  let word := prepare ftz absolute bits
  if isNaN word then ⟨none, true⟩ else ⟨some (orderKey word), false⟩

def finish (propagateNaN : Bool) (state : Selected) : Nat :=
  if propagateNaN && state.nanSeen then canonicalNaN
  else match state.key with
    | none => canonicalNaN
    | some key => fromKey key

def minmax3 (minimum ftz propagateNaN absolute : Bool) (a b c : Nat) : Nat :=
  finish propagateNaN (join minimum
    (join minimum (wordState ftz absolute a) (wordState ftz absolute b))
    (wordState ftz absolute c))

theorem minmax3_reassociate (minimum ftz propagateNaN absolute : Bool) (a b c : Nat) :
    minmax3 minimum ftz propagateNaN absolute a b c =
      finish propagateNaN (join minimum (wordState ftz absolute a)
        (join minimum (wordState ftz absolute b) (wordState ftz absolute c))) := by
  unfold minmax3
  rw [join_assoc]

def reduceWords (minimum propagateNaN absolute : Bool) (words : List Nat) : Nat :=
  finish propagateNaN (words.foldl
    (fun state word => join minimum state (wordState false absolute word))
    ⟨none, false⟩)

theorem reduceWords_perm (minimum propagateNaN absolute : Bool) (xs ys : List Nat)
    (h : xs.Perm ys) :
    reduceWords minimum propagateNaN absolute xs =
      reduceWords minimum propagateNaN absolute ys := by
  unfold reduceWords
  congr 1
  apply h.foldl_eq'
  intro x _ y _ state
  rw [join_assoc, join_assoc, join_comm minimum (wordState false absolute x)]

example : minmax3 true false false false 0 0x80000000 0 = 0x80000000 := by decide
example : minmax3 false false false false 0 0x80000000 0x80000000 = 0 := by decide
example : minmax3 true false true true 0xffc00001 0x3f800000 0 = canonicalNaN := by decide
example : minmax3 true true false false 0x80000001 0x80000000 1 = 0x80000000 := by decide
example : reduceWords false false false [0x7fc00000, 0xff800000] = 0xff800000 := by decide
example : reduceWords true false false [0x7fc00000, 0xffc00001] = canonicalNaN := by decide

def readWord (data : ByteArray) (offset bytes : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:bytes] do word := word + data[offset + j]!.toNat * 2^(8*j)
  return word

def active (mask lane : Nat) : Bool := (mask / 2^lane) % 2 = 1

end GB300Math.MinMax3Redux

open GB300Math.MinMax3Redux

def parseBool (text : String) : IO Bool :=
  if text = "0" then pure false else if text = "1" then pure true
  else throw (IO.userError "Expected Boolean flag 0 or 1")

def main (args : List String) : IO UInt32 := do
  let [kind, op, ftzFlag, nanFlag, absFlag, input, output] := args
    | throw (IO.userError "Expected kind min|max ftz nan abs input output")
  unless (kind = "minmax3" || kind = "redux") && (op = "min" || op = "max") do
    throw (IO.userError "Unknown operation")
  let ftz ← parseBool ftzFlag
  let propagateNaN ← parseBool nanFlag
  let absolute ← parseBool absFlag
  let minimum := op = "min"
  let warp := kind = "redux"
  unless !warp || !ftz do throw (IO.userError "FP32 REDUX has no FTZ modifier")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  let stride := if warp then 144 else 32
  let outStride := if warp then 128 else 8
  unless xs.size > 0 && xs.size % stride = 0 && ys.size = xs.size / stride * outStride do
    throw (IO.userError "Invalid min/max capture sizes")
  let mut mismatches := 0
  let mut comparisons := 0
  let mut activeComparisons := 0
  let mut inactiveChecks := 0
  for i in [:xs.size / stride] do
    if warp then
      let mask := readWord xs (144*i + 128) 4
      unless mask > 0 do throw (IO.userError "Empty membermask has no modeled executing lane")
      let words := (List.range 32).filterMap fun lane =>
        if active mask lane then some (readWord xs (144*i + 4*lane) 4) else none
      let value := reduceWords minimum propagateNaN absolute words
      for lane in [:32] do
        let model := if active mask lane then value else 0xdeadbeef
        let actual := readWord ys (128*i + 4*lane) 4
        comparisons := comparisons + 1
        if active mask lane then activeComparisons := activeComparisons + 1
        else inactiveChecks := inactiveChecks + 1
        if model ≠ actual then
          if mismatches < 3 then IO.println s!"witness row={i} lane={lane} model={model} actual={actual}"
          mismatches := mismatches + 1
    else
      let model := minmax3 minimum ftz propagateNaN absolute
        (readWord xs (32*i) 8) (readWord xs (32*i+8) 8) (readWord xs (32*i+16) 8)
      let actual := readWord ys (8*i) 8
      comparisons := comparisons + 1
      activeComparisons := activeComparisons + 1
      if model ≠ actual then
        if mismatches < 3 then IO.println s!"witness row={i} model={model} actual={actual}"
        mismatches := mismatches + 1
  IO.println s!"result {kind} cases={xs.size / stride} comparisons={comparisons} mismatches={mismatches}"
  IO.println s!"scope active_results={activeComparisons} inactive_sentinels={inactiveChecks}"
  return if mismatches = 0 then 0 else 1
