import GB300Math.Mufu
import GB300Math.Fp64Seed

namespace GB300Math.MufuModifierExhaustive

def modify (modifier input : Nat) : Nat :=
  let word := input % 2^32
  let value := if modifier / 2 % 2 = 1 then word % 2^31 else word
  if modifier % 2 = 1 then value ^^^ 2^31 else value

def sample (op : String) (modifier input actual : Nat) : Bool :=
  let expected := if op = "rcp64h" ∨ op = "rsq64h"
    then some (Fp64Seed.highWord (op = "rsq64h") (modify modifier input))
    else Mufu.evaluate op (modify modifier input)
  match expected with
  | none => false
  | some expected => actual % 2^32 = expected && actual / 2^32 = expected

def summary (control : Bool) (hiWord : Nat) : Nat :=
  if control then 3*2^32 + hiWord*2^20 else 2^32-1

theorem full_word_partition (word : Nat) (h : word < 2^32) :
    word / 2^20 < 4096 ∧ word % 2^20 < 2^20 ∧
    word = word / 2^20 * 2^20 + word % 2^20 := by omega

theorem planted_positions_distinct :
    (0 : Nat) < 2^19 ∧ 2^19 < 2^20-1 ∧ 2^20-1 < 2^20 := by decide +kernel

theorem first_planted_word (hiWord suffix : Nat) (h : suffix < 2^20) :
    hiWord*2^20 ≤ hiWord*2^20+suffix := by omega

#print axioms full_word_partition
#print axioms planted_positions_distinct
#print axioms first_planted_word

def readWord (data : ByteArray) (offset bytes : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:bytes] do word := word + data[offset+j]!.toNat * 2^(8*j)
  return word

end GB300Math.MufuModifierExhaustive

open GB300Math.MufuModifierExhaustive

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected manifest")
  let rows := (← IO.FS.readFile manifest).splitOn "\n" |>.filter (! ·.isEmpty)
  unless !rows.isEmpty do throw (IO.userError "Empty manifest")
  let mut failed := 0
  for row in rows do
    let [op, modifier, kind, input, output, expected] := row.splitOn "\t" |
      throw (IO.userError "Malformed row")
    let some modifier := modifier.toNat? | throw (IO.userError "Invalid modifier")
    let some expected := expected.toNat? | throw (IO.userError "Invalid mismatch count")
    unless modifier ∈ [0,1,2,3] ∧ kind ∈ ["sample","sweep","control"] do
      throw (IO.userError "Unsupported form")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 4 = 0 ∧ ys.size = xs.size*2 do
      throw (IO.userError "Invalid sizes")
    let mut errors := 0
    for i in [:xs.size/4] do
      let word := readWord xs (4*i) 4
      let actual := readWord ys (8*i) 8
      let valid := if kind = "sample" then sample op modifier word actual
        else word < 4096 && actual = summary (kind = "control") word
      if !valid then
        if errors < 4 then IO.println s!"witness row={i} input={word} actual={actual}"
        errors := errors+1
    IO.println s!"result {output} records={xs.size/4} mismatches={errors} expected={expected}"
    if errors ≠ expected then failed := failed+1
  return if failed = 0 then 0 else 1
