import GB300Math.MufuPacked

namespace GB300Math.MufuPackedExhaustive

def expected (corrupt : Option Nat) (low : Nat) : Nat :=
  match corrupt with
  | none => 0
  | some index =>
    let count := if low = index then 65536 else 1
    let first := if low = index then low else index * 65536 + low
    count * 2^32 + first

theorem every_word (word : Nat) :
    word % 65536 + (word / 65536) * 65536 = word := by omega

theorem every_pair (low high : Nat) (hl : low < 65536) (hh : high < 65536) :
    low + high * 65536 < 2^32 ∧
    (low + high * 65536) % 65536 = low ∧
    (low + high * 65536) / 65536 = high := by omega

theorem distinct_pairs (lo hi lo' hi' : Nat) (hlo : lo < 65536) (hlo' : lo' < 65536)
    (h : lo + hi * 65536 = lo' + hi' * 65536) : lo = lo' ∧ hi = hi' := by omega

theorem corrupt_same_low (index : Nat) :
    expected (some index) index = 65536 * 2^32 + index := by simp [expected]

theorem corrupt_other_low (index low : Nat) (h : low ≠ index) :
    expected (some index) low = 2^32 + index * 65536 + low := by
  simp [expected, h, Nat.add_assoc]

theorem corrupted_pair_count : 65536 + 65535 = 131071 := by decide

#print axioms every_word
#print axioms every_pair
#print axioms distinct_pairs
#print axioms corrupt_same_low
#print axioms corrupt_other_low
#print axioms corrupted_pair_count

end GB300Math.MufuPackedExhaustive

open GB300Math.MufuPacked GB300Math.MufuPackedExhaustive

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected scan-result manifest")
  let rows := (← IO.FS.readFile manifest).splitOn "\n" |>.filter (! ·.isEmpty)
  unless !rows.isEmpty do throw (IO.userError "Empty manifest")
  let mut failures := 0
  for row in rows do
    let [control, path, expectedErrors] := row.splitOn "\t" |
      throw (IO.userError "Malformed scan-result row")
    let corrupt ← if control = "none" then pure none else do
      let some index := control.toNat? | throw (IO.userError "Invalid corruption index")
      unless index < 65536 do throw (IO.userError "Corruption index out of bounds")
      pure (some index)
    let some expectedErrors := expectedErrors.toNat? |
      throw (IO.userError "Invalid expected mismatch count")
    let actual ← IO.FS.readBinFile path
    unless actual.size = 65536 * 8 do throw (IO.userError "Incomplete scan result")
    let mut mismatches := 0
    for low in [:65536] do
      let word := readWord actual (8*low)
      let target := expected corrupt low
      if word ≠ target then
        if mismatches < 5 then IO.println s!"witness low={low} expected={target} actual={word}"
        mismatches := mismatches+1
    IO.println s!"result {path} records=65536 mismatches={mismatches} expected={expectedErrors}"
    if mismatches ≠ expectedErrors then failures := failures+1
  return if failures = 0 then 0 else 1
