import GB300Math.NativeIntegerModifiers

namespace GB300Math.NativeIntegerControls

open NativeIntegerModifiers

def selectedOperation (predicate invert : Bool) : String :=
  if predicate != invert then "min" else "max"

def output (kind : String) (signed relu invert : Bool) (width immediate a b c : Nat) : Nat :=
  if kind = "immediate" then evaluate "add" false false false false width 0 a immediate c
  else evaluate (selectedOperation (c % 2 = 1) invert) signed relu false false width 0 a b c

theorem complement_selection (predicate invert : Bool) :
    selectedOperation (!predicate) (!invert) = selectedOperation predicate invert := by
  cases predicate <;> cases invert <;> rfl

theorem immediate_zero_low_lane (a : Nat) :
    lane "add" false false false 16 a 0 = a % 65536 := by
  simp [lane, decode, encode]
  exact Int.toNat_emod (by omega) (by decide)

theorem packed_carry_stays_in_lane :
    output "immediate" false false false 16 1 65535 0 0 = 0 := by decide

theorem scalar_carry_crosses_bit16 :
    output "immediate" false false false 32 1 65535 0 0 = 65536 := by decide

#print axioms complement_selection
#print axioms immediate_zero_low_lane
#print axioms packed_carry_stays_in_lane
#print axioms scalar_carry_crosses_bit16

end GB300Math.NativeIntegerControls

open GB300Math.NativeIntegerModifiers GB300Math.NativeIntegerControls

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected integer control manifest")
  let rows := (← IO.FS.readFile manifest).splitOn "\n" |>.filter (! ·.isEmpty)
  unless !rows.isEmpty do throw (IO.userError "Empty manifest")
  let mut failed := 0
  for row in rows do
    let [kind, signed, relu, invert, width, immediate, input, actualOutput, expected] := row.splitOn "\t" |
      throw (IO.userError "Malformed integer control row")
    unless kind ∈ ["immediate", "dynamic"] ∧ signed ∈ ["0", "1"] ∧ relu ∈ ["0", "1"] ∧ invert ∈ ["0", "1"] do
      throw (IO.userError "Invalid operation flags")
    let some width := width.toNat? | throw (IO.userError "Invalid width")
    unless width ∈ [16,32] do throw (IO.userError "Unsupported width")
    let some immediate := immediate.toNat? | throw (IO.userError "Invalid immediate")
    unless immediate < 2^32 do throw (IO.userError "Immediate outside uint32")
    let some expected := expected.toNat? | throw (IO.userError "Invalid mismatch count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile actualOutput
    unless xs.size > 0 ∧ xs.size % 16 = 0 ∧ ys.size = xs.size/2 do
      throw (IO.userError "Invalid record sizes")
    let mut errors := 0
    for i in [:xs.size/16] do
      let target := output kind (signed = "1") (relu = "1") (invert = "1") width immediate
        (readWord xs (16*i) 4) (readWord xs (16*i+4) 4) (readWord xs (16*i+8) 4)
      let actual := readWord ys (8*i) 8
      if actual ≠ target then
        if errors < 5 then IO.println s!"witness row={i} expected={target} actual={actual}"
        errors := errors+1
    IO.println s!"result {actualOutput} records={xs.size/16} mismatches={errors} expected={expected}"
    if errors ≠ expected then failed := failed+1
  return if failed = 0 then 0 else 1
