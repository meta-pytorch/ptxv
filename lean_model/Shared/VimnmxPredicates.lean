import GB300Math.NativeIntegerModifiers

namespace GB300Math.VimnmxPredicates

open NativeIntegerModifiers

def flags (width a result : Nat) : Nat :=
  if width = 32 then if result = a then 3 else 0
  else (if result % 65536 = a % 65536 then 1 else 0) +
    (if result / 65536 = a / 65536 then 2 else 0)

def output (kind operation : String) (signed relu : Bool) (width a b c : Nat) : Nat :=
  let result := evaluate operation signed relu false false width 0 a b c
  let predicates := if kind = "compiler_control" then
    (if result < c then 1 else 0) + (if result = a then 2 else 0)
    else flags width a result
  result + predicates * 2^32

theorem flags_bounded (width a result : Nat) : flags width a result ≤ 3 := by
  unfold flags
  split <;> split <;> (try split) <;> omega

theorem scalar_predicates_agree (a result : Nat) :
    flags 32 a result = 0 ∨ flags 32 a result = 3 := by
  simp only [flags, ite_true]
  split <;> simp

theorem relu_zero_selects_equal_source :
    output "native" "min" true true 32 0 4294967295 0 = 3 * 2^32 := by decide

theorem equal_sources_set_both (a : Nat) : flags 16 a a = 3 := by simp [flags]

#print axioms flags_bounded
#print axioms scalar_predicates_agree
#print axioms relu_zero_selects_equal_source
#print axioms equal_sources_set_both

end GB300Math.VimnmxPredicates

open GB300Math.NativeIntegerModifiers GB300Math.VimnmxPredicates

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected VIMNMX predicate manifest")
  let rows := (← IO.FS.readFile manifest).splitOn "\n" |>.filter (! ·.isEmpty)
  unless !rows.isEmpty do throw (IO.userError "Empty manifest")
  let mut failed := 0
  for row in rows do
    let [kind, operation, signed, relu, width, input, actualOutput, expected] := row.splitOn "\t" |
      throw (IO.userError "Malformed VIMNMX predicate row")
    unless kind ∈ ["compiler_control", "native_predicate_diagnostic"] ∧ operation ∈ ["min", "max"] ∧ signed ∈ ["0", "1"] ∧ relu ∈ ["0", "1"] do
      throw (IO.userError "Invalid operation flags")
    let some width := width.toNat? | throw (IO.userError "Invalid width")
    unless width ∈ [16,32] do throw (IO.userError "Unsupported width")
    let some expected := expected.toNat? | throw (IO.userError "Invalid mismatch count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile actualOutput
    unless xs.size > 0 ∧ xs.size % 16 = 0 ∧ ys.size = xs.size/2 do
      throw (IO.userError "Invalid record sizes")
    let mut errors := 0
    for i in [:xs.size/16] do
      let target := output kind operation (signed = "1") (relu = "1") width
        (readWord xs (16*i) 4) (readWord xs (16*i+4) 4) (readWord xs (16*i+8) 4)
      let actual := readWord ys (8*i) 8
      if actual ≠ target then
        if errors < 5 then IO.println s!"witness row={i} expected={target} actual={actual}"
        errors := errors+1
    IO.println s!"result {actualOutput} records={xs.size/16} mismatches={errors} expected={expected}"
    if errors ≠ expected then failed := failed+1
  return if failed = 0 then 0 else 1
