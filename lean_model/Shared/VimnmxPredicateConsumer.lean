import GB300Math.NativeIntegerModifiers

namespace GB300Math.VimnmxPredicateConsumer
open NativeIntegerModifiers

def compilerOutput (a b c : Nat) : Nat :=
  let value := evaluate "min" true false false false 16 0 a b c
  value + ((if value < c then 1 else 0) + (if value = a then 2 else 0)) * 2^32

theorem both_predicates : compilerOutput 0 1 1 = 3 * 2^32 := by decide
theorem only_comparison : compilerOutput 2 1 2 = 1 + 2^32 := by decide
theorem neither_predicate : compilerOutput 2 1 0 = 1 := by decide
#print axioms both_predicates
#print axioms only_comparison
#print axioms neither_predicate
end GB300Math.VimnmxPredicateConsumer

open GB300Math.NativeIntegerModifiers GB300Math.VimnmxPredicateConsumer

def main (args : List String) : IO UInt32 := do
  let [input, output, expectedText] := args | throw (IO.userError "Expected input, output, mismatch count")
  let some expected := expectedText.toNat? | throw (IO.userError "Invalid mismatch count")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 ∧ xs.size % 16 = 0 ∧ ys.size = xs.size/2 do
    throw (IO.userError "Invalid record sizes")
  let mut errors := 0
  for i in [:xs.size/16] do
    let target := compilerOutput (readWord xs (16*i) 4) (readWord xs (16*i+4) 4) (readWord xs (16*i+8) 4)
    let actual := readWord ys (8*i) 8
    if actual ≠ target then
      if errors < 5 then IO.println s!"witness row={i} expected={target} actual={actual}"
      errors := errors+1
  IO.println s!"result records={xs.size/16} mismatches={errors} expected={expected}"
  return if errors = expected then 0 else 1
