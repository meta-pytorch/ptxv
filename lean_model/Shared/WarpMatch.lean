import Std

namespace GB300Math.WarpMatch

def member (mask lane : Nat) : Bool := mask / 2^lane % 2 = 1

def groupMask (values : Array Nat) (mask value : Nat) : Nat :=
  (List.range 32).foldl (fun bits lane =>
    if member mask lane && values[lane]! = value then bits + 2^lane else bits) 0

def evaluate (width : Nat) (all : Bool) (values : Array Nat) (mask lane : Nat) : Option Nat := do
  unless width = 32 ∨ width = 64 do none
  unless values.size = 32 ∧ mask < 2^32 ∧ lane < 32 do none
  unless values.all (· < 2^width) do none
  if !member mask lane then return 0
  let group := groupMask values mask values[lane]!
  if all then return if group = mask then group + 2^32 else 0
  return group

theorem equal_words_match :
    evaluate 32 false (Array.replicate 32 0x7fc00001) 3 0 = some 3 := by decide +kernel

theorem distinct_zero_encodings :
    evaluate 32 false (#[0,0x80000000] ++ Array.replicate 30 0) 3 0 = some 1 := by decide +kernel

theorem high_word_participates :
    evaluate 64 true (#[1,0x100000001] ++ Array.replicate 30 0) 3 0 = some 0 := by decide +kernel

theorem all_returns_mask_and_predicate :
    evaluate 64 true (Array.replicate 32 0xffffffffffffffff) 5 2 = some 0x100000005 := by decide +kernel

theorem inactive_wrapper_output :
    evaluate 32 false (Array.replicate 32 1) 3 2 = some 0 := by decide +kernel

theorem equal_values_have_equal_groups (values : Array Nat) (mask a b : Nat) (h : a = b) :
    groupMask values mask a = groupMask values mask b := by rw [h]

#print axioms equal_words_match
#print axioms distinct_zero_encodings
#print axioms high_word_participates
#print axioms all_returns_mask_and_predicate
#print axioms inactive_wrapper_output
#print axioms equal_values_have_equal_groups

def word (data : ByteArray) (offset size : Nat) : Nat := Id.run do
  let mut value := 0
  for i in [:size] do value := value + data[offset+i]!.toNat * 2^(8*i)
  return value

end GB300Math.WarpMatch

open GB300Math.WarpMatch

def main (args : List String) : IO UInt32 := do
  let [widthText, kind, input, output] := args |
    throw (IO.userError "Expected width any|all inputs.u64x32 outputs.u64x32")
  let some width := widthText.toNat? | throw (IO.userError "Invalid width")
  unless kind = "any" ∨ kind = "all" do throw (IO.userError "Invalid match operation")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 ∧ xs.size % 264 = 0 ∧ ys.size = xs.size/264*256 do
    throw (IO.userError "Invalid warp-match record sizes")
  let mut errors := 0
  for sample in [:xs.size/264] do
    let values := (Array.range 32).map fun i => word xs (sample*264+8*i) 8
    let mask := word xs (sample*264+256) 4
    for lane in [:32] do
      let some expected := evaluate width (kind = "all") values mask lane |
        throw (IO.userError s!"Unsupported input at sample {sample}")
      let actual := word ys (sample*256+8*lane) 8
      if expected ≠ actual then
        if errors < 3 then IO.println s!"Mismatch sample={sample} lane={lane} expected={expected} actual={actual}"
        errors := errors+1
  IO.println s!"result lanes={ys.size/8} mismatches={errors}"
  return if errors = 0 then 0 else 1
