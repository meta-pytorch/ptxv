import Std

namespace GB300Math.PackedDot

def field (word bits index : Nat) (signed : Bool) : Int :=
  let value := word / 2^(bits*index) % 2^bits
  if signed && value ≥ 2^(bits-1) then (value : Int)-2^bits else value

def evaluate (op : String) (signedA signedB : Bool) (a b c : Nat) : Option Nat := do
  let (bits, count, start) ← match op with
    | "dp4a" => some (8, 4, 0)
    | "dp2a.lo" => some (16, 2, 0)
    | "dp2a.hi" => some (16, 2, 2)
    | _ => none
  let total := (List.range count).foldl
    (fun sum i => sum + field a bits i signedA * field b 8 (i+start) signedB) (c : Int)
  return (total % (2^32 : Nat)).toNat

theorem signed_lanes : evaluate "dp4a" true true 0x80808080 0xffffffff 0 = some 512 := by
  decide +kernel
theorem mixed_signs : evaluate "dp4a" true false 0xffffffff 0xffffffff 0 = some 0xfffffc04 := by
  decide +kernel
theorem accumulator_wrap : evaluate "dp4a" false false 0x01010101 0x01010101 0xffffffff = some 3 := by
  decide +kernel
theorem high_half_selection : evaluate "dp2a.hi" false false 0x00020003 0x05040706 0 = some 22 := by
  decide +kernel

#print axioms signed_lanes
#print axioms mixed_signs
#print axioms accumulator_wrap
#print axioms high_half_selection

def word (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut value := 0
  for j in [:4] do value := value + data[offset+j]!.toNat*2^(8*j)
  return value

end GB300Math.PackedDot

open GB300Math.PackedDot

def main (args : List String) : IO UInt32 := do
  let [op, sa, sb, input, output] := args |
    throw (IO.userError "Expected op signedA signedB input.u32x4 output.u32")
  unless (sa = "0" ∨ sa = "1") ∧ (sb = "0" ∨ sb = "1") do
    throw (IO.userError "Signedness must be 0 or 1")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 ∧ xs.size % 16 = 0 ∧ ys.size = xs.size/4 do
    throw (IO.userError "Invalid packed-dot capture sizes")
  let mut mismatches := 0
  for i in [:xs.size/16] do
    let some expected := evaluate op (sa = "1") (sb = "1")
      (word xs (16*i)) (word xs (16*i+4)) (word xs (16*i+8)) |
      throw (IO.userError s!"Unsupported packed dot: {op}")
    let actual := word ys (4*i)
    if expected ≠ actual then
      if mismatches < 3 then IO.println s!"Mismatch row={i} model={expected} hardware={actual}"
      mismatches := mismatches+1
  IO.println s!"result count={xs.size/16} mismatches={mismatches}"
  return if mismatches = 0 then 0 else 1
