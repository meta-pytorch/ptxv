import AtomicSpec
import PackedMinMaxSpec
import IntegerAtomicSpec

set_option maxRecDepth 100000
set_option maxHeartbeats 2000000
namespace GB300Math.BulkReduction
open FloatLib.Floats.Formats.BinaryInterchange

def bits (precision : String) : Nat :=
  if precision = "f16" || precision = "bf16" then 16
  else if precision.endsWith "64" then 64 else 32

def evaluate (op precision : String) (a b : Nat) : Option Nat := do
  let width := bits precision
  unless a < 2^width ∧ b < 2^width do none
  if op = "copy" then return b
  if precision.startsWith "f" || precision = "bf16" then
    if op = "add" then
      let some fmt := (match precision with
        | "f16" => some FloatFormat.binary16
        | "bf16" => some FloatFormat.bfloat16
        | "f32" => some FloatFormat.binary32
        | "f64" => some FloatFormat.binary64
        | _ => none) | none
      return Atomic.add fmt (precision = "f64") a b
    unless precision = "f16" ∨ precision = "bf16" do none
    unless op = "min" ∨ op = "max" do none
    return PackedMinMax.select ⟨16, if precision = "f16" then 10 else 7⟩ (op = "max") false a b
  let instruction := if op = "min" || op = "max" then
      op ++ (if precision.startsWith "s" then "_s" else "_u") else op
  IntegerAtomic.update width instruction a b 0

def legal (space op precision : String) (noftz : Bool) : Bool :=
  if space ≠ "global" && space ≠ "cluster" then false
  else if op = "copy" then !noftz &&
    (if space = "cluster" then precision ∈ ["u32", "s32", "u64", "b32"]
    else precision ∈ ["u32", "s32", "u64", "s64", "b32", "b64", "f16", "bf16", "f32", "f64"])
  else if precision = "f16" || precision = "bf16" then
    space = "global" && ((op = "add" && noftz) || ((op = "min" || op = "max") && !noftz))
  else if precision = "f32" then space = "global" && op = "add"
  else if precision = "f64" then space = "global" && op = "add" && !noftz
  else !noftz &&
    if precision = "b32" || (precision = "b64" && space = "global") then op ∈ ["and", "or", "xor"]
    else if precision = "u32" then op ∈ ["add", "min", "max", "inc", "dec"]
    else if precision = "s32" then op ∈ ["add", "min", "max"]
    else if precision = "u64" then op = "add" || (space = "global" && (op = "min" || op = "max"))
    else space = "global" && precision = "s64" && (op = "min" || op = "max")


theorem float_cluster_rejected (op : String) (noftz : Bool) : legal "cluster" op "f32" noftz = false := by simp [legal]
set_option maxRecDepth 100000 in
set_option maxHeartbeats 2000000 in
theorem f32_subnormal_preserved : evaluate "add" "f32" 1 1 = some 2 := by
  simp only [evaluate, bits]
  decide +kernel
theorem fp16_min_signed_zero : evaluate "min" "f16" 0 0x8000 = some 0x8000 := by decide +kernel
theorem unsigned_add_wraps : evaluate "add" "u32" 0xffffffff 1 = some 0 := by decide +kernel
#print axioms float_cluster_rejected
#print axioms f32_subnormal_preserved
#print axioms fp16_min_signed_zero
#print axioms unsigned_add_wraps

def read64 (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut result := 0
  for j in [:8] do result := result + data[offset+j]!.toNat * 2^(8*j)
  return result
end GB300Math.BulkReduction
open GB300Math.BulkReduction

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected bulk reduction manifest")
  let lines := (← IO.FS.readFile manifest).splitOn "\n" |>.filter (! ·.isEmpty)
  unless !lines.isEmpty do throw (IO.userError "Empty bulk reduction manifest")
  let mut failed := 0
  for line in lines do
    let [space, op, precision, noftzText, input, output, expectedText] := line.splitOn "\t" |
      throw (IO.userError "Malformed bulk reduction row")
    unless noftzText = "0" ∨ noftzText = "1" do throw (IO.userError "Invalid noftz flag")
    unless legal space op precision (noftzText = "1") do throw (IO.userError "Illegal bulk reduction form")
    let some expected := expectedText.toNat? | throw (IO.userError "Invalid mismatch count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 128 = 0 ∧ xs.size = ys.size do throw (IO.userError "Invalid bulk record sizes")
    let mut errors := 0
    for i in [:xs.size / 128] do
      let mut matched := true
      for j in [:16] do
        let target ← if j < 8 then do
            let some value := evaluate op precision (read64 xs (128*i+8*j)) (read64 xs (128*i+64+8*j)) |
              throw (IO.userError "Out-of-domain scalar input")
            pure value
          else pure 0
        if read64 ys (128*i+8*j) ≠ target then matched := false
      if !matched then
        if errors < 3 then IO.println s!"witness {space} {op} {precision} row={i}"
        errors := errors+1
    IO.println s!"result {output} records={xs.size/128} mismatches={errors} expected={expected}"
    if errors ≠ expected then failed := failed+1
  return if failed = 0 then 0 else 1
