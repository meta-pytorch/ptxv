import Std

namespace GB300Math.IntegerAtomic

def signed (width value : Nat) : Int :=
  if value < 2^(width-1) then value else (value : Int) - 2^width

def update (width : Nat) (op : String) (a b c : Nat) : Option Nat := do
  unless width = 32 ∨ width = 64 do none
  unless a < 2^width ∧ b < 2^width ∧ c < 2^width do none
  match op with
  | "add" => return (a+b) % 2^width
  | "min_u" => return min a b
  | "max_u" => return max a b
  | "min_s" => return if signed width a < signed width b then a else b
  | "max_s" => return if signed width a > signed width b then a else b
  | "inc" => if width = 32 then return if a ≥ b then 0 else a+1 else none
  | "dec" => if width = 32 then return if a = 0 ∨ a > b then b else a-1 else none
  | "and" => return a &&& b
  | "or" => return a ||| b
  | "xor" => return a ^^^ b
  | "exch" => return b
  | "cas" => return if a = b then c else a
  | _ => none

theorem add_wraps : update 32 "add" 0xffffffff 1 0 = some 0 := by decide +kernel
theorem signed_minimum : update 64 "min_s" 0x8000000000000000 1 0 = some 0x8000000000000000 := by decide +kernel
theorem increment_limit_resets : update 32 "inc" 7 7 0 = some 0 := by decide +kernel
theorem decrement_above_limit : update 32 "dec" 8 7 0 = some 7 := by decide +kernel
theorem cas_bit_equality : update 32 "cas" 0x80000000 0 4 = some 0x80000000 := by decide +kernel
theorem cas_replaces : update 64 "cas" 0xffffffffffffffff 0xffffffffffffffff 7 = some 7 := by decide +kernel

#print axioms add_wraps
#print axioms signed_minimum
#print axioms increment_limit_resets
#print axioms decrement_above_limit
#print axioms cas_bit_equality
#print axioms cas_replaces

def word (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut value := 0
  for i in [:8] do value := value + data[offset+i]!.toNat * 2^(8*i)
  return value

def linearize (width : Nat) (op : String) (bs cs old : Array Nat) (final : Nat) :
    Nat → Nat → Nat → Bool
  | 0, state, used => state = final && used = 15
  | fuel+1, state, used => (List.range 4).any fun lane =>
    if used &&& 2^lane ≠ 0 ∨ old[lane]! ≠ state then false
    else match update width op state bs[lane]! cs[lane]! with
      | none => false
      | some next => linearize width op bs cs old final fuel next (used+2^lane)

def checkTrace (width : Nat) (op : String) (initial : Nat)
    (bs cs old : Array Nat) (final : Nat) : Option Bool := do
  unless bs.size = 4 ∧ cs.size = 4 ∧ old.size = 4 do none
  unless old.all (· < 2^width) ∧ final < 2^width do none
  for i in [:4] do
    let _ ← update width op initial bs[i]! cs[i]!
  return linearize width op bs cs old final 4 initial 0

theorem serialized_add_history :
    checkTrace 32 "add" 0 #[1,2,4,8] #[0,0,0,0] #[0,1,3,7] 15 = some true := by decide +kernel

theorem reordered_add_history :
    checkTrace 32 "add" 0 #[1,2,4,8] #[0,0,0,0] #[10,8,11,0] 15 = some true := by decide +kernel

theorem correct_final_does_not_validate_returns :
    checkTrace 32 "add" 0 #[1,2,4,8] #[0,0,0,0] #[0,0,0,0] 15 = some false := by decide +kernel

theorem chained_cas_history :
    checkTrace 64 "cas" 1 #[1,2,3,4] #[2,3,4,5] #[1,2,3,4] 5 = some true := by decide +kernel

#print axioms serialized_add_history
#print axioms reordered_add_history
#print axioms correct_final_does_not_validate_returns
#print axioms chained_cas_history

def runTraces (manifest : String) : IO UInt32 := do
  let lines := (← IO.FS.readFile manifest).splitOn "\n" |>.filter (!·.isEmpty)
  unless !lines.isEmpty do throw (IO.userError "Empty trace manifest")
  let mut total := 0
  let mut errors := 0
  for line in lines do
    let [widthText, op, input, output] := line.splitOn "\t" |
      throw (IO.userError "Malformed trace manifest")
    let some width := widthText.toNat? | throw (IO.userError "Invalid width")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 72 = 0 ∧ ys.size = xs.size/72*40 do
      throw (IO.userError "Invalid trace record sizes")
    let mut failures := 0
    for i in [:xs.size/72] do
      let initial := word xs (i*72)
      let bs := (Array.range 4).map fun j => word xs (i*72+8+j*16)
      let cs := (Array.range 4).map fun j => word xs (i*72+16+j*16)
      let old := (Array.range 4).map fun j => word ys (i*40+j*8)
      let final := word ys (i*40+32)
      let some valid := checkTrace width op initial bs cs old final |
        throw (IO.userError s!"Invalid trace input at {i}")
      if !valid then
        if failures < 3 then IO.println s!"Nonlinearizable trace {width} {op} sample={i} initial={initial} old={old} final={final}"
        failures := failures+1
    IO.println s!"trace {width} {op} records={xs.size/72} mismatches={failures}"
    (← IO.getStdout).flush
    total := total+xs.size/72
    errors := errors+failures
  IO.println s!"total traces={total} mismatches={errors}"
  return if errors = 0 then 0 else 1

end GB300Math.IntegerAtomic

open GB300Math.IntegerAtomic

def main (args : List String) : IO UInt32 := do
  if let ["trace", manifest] := args then return ← runTraces manifest
  let [manifest] := args | throw (IO.userError "Expected integer atomic manifest")
  let lines := (← IO.FS.readFile manifest).splitOn "\n" |>.filter (!·.isEmpty)
  unless !lines.isEmpty do throw (IO.userError "Empty manifest")
  let mut total := 0
  let mut errors := 0
  for line in lines do
    let [widthText, op, mode, input, output] := line.splitOn "\t" |
      throw (IO.userError "Malformed integer atomic record")
    let some width := widthText.toNat? | throw (IO.userError "Invalid width")
    unless mode = "return" ∨ mode = "reduce" do throw (IO.userError "Invalid mode")
    unless mode ≠ "reduce" ∨ (op ≠ "cas" ∧ op ≠ "exch") do
      throw (IO.userError "Unsupported reduction operation")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 24 = 0 ∧ ys.size = xs.size/24*16 do
      throw (IO.userError "Invalid record sizes")
    let mut failures := 0
    for sample in [:xs.size/24] do
      let a := word xs (sample*24)
      let b := word xs (sample*24+8)
      let c := word xs (sample*24+16)
      let some expected := update width op a b c |
        throw (IO.userError s!"Unsupported input at sample {sample}")
      let old := word ys (sample*16)
      let final := word ys (sample*16+8)
      if final ≠ expected ∨ old ≠ (if mode = "return" then a else 0) then
        if failures < 3 then IO.println s!"Mismatch sample={sample} expected={expected} final={final} old={old}"
        failures := failures+1
    IO.println s!"result {width} {op} {mode} records={xs.size/24} mismatches={failures}"
    total := total+xs.size/24
    errors := errors+failures
  IO.println s!"total records={total} mismatches={errors}"
  return if errors = 0 then 0 else 1
