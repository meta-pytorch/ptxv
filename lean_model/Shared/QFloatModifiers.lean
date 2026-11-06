import GB300Math.AlternateFloat

namespace GB300Math.QFloatModifiers

open GB300Math.AlternateFloat

def select (selector word lane : Nat) : Nat :=
  if selector = 0 then word / 2^(8*lane) % 256
  else if selector < 5 then word / 2^(8*(4-selector)) % 256
  else if selector = 5 then word / 2^(4*(lane+4)) % 16
  else if selector = 6 then word / 2^(4*lane) % 16
  else word / 2^(8*lane) % 16

def absolute : Value → Value
  | .finite _ magnitude => .finite false magnitude
  | .infinity _ => .infinity false
  | .nan => .nan

def modify (modifier : Nat) (value : Value) : Value :=
  let value := if modifier % 2 = 1 then absolute value else value
  if modifier / 2 % 2 = 1 then negate value else value

def defaultSelector (format : String) : Nat :=
  if format = "e2m1" then 6 else if format = "e2m1p4" then 7 else 0

def operand (fmt : String) (word lane index selected selector modifier : Nat) : Option Value := do
  let value ← decode fmt (select (if index = selected then selector else defaultSelector fmt) word lane)
  return if index = selected then modify modifier value else value

def evaluate (op dst af bf : String) (sat : Bool) (selected selector modifier a b c : Nat) : Option Nat := do
  unless legal op dst af bf ∧ op ≠ "sub" ∧ selected < (if op = "fma" then 3 else 2) ∧ selector < 8 ∧ modifier < 4 do none
  unless (selected = 2 ∨ (op = "add" ∧ selected = 1)) → selector < 5 do none
  unless a < 2^32 ∧ b < 2^32 ∧ c < 2^32 do none
  let mut output := 0
  for lane in [:4] do
    let x ← operand af a lane 0 selected selector modifier
    let y ← operand bf b lane 1 selected selector modifier
    let z ← operand dst c lane 2 selected selector modifier
    let value := if op = "add" then add x y else if op = "mul" then mul x y else add (mul x y) z
    output := output + encode dst sat value * 2^(8*lane)
  return output

theorem selected_byte_bound (selector word lane : Nat) : select selector word lane < 256 := by
  unfold select
  split
  · exact Nat.mod_lt _ (by decide)
  · split
    · exact Nat.mod_lt _ (by decide)
    · split
      · have h := Nat.mod_lt (word / 2 ^ (4 * (lane + 4))) (by decide : 0 < 16)
        omega
      · split
        · have h := Nat.mod_lt (word / 2 ^ (4 * lane)) (by decide : 0 < 16)
          omega
        · have h := Nat.mod_lt (word / 2 ^ (8 * lane)) (by decide : 0 < 16)
          omega

theorem absolute_idempotent (value : Value) : absolute (absolute value) = absolute value := by
  cases value <;> rfl

theorem negate_involution (value : Value) : negate (negate value) = value := by
  cases value with
  | finite sign magnitude => cases sign <;> rfl
  | infinity sign => cases sign <;> rfl
  | nan => rfl

theorem nibble_upper_example :
    (List.range 4).map (select 5 0x76543210) = [4, 5, 6, 7] := by decide
theorem nibble_lower_example :
    (List.range 4).map (select 6 0x76543210) = [0, 1, 2, 3] := by decide
theorem nibble_even_example :
    (List.range 4).map (select 7 0x76543210) = [0, 2, 4, 6] := by decide

#print axioms selected_byte_bound
#print axioms absolute_idempotent
#print axioms negate_involution
#print axioms nibble_upper_example
#print axioms nibble_lower_example
#print axioms nibble_even_example

end GB300Math.QFloatModifiers

open GB300Math.AlternateFloat

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected Q modifier manifest")
  let rows := (← IO.FS.readFile manifest).splitOn "\n" |>.filter (! ·.isEmpty)
  unless !rows.isEmpty do throw (IO.userError "Empty manifest")
  let mut failures := 0
  for row in rows do
    let [op, dst, af, bf, sat, selected, selector, modifier, input, output, errors] := row.splitOn "\t" |
      throw (IO.userError "Malformed Q modifier row")
    let some selected := selected.toNat? | throw (IO.userError "Invalid selected operand")
    let some selector := selector.toNat? | throw (IO.userError "Invalid selector")
    let some modifier := modifier.toNat? | throw (IO.userError "Invalid modifier")
    let some expected := errors.toNat? | throw (IO.userError "Invalid mismatch count")
    unless sat = "0" ∨ sat = "1" do throw (IO.userError "Invalid saturation flag")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 16 = 0 ∧ ys.size = xs.size do
      throw (IO.userError "Invalid input/output record sizes")
    let mut mismatches := 0
    for i in [:xs.size/16] do
      let a := read32 xs (16*i)
      let b := read32 xs (16*i+4)
      let c := read32 xs (16*i+8)
      unless read32 xs (16*i+12) = 0x93b471ea do throw (IO.userError "Invalid sentinel")
      let some target := GB300Math.QFloatModifiers.evaluate op dst af bf (sat = "1") selected selector modifier a b c |
        throw (IO.userError "Unsupported native configuration")
      let actual := read32 ys (16*i)
      if actual ≠ target ∨ read32 ys (16*i+4) ≠ a ∨ read32 ys (16*i+8) ≠ b ∨ read32 ys (16*i+12) ≠ c then
        if mismatches < 5 then IO.println s!"witness row={i} a={a} b={b} c={c} expected={target} actual={actual}"
        mismatches := mismatches+1
    IO.println s!"result {output} records={xs.size/16} mismatches={mismatches} expected={expected}"
    if mismatches ≠ expected then failures := failures+1
  return if failures = 0 then 0 else 1
