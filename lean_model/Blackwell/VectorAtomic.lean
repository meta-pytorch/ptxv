import AtomicSpec
import PackedMinMaxSpec

namespace GB300Math.VectorAtomic

open FloatLib.Floats.Formats.BinaryInterchange

def wordBits (precision : String) : Nat :=
  if precision = "f16" || precision = "bf16" then 16 else 32

def lane (op precision : String) (global noftz : Bool) (a b : Nat) : Nat :=
  if op = "add" then
    let format := if precision = "f16" then FloatFormat.binary16
      else if precision = "bf16" then FloatFormat.bfloat16 else FloatFormat.binary32
    GB300Math.Atomic.add format (global && !noftz) a b
  else
    let format : GB300Math.PackedMinMax.Format :=
      if precision = "f16" then ⟨16, 10⟩ else ⟨16, 7⟩
    GB300Math.PackedMinMax.select format (op = "max") false a b

def word (op precision : String) (global noftz : Bool) (a b : Nat) : Nat :=
  if precision = "f16x2" || precision = "bf16x2" then
    let half := if precision = "f16x2" then "f16" else "bf16"
    lane op half global noftz (a % 65536) (b % 65536) +
      65536 * lane op half global noftz (a / 65536) (b / 65536)
  else lane op precision global noftz a b

def legal (kind op precision space : String) (width : Nat) (noftz : Bool) : Bool :=
  (kind = "atom" || kind = "red") && (space = "global" || space = "shared") &&
  if precision = "f32" then op = "add" && (width = 1 || (space = "global" && (width = 2 || width = 4)))
  else noftz && space = "global" && (op = "add" || op = "min" || op = "max") &&
    if precision = "f16" || precision = "bf16" then width = 2 || width = 4 || width = 8
    else (precision = "f16x2" || precision = "bf16x2") && (width = 2 || width = 4)

theorem packed_components (op precision : String) (global noftz : Bool) (a b : Nat)
    (h : precision = "f16x2" ∨ precision = "bf16x2") :
    word op precision global noftz a b =
      lane op (if precision = "f16x2" then "f16" else "bf16") global noftz (a % 65536) (b % 65536) +
      65536 * lane op (if precision = "f16x2" then "f16" else "bf16") global noftz (a / 65536) (b / 65536) := by
  rcases h with h | h <;> simp [word, h]

theorem shared_vector_rejected (kind op precision : String) (width : Nat) (noftz : Bool)
    (h : width ≠ 1) : legal kind op precision "shared" width noftz = false := by
  simp [legal, h]

theorem signed_zero_min : word "min" "f16x2" true true 0x00008000 0x80000000 = 0x80008000 := by decide
theorem signed_zero_max : word "max" "bf16x2" true true 0x00008000 0x80000000 = 0 := by decide

#print axioms packed_components
#print axioms shared_vector_rejected
#print axioms signed_zero_min
#print axioms signed_zero_max

def read32 (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:4] do word := word + data[offset+j]!.toNat * 2^(8*j)
  return word

end GB300Math.VectorAtomic

open GB300Math.VectorAtomic

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected vector atomic case manifest")
  let cases ← IO.FS.readFile manifest
  unless (cases.splitOn "\n").any (! ·.isEmpty) do throw (IO.userError "Empty atomic manifest")
  let mut failures := 0
  for line in cases.splitOn "\n" do
    if line.isEmpty then continue
    let [kind, op, precision, space, widthText, noftzText, input, output, errors] := line.splitOn "\t" |
      throw (IO.userError "Malformed vector atomic case")
    let some width := widthText.toNat? | throw (IO.userError "Invalid vector width")
    let some expectedErrors := errors.toNat? | throw (IO.userError "Invalid error count")
    unless noftzText = "0" ∨ noftzText = "1" do throw (IO.userError "Invalid noftz flag")
    let noftz := noftzText == "1"
    unless legal kind op precision space width noftz do throw (IO.userError "Illegal vector atomic form")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 64 = 0 ∧ ys.size = xs.size / 64 * 128 do
      throw (IO.userError "Invalid atomic record sizes")
    let mut mismatches := 0
    for i in [:xs.size / 64] do
      let mut matched := true
      for j in [:32] do
        let expected := if j < width then
            if kind = "atom" then read32 xs (64*i+4*j) % 2^(wordBits precision) else 0
          else if 8 ≤ j ∧ j < 8+width then
            word op precision (space = "global") noftz
              (read32 xs (64*i+4*(j-8)) % 2^(wordBits precision))
              (read32 xs (64*i+4*j) % 2^(wordBits precision))
          else 0
        if read32 ys (128*i+4*j) ≠ expected then matched := false
      if !matched then
        if mismatches < 3 then IO.println s!"witness {kind} {op} {precision} width={width} row={i}"
        mismatches := mismatches+1
    IO.println s!"result {output} records={xs.size/64} mismatches={mismatches} expected={expectedErrors}"
    if mismatches ≠ expectedErrors then failures := failures+1
  return if failures = 0 then 0 else 1
