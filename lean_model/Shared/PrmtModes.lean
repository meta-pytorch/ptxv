import Std

namespace GB300Math.PrmtModes

inductive Mode where
  | f4e | b4e | rc8 | ecl | ecr | rc16
  deriving DecidableEq

def parseMode : String → Option Mode
  | "f4e" => some .f4e
  | "b4e" => some .b4e
  | "rc8" => some .rc8
  | "ecl" => some .ecl
  | "ecr" => some .ecr
  | "rc16" => some .rc16
  | _ => none

def index (mode : Mode) (selector lane : Nat) : Nat :=
  let control := selector % 4
  match mode with
  | .f4e => control + lane
  | .b4e => (control + 8 - lane) % 8
  | .rc8 => control
  | .ecl => max control lane
  | .ecr => min control lane
  | .rc16 => 2 * (control % 2) + lane % 2

def byte (a b position : Nat) : Nat :=
  let word := if position < 4 then a else b
  word / 2 ^ (8 * (position % 4)) % 256

def evaluate (mode : Mode) (a b selector : Nat) : Nat :=
  byte a b (index mode selector 0) +
  256 * byte a b (index mode selector 1) +
  65536 * byte a b (index mode selector 2) +
  16777216 * byte a b (index mode selector 3)

theorem output_fits_word (mode : Mode) (a b selector : Nat) :
    evaluate mode a b selector < 4294967296 := by
  have h0 : byte a b (index mode selector 0) < 256 := Nat.mod_lt _ (by decide)
  have h1 : byte a b (index mode selector 1) < 256 := Nat.mod_lt _ (by decide)
  have h2 : byte a b (index mode selector 2) < 256 := Nat.mod_lt _ (by decide)
  have h3 : byte a b (index mode selector 3) < 256 := Nat.mod_lt _ (by decide)
  unfold evaluate
  omega

theorem selector_high_bits_ignored (mode : Mode) (a b selector : Nat) :
    evaluate mode a b (selector % 4) = evaluate mode a b selector := by
  simp [evaluate, index]

theorem backward_extraction_wraps :
    evaluate .b4e 0x03020100 0x07060504 0 = 0x05060700 := by decide

theorem replicate_halfword_ignores_second_word :
    ∀ b, evaluate .rc16 0x03020100 b 3 = 0x03020302 := by
  intro b
  simp [evaluate, index, byte]

#print axioms output_fits_word
#print axioms selector_high_bits_ignored
#print axioms backward_extraction_wraps
#print axioms replicate_halfword_ignores_second_word

def readWord (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:8] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

end GB300Math.PrmtModes

open GB300Math.PrmtModes

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected a tab-separated case manifest")
  let cases ← IO.FS.readFile manifest
  unless cases.splitOn "\n" |>.any (fun line => !line.isEmpty) do
    throw (IO.userError "Empty case manifest")
  let mut failures := 0
  for line in cases.splitOn "\n" do
    if line.isEmpty then continue
    let [modeName, input, output, expectedErrors] := line.splitOn "\t" |
      throw (IO.userError s!"Malformed case: {line}")
    let some mode := parseMode modeName | throw (IO.userError s!"Unsupported mode: {modeName}")
    let some expected := expectedErrors.toNat? | throw (IO.userError "Invalid error count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 32 = 0 ∧ ys.size = xs.size / 4 do
      throw (IO.userError s!"Invalid capture sizes: {input}")
    let mut mismatches := 0
    for i in [:xs.size / 32] do
      let a := readWord xs (32 * i)
      let b := readWord xs (32 * i + 8)
      let c := readWord xs (32 * i + 16)
      let model := evaluate mode a b c
      let actual := readWord ys (8 * i)
      if model ≠ actual then
        if mismatches < 3 then
          IO.println s!"witness {modeName} row={i} a={a} b={b} c={c} model={model} hardware={actual}"
        mismatches := mismatches + 1
    IO.println s!"result {output} cases={xs.size / 32} mismatches={mismatches} expected={expected}"
    if mismatches != expected then failures := failures + 1
  return if failures = 0 then 0 else 1
