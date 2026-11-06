import Std

namespace GB300Math.Warp

def modulus : Nat := 4294967296

def signed (x : Nat) : Int := if x < 2147483648 then x else (x : Int) - modulus

def member (mask lane : Nat) : Bool := mask / 2 ^ lane % 2 = 1

def shuffle (op : String) (values : Array Nat) (offset control lane : Nat) : Option Nat := do
  let b := offset % 32
  let clamp := control % 32
  let segment := control / 256 % 32
  let low := lane &&& segment
  let high := low ||| (clamp &&& (segment ^^^ 31))
  let source : Int ← match op with
    | "shfl_up" => some ((lane : Int) - b)
    | "shfl_down" => some (lane + b)
    | "shfl_bfly" => some (lane ^^^ b)
    | "shfl_idx" => some (low ||| (b &&& (segment ^^^ 31)))
    | _ => none
  let valid := if op = "shfl_up" then source ≥ (high : Int) else source ≤ (high : Int)
  let index := if valid then source.toNat else lane
  if index ≥ values.size then none
  else return values[index]! + (if valid then modulus else 0)

def aggregate (op : String) (values : Array Nat) (mask : Nat) : Option Nat := do
  let mut acc := 0
  let mut seen := false
  let mut ballot := 0
  for lane in [:32] do
    if !member mask lane then continue
    let value := values[lane]!
    if value != 0 then ballot := ballot + 2 ^ lane
    if !seen then acc := value
    else
      acc ← match op with
      | "redux_add" => some ((acc + value) % modulus)
      | "redux_min_u" => some (min acc value)
      | "redux_max_u" => some (max acc value)
      | "redux_min_s" => some (if signed acc < signed value then acc else value)
      | "redux_max_s" => some (if signed acc > signed value then acc else value)
      | "redux_and" => some (acc &&& value)
      | "redux_or" => some (acc ||| value)
      | "redux_xor" => some (acc ^^^ value)
      | "vote_ballot" | "vote_all" | "vote_any" | "vote_uni" => some acc
      | _ => none
    seen := true
  match op with
  | "vote_ballot" => return ballot
  | "vote_all" => return if ballot = mask then 1 else 0
  | "vote_any" => return if ballot != 0 then 1 else 0
  | "vote_uni" => return if ballot = 0 || ballot = mask then 1 else 0
  | "redux_add" | "redux_min_u" | "redux_max_u" | "redux_min_s" | "redux_max_s"
    | "redux_and" | "redux_or" | "redux_xor" => return acc
  | _ => none

def readWord (data : ByteArray) (offset bytes : Nat) : Nat := Id.run do
  let mut word := 0
  for j in [:bytes] do word := word + data[offset + j]!.toNat * 2 ^ (8 * j)
  return word

end GB300Math.Warp

open GB300Math.Warp

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected a tab-separated case manifest")
  let cases ← IO.FS.readFile manifest
  let mut failures := 0
  for line in cases.splitOn "\n" do
    if line.isEmpty then continue
    let [op, input, output, expectedErrors] := line.splitOn "\t" |
      throw (IO.userError s!"Malformed case: {line}")
    let some expected := expectedErrors.toNat? | throw (IO.userError "Invalid error count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 144 = 0 ∧ ys.size = xs.size / 144 * 256 do
      throw (IO.userError s!"Invalid capture sizes: {input}")
    let mut mismatches := 0
    for row in [:xs.size / 144] do
      let base := row * 144
      let values := (Array.range 32).map fun i => readWord xs (base + 4 * i) 4
      let offset := readWord xs (base + 128) 4
      let control := readWord xs (base + 132) 4
      let mask := readWord xs (base + 136) 4
      if op.startsWith "shfl" && mask != 0xffffffff then
        throw (IO.userError "SHFL replay requires every source lane to participate")
      let reduced := if op.startsWith "shfl" then some 0 else aggregate op values mask
      let some reduced := reduced | throw (IO.userError s!"Unsupported aggregation: {op}")
      for lane in [:32] do
        let model := if op.startsWith "shfl" then shuffle op values offset control lane
          else some (if member mask lane then reduced else 0)
        let some model := model | throw (IO.userError s!"Unsupported or invalid lane: {op} {row} {lane}")
        let actual := readWord ys (row * 256 + lane * 8) 8
        if model != actual then
          if mismatches < 3 then
            IO.println s!"witness {op} row={row} lane={lane} offset={offset} control={control} mask={mask} model={model} hardware={actual}"
          mismatches := mismatches + 1
    IO.println s!"result {output} words={ys.size / 8} mismatches={mismatches} expected={expected}"
    if mismatches != expected then failures := failures + 1
  return if failures = 0 then 0 else 1
