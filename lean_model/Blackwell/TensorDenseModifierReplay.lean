import GB300Math.TensorBlackwell

def word (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat + data[4*i+1]!.toNat*256 +
    data[4*i+2]!.toNat*65536 + data[4*i+3]!.toNat*16777216

def main (args : List String) : IO UInt32 := do
  let [fmt, accumulateText, modifierText, inputPath, outputPath, expectedText] := args |
    throw (IO.userError "Expected format accumulate modifier input output expected-mismatches")
  unless (accumulateText = "0" ∨ accumulateText = "1") ∧
      (modifierText = "0" ∨ modifierText = "1" ∨ modifierText = "2" ∨ modifierText = "3") do
    throw (IO.userError "Invalid accumulate or modifier")
  let some expected := expectedText.toNat? | throw (IO.userError "Invalid mismatch count")
  let k ← match fmt with
    | "f16" | "bf16" => pure 16
    | "tf32" => pure 8
    | _ => throw (IO.userError "Unsupported format")
  let some modifier := modifierText.toNat? | throw (IO.userError "Invalid modifier")
  let sign := if fmt = "tf32" then 2^31 else 2^15
  let input ← IO.FS.readBinFile inputPath
  let output ← IO.FS.readBinFile outputPath
  let stride := 136*k+1024
  unless input.size > 0 ∧ input.size % (4*stride) = 0 ∧
      output.size = input.size/(4*stride)*4096 do
    throw (IO.userError "Invalid input/output extent")
  let mut mismatches := 0
  for matrix in [:input.size/(4*stride)] do
    let base := matrix*stride
    let bs := (List.range 8).toArray.map fun col =>
      (List.range k).toArray.map fun j =>
        let v := word input (base+128*k+col*k+j)
        if modifier/2 = 1 then v ^^^ sign else v
    for row in [:128] do
      let a := (List.range k).toArray.map fun j =>
        let v := word input (base+row*k+j)
        if modifier%2 = 1 then v ^^^ sign else v
      for col in [:8] do
        let cell := row*8+col
        let c := if accumulateText = "1" then word input (base+136*k+cell) else 0
        let some predicted := GB300Math.TensorBlackwell.evaluate fmt a bs[col]! c
            true |
          throw (IO.userError s!"Rejected operands: matrix {matrix}, cell {cell}")
        let observed := word output (matrix*1024+cell)
        if predicted != observed then
          if mismatches < 8 then
            IO.println s!"difference matrix={matrix} cell={cell} predicted={predicted} observed={observed}"
          mismatches := mismatches+1
  IO.println s!"cells={output.size/4} mismatches={mismatches} expected={expected}"
  return if mismatches = expected then 0 else 1
