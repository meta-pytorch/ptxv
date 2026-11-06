import GB300Math.TensorSparseBF16Blackwell
open GB300Math.SparseTensorExtra

def word (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat + data[4*i+1]!.toNat*256 +
    data[4*i+2]!.toNat*65536 + data[4*i+3]!.toNat*16777216

def main (args : List String) : IO UInt32 := do
  let [name,inputPath,outputPath,expectedText] := args |
    throw (IO.userError "Expected mode input output expected-mismatches")
  let some m := parseMode name | throw (IO.userError "Invalid sparse mode")
  unless m.a = "bf16" ∧ m.b = "bf16" ∧ m.acc = "f32" do
    throw (IO.userError "Only BF16 sparse MMA is supported")
  let some expected := expectedText.toNat? | throw (IO.userError "Invalid mismatch count")
  let input ← IO.FS.readBinFile inputPath
  let output ← IO.FS.readBinFile outputPath
  let stride := 16*m.k+160
  unless input.size > 0 ∧ input.size % (4*stride) = 0 ∧
      output.size = input.size/(4*stride)*512 do
    throw (IO.userError "Invalid input/output extent")
  let mut mismatches := 0
  for sample in [:input.size/(4*stride)] do
    let base := sample*stride
    let a := (Array.range (8*m.k)).map fun j => word input (base+j)
    let b := (Array.range (8*m.k)).map fun j => word input (base+8*m.k+j)
    let c := (Array.range 128).map fun j => word input (base+16*m.k+j)
    let e := (Array.range 32).map fun j => word input (base+16*m.k+128+j)
    let some predicted := GB300Math.TensorSparseBF16Blackwell.evaluate m a b c e |
      throw (IO.userError "Rejected sparse operands")
    for cell in [:128] do
      let observed := word output (sample*128+cell)
      if predicted[cell]! != observed then
        if mismatches < 8 then
          IO.println s!"difference sample={sample} cell={cell} predicted={predicted[cell]!} observed={observed}"
        mismatches := mismatches+1
  IO.println s!"cells={output.size/4} mismatches={mismatches} expected={expected}"
  return if mismatches = expected then 0 else 1
