import GB300Math.MatrixStore

def readWord (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat + data[4*i+1]!.toNat*256 +
    data[4*i+2]!.toNat*65536 + data[4*i+3]!.toNat*16777216

def main (args : List String) : IO UInt32 := do
  let [shapeText,numText,transposeText,inputPath,outputPath,expectedText] := args |
    throw (IO.userError "Expected shape matrices transpose input output expected-mismatches/predict")
  let some shape := shapeText.toNat? | throw (IO.userError "Invalid shape")
  let some num := numText.toNat? | throw (IO.userError "Invalid matrices")
  let some transpose := transposeText.toNat? | throw (IO.userError "Invalid transpose")
  unless (shape = 8 ∨ shape = 16) ∧ (num = 1 ∨ num = 2 ∨ num = 4) ∧ transpose ≤ 1 do
    throw (IO.userError "Unsupported matrix store")
  let input ← IO.FS.readBinFile inputPath
  unless input.size > 0 ∧ input.size % 512 = 0 do
    throw (IO.userError "Invalid input extent")
  let predict := expectedText = "predict"
  let expected := expectedText.toNat?.getD 0
  unless predict ∨ expectedText.toNat?.isSome do
    throw (IO.userError "Invalid mismatch count")
  let output ← if predict then pure ByteArray.empty else IO.FS.readBinFile outputPath
  unless predict ∨ output.size = input.size do
    throw (IO.userError "Invalid output extent")
  let mut result := ByteArray.empty
  let mut mismatches := 0
  for sample in [:input.size/512] do
    let words := (Array.range 128).map fun j => readWord input (sample*128+j)
    for cell in [:128] do
      let predicted := GB300Math.MatrixStore.word shape transpose num words cell
      if predict then
        for b in [:4] do
          result := result.push (UInt8.ofNat (predicted / 2^(8*b) % 256))
      else
        let observed := readWord output (sample*128+cell)
        if predicted != observed then
          if mismatches < 8 then
            IO.println s!"difference sample={sample} cell={cell} predicted={predicted} observed={observed}"
          mismatches := mismatches+1
  if predict then IO.FS.writeBinFile outputPath result
  IO.println s!"cells={input.size/4} mismatches={mismatches} expected={expected}"
  return if mismatches = expected then 0 else 1
