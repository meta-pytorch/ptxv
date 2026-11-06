import GB300Math.WmmaMovement

def readWord (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat + data[4*i+1]!.toNat*256 +
    data[4*i+2]!.toNat*65536 + data[4*i+3]!.toNat*16777216

def main (args : List String) : IO UInt32 := do
  let [mText,nText,sizeText,aText,bText,padText,inputPath,outputPath,expectedText] := args |
    throw (IO.userError "Expected m n size input-layout output-layout pad input output expected/predict")
  let params := [mText,nText,sizeText,aText,bText,padText].map String.toNat?
  let [some m,some n,some size,some a,some b,some pad] := params |
    throw (IO.userError "Invalid dimensions")
  unless (m = 8 ∨ m = 16 ∨ m = 32) ∧ (n = 8 ∨ n = 16 ∨ n = 32) ∧
      (size = 2 ∨ size = 4 ∨ size = 8) ∧ a ≤ 1 ∧ b ≤ 1 ∧ pad ≤ 1 do
    throw (IO.userError "Unsupported matrix movement")
  let input ← IO.FS.readBinFile inputPath
  unless input.size > 0 ∧ input.size % 4096 = 0 do
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
  for sample in [:input.size/4096] do
    let words := (Array.range 1024).map fun j => readWord input (sample*1024+j)
    for cell in [:1024] do
      let predicted := GB300Math.WmmaMovement.word m n size a b pad words cell
      if predict then
        for i in [:4] do
          result := result.push (UInt8.ofNat (predicted / 2^(8*i) % 256))
      else
        let observed := readWord output (sample*1024+cell)
        if predicted != observed then
          if mismatches < 8 then
            IO.println s!"difference sample={sample} cell={cell} predicted={predicted} observed={observed}"
          mismatches := mismatches+1
  if predict then IO.FS.writeBinFile outputPath result
  IO.println s!"cells={input.size/4} mismatches={mismatches} expected={expected}"
  return if mismatches = expected then 0 else 1
