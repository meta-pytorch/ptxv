import GB300Math.WmmaArithmetic

def readWord (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat + data[4*i+1]!.toNat*256 +
    data[4*i+2]!.toNat*65536 + data[4*i+3]!.toNat*16777216

def main (args : List String) : IO UInt32 := do
  let [mText,nText,kText,typ,dtype,mode,inputPath,outputPath,expectedText] := args |
    throw (IO.userError "Expected m n k type dtype mode input output expected/predict")
  let [some m,some n,some k] := [mText,nText,kText].map String.toNat? |
    throw (IO.userError "Invalid dimensions")
  unless m*n ≤ 256 ∧ m*k+k*n+m*n ≤ 4096 do
    throw (IO.userError "Invalid WMMA shape")
  let input ← IO.FS.readBinFile inputPath
  unless input.size > 0 ∧ input.size % 16384 = 0 do
    throw (IO.userError "Invalid input extent")
  let predict := expectedText = "predict"
  let expected := expectedText.toNat?.getD 0
  unless predict ∨ expectedText.toNat?.isSome do
    throw (IO.userError "Invalid mismatch count")
  let output ← if predict then pure ByteArray.empty else IO.FS.readBinFile outputPath
  unless predict ∨ output.size = input.size/4 do
    throw (IO.userError "Invalid output extent")
  let mut result := ByteArray.empty
  let mut mismatches := 0
  for sample in [:input.size/16384] do
    let words := (Array.range 4096).map fun j => readWord input (sample*4096+j)
    for cell in [:1024] do
      let predicted := GB300Math.WmmaArithmetic.word m n k typ dtype mode words cell
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
  IO.println s!"cells={input.size/16} mismatches={mismatches} expected={expected}"
  return if mismatches = expected then 0 else 1
