import GB300Math.MultisampleFetch

def readWord (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat + data[4*i+1]!.toNat*256 +
    data[4*i+2]!.toNat*65536 + data[4*i+3]!.toNat*16777216

def main (args : List String) : IO UInt32 := do
  let [samplesText,layersText,path,expectedText] := args |
    throw (IO.userError "Expected samples layers output expected_mismatches")
  let [some samples,some layers,some expected] :=
      [samplesText,layersText,expectedText].map String.toNat? |
    throw (IO.userError "Invalid numeric argument")
  unless [2,4,8].contains samples ∧ [1,2].contains layers do
    throw (IO.userError "Unsupported sample/layer shape")
  let output ← IO.FS.readBinFile path
  let size := samples*layers*1024
  unless output.size = (size+64)*4 do
    throw (IO.userError "Invalid output extent")
  let mut mismatches := 0
  for index in [:size+64] do
    let predicted := if index < size then GB300Math.MultisampleFetch.word samples index else 0xdeadbeef
    let observed := readWord output index
    if predicted != observed then
      if mismatches < 8 then IO.println s!"difference index={index} predicted={predicted} observed={observed}"
      mismatches := mismatches+1
  IO.println s!"words={size} guards=64 mismatches={mismatches} expected={expected}"
  return if mismatches = expected then 0 else 1
