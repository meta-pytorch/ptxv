import GB300Math.TextureDepth24

open GB300Math

def readWord (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat + data[4*i+1]!.toNat*256 +
    data[4*i+2]!.toNat*65536 + data[4*i+3]!.toNat*16777216

def main (args : List String) : IO UInt32 := do
  let [path,formatText,baseText,countText,shiftText,offsetText,expectedText] := args |
    throw (IO.userError "Expected output format base count shift offset expected_mismatches")
  let [some format,some base,some count,some shift,some offset,some expected] :=
      [formatText,baseText,countText,shiftText,offsetText,expectedText].map String.toNat? |
    throw (IO.userError "Invalid numeric argument")
  unless (format = 16 || format = 24) && count > 0 && base+count ≤ 2^32 && shift ≤ 16 && offset ≤ 1 do
    throw (IO.userError "Unsupported descriptor")
  let output ← IO.FS.readBinFile path
  let samples := count / 2^shift
  unless output.size = 4*(64+3*samples+64) do throw (IO.userError "Invalid extent")
  let mut mismatches := 0
  if readWord output 0 != count then mismatches := mismatches+1
  if readWord output 1 != (if offset = 0 then 0 else count) then mismatches := mismatches+1
  if offset = 0 && readWord output 2 != 0xffffffff then mismatches := mismatches+1
  for i in [:samples] do
    let word := readWord output (64+3*i)
    let actualCode := readWord output (65+3*i)
    let answer := readWord output (66+3*i)
    unless base+i*2^shift ≤ word && word < base+(i+1)*2^shift do
      throw (IO.userError "Companion word outside its disjoint scan interval")
    let value := if format = 16 then TextureDepthNormalized.reference word else TextureDepth24.reference word
    if actualCode != (value+offset) % 2^format then mismatches := mismatches+1
    if answer != (if offset = 0 then 0x3f800000 else 0) then mismatches := mismatches+1
  for i in [:64] do
    if readWord output (64+3*samples+i) != 0xcafebabe then mismatches := mismatches+1
  IO.println s!"companions={samples} guards=64 mismatches={mismatches} expected={expected}"
  return if mismatches = expected then 0 else 1
