import GB300Math.TextureAnisotropyGeneral
import GB300Math.TextureAnisotropyBlend
open GB300Math

def word (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat + data[4*i+1]!.toNat*256 +
    data[4*i+2]!.toNat*65536 + data[4*i+3]!.toNat*16777216

def finiteNormal (v : Nat) : Bool :=
  let e := v/2^23%256
  0<e && e<255

def main (args : List String) : IO UInt32 := do
  let [aText, disabledText, domainText, inputPath, outputPath] := args |
    throw (IO.userError "Expected anisotropy disabled domain input output")
  let some a := aText.toNat? | throw (IO.userError "Invalid anisotropy")
  unless [1,2,4,8,16].contains a do
    throw (IO.userError "Unsupported anisotropy")
  unless ["0","1"].contains disabledText do
    throw (IO.userError "Invalid trilinear optimization selector")
  unless ["normal", "all"].contains domainText do
    throw (IO.userError "Invalid input domain selector")
  let inputs ← IO.FS.readBinFile inputPath
  let outputs ← IO.FS.readBinFile outputPath
  let count := inputs.size/64
  unless count>0 ∧ inputs.size%64=0 ∧ outputs.size=count*32+64 do
    throw (IO.userError "Invalid capture extent")
  let mut bad := 0
  for row in [:count] do
    unless word inputs (row*16)=0x3f000000 ∧ word inputs (row*16+1)=0x3f000000 do
      throw (IO.userError "Expected fixed center coordinates")
    let dx0 := word inputs (row*16+4)
    let dx1 := word inputs (row*16+5)
    let dy0 := word inputs (row*16+8)
    let dy1 := word inputs (row*16+9)
    unless domainText="all" || [dx0,dx1,dy0,dy1].all (fun v => finiteNormal v || v%2^31=0) do
      throw (IO.userError s!"Outside finite normal or zero gradient domain at row {row}")
    let q := TextureAnisotropyGeneral.position dx0 dx1 dy0 dy1 a
    let q := if disabledText="1" then q else TextureAnisotropyBlend.remap q a
    let expected := TextureMipmap.encodeQ8 (256+q)
    for channel in [:4] do
      let observed := word outputs (row*8+channel)
      if observed != expected then
        if bad<8 then IO.println s!"row={row} channel={channel} expected={expected} observed={observed}"
        bad := bad+1
    if word outputs (row*8+4) != 1 then bad := bad+1
    for k in [5:8] do
      if word outputs (row*8+k) != 0xa5a5a5a5 then bad := bad+1
  for k in [:16] do
    if word outputs (count*8+k) != 0xa5a5a5a5 then bad := bad+1
  IO.println s!"rows={count} numeric_words={count*4} state_words={count*4} guard_words=16 differences={bad}"
  return if bad=0 then 0 else 1
