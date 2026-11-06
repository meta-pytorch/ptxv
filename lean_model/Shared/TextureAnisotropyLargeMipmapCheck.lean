import GB300Math.TextureAnisotropyLargeMipmap
open GB300Math

def word (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat+data[4*i+1]!.toNat*256+
    data[4*i+2]!.toNat*65536+data[4*i+3]!.toNat*16777216

def dyadicBits (n : Int) (shift : Nat) : Nat :=
  let a := n.natAbs
  if a=0 then 0 else
    let e := a.log2
    (if n<0 then 2^31 else 0)+(127+e-shift)*2^23+(a-2^e)*2^(23-e)

def integerValue (v : Nat) : Int :=
  let e := v/2^23%256
  let n : Nat := (v%2^23+2^23)/2^(150-e)
  if v≥2^31 then -(n:Int) else n

def coordinate (v : Nat) : Int :=
  ((v%2^23+2^23)/2^(134-v/2^23%256):Nat)-128

def main (args : List String) : IO UInt32 := do
  let [anisotropyText,optimizedText,inputPath,imageDirectory,pattern,outputPath] := args |
    throw (IO.userError "Expected anisotropy optimized inputs image_directory pattern outputs")
  let some anisotropy := anisotropyText.toNat? | throw (IO.userError "Invalid anisotropy")
  unless anisotropy≥1 ∧ anisotropy≤16 do
    throw (IO.userError "Unsupported anisotropy")
  unless optimizedText="0" ∨ optimizedText="1" do
    throw (IO.userError "Invalid optimization selector")
  let optimized := optimizedText="1"
  let inputs ← IO.FS.readBinFile inputPath
  let outputs ← IO.FS.readBinFile outputPath
  let count := inputs.size/64
  unless count>0 ∧ inputs.size%64=0 ∧ outputs.size=count*32+64 do
    throw (IO.userError "Invalid capture extent")
  let mut images : Array (Array (Array Nat)) := #[]
  for channel in [:4] do
    let mut planes := #[]
    for level in [:9] do
      let image ← IO.FS.readBinFile (imageDirectory++s!"/plane-{pattern}-{level}.bin")
      let size := 256/2^level
      unless image.size=size*size*16 do throw (IO.userError "Invalid image extent")
      let mut values := #[]
      for pixel in [:size*size] do
        let bits := word image (pixel*4+channel)
        values := values.push bits
      planes := planes.push values
    images := images.push planes
  let mut bad := 0
  for row in [:count] do
    let dx0 := word inputs (row*16+4)
    let dx1 := word inputs (row*16+5)
    let dy0 := word inputs (row*16+8)
    let dy1 := word inputs (row*16+9)
    unless dy0/2^23%256<255 ∧ dy1/2^23%256<255 ∧ dx0/2^23%256<255 ∧ dx1/2^23%256<255 do
      throw (IO.userError s!"Outside finite gradient domain at {row}")
    unless (TextureAnisotropyRankTwo.selection dx0 dx1 dy0 dy1).logarithm<2048 do
      throw (IO.userError s!"Outside mip width domain at {row}")
    let xb := word inputs (row*16)
    let yb := word inputs (row*16+1)
    unless xb≥0x3e000000 ∧ xb≤0x3f600000 ∧ yb≥0x3e000000 ∧ yb≤0x3f600000 do
      throw (IO.userError s!"Outside positive coordinate domain at {row}")
    for channel in [:4] do
      let expected := TextureAnisotropyLargeMipmap.evaluate dx0 dx1 dy0 dy1 anisotropy optimized xb yb images[channel]!
      let actual := word outputs (row*8+channel)
      if expected≠actual then
        if bad<8 then IO.println s!"row={row} channel={channel} expected={expected} actual={actual}"
        bad := bad+1
    if word outputs (row*8+4)≠1 then bad := bad+1
    for k in [5:8] do
      if word outputs (row*8+k)≠0xa5a5a5a5 then bad := bad+1
  for k in [:16] do
    if word outputs (count*8+k)≠0xa5a5a5a5 then bad := bad+1
  IO.println s!"rows={count} numeric_words={count*4} state_words={count*4} guard_words=16 differences={bad}"
  return if bad=0 then 0 else 1
