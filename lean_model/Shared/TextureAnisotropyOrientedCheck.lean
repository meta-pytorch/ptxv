import GB300Math.TextureAnisotropyOriented
open GB300Math

def word (data : ByteArray) (i : Nat) : Nat :=
  data[4*i]!.toNat + data[4*i+1]!.toNat*256 +
    data[4*i+2]!.toNat*65536 + data[4*i+3]!.toNat*16777216

def dyadicBits (n : Int) (shift : Nat) : Nat :=
  let a := n.natAbs
  if a = 0 then 0 else
    let e := a.log2
    (if n < 0 then 2^31 else 0)+(127+e-shift)*2^23+
      (a-2^e)*2^(23-e)

def integerValue (v : Nat) : Int :=
  let e := v/2^23%256
  let n : Nat := (v%2^23+2^23)/2^(150-e)
  if v ≥ 2^31 then -(n:Int) else n

def coordinate (v : Nat) : Int :=
  let e := v/2^23%256
  ((v%2^23+2^23)/2^(134-e):Nat)-128

def main (args : List String) : IO UInt32 := do
  let [inputPath,imagePath,outputPath] := args |
    throw (IO.userError "Expected inputs image outputs")
  let inputs ← IO.FS.readBinFile inputPath
  let image ← IO.FS.readBinFile imagePath
  let outputs ← IO.FS.readBinFile outputPath
  let count := inputs.size/64
  unless count>0 ∧ inputs.size%64=0 ∧ image.size=256*256*16 ∧ outputs.size=count*32+64 do
    throw (IO.userError "Invalid capture extent")
  let mut values : Array Int := #[]
  for i in [:256*256*4] do
    let bits := word image i
    let n := integerValue bits
    unless n.natAbs ≤ 255 ∧ dyadicBits n 0 = bits do
      throw (IO.userError s!"Image outside exact integer domain at {i}")
    values := values.push n
  let mut bad := 0
  for row in [:count] do
    let dx0 := word inputs (row*16+4)
    let dx1 := word inputs (row*16+5)
    let dy0 := word inputs (row*16+8)
    let dy1 := word inputs (row*16+9)
    unless dy0%2^31=0 ∧ dy1%2^31=0 ∧ dx0%2^31≤0x3d800000 ∧ dx1%2^31≤0x3d800000 do
      throw (IO.userError s!"Outside rank-one finite gradient domain at {row}")
    let cx := word inputs (row*16)
    let cy := word inputs (row*16+1)
    let qx := coordinate cx
    let qy := coordinate cy
    unless qx≥2560 ∧ qx≤245*256 ∧ qy≥2560 ∧ qy≤245*256 ∧
        dyadicBits (qx+128) 16=cx ∧ dyadicBits (qy+128) 16=cy do
      throw (IO.userError s!"Outside exact coordinate domain at {row}")
    let log := TextureAnisotropyOriented.majorLog dx0 dx1
    unless log≤1024 do
      throw (IO.userError s!"Major footprint exceeds16 at {row}")
    let direction := TextureAnisotropyOriented.direction dx0 dx1
    let (groups,taps) := TextureAnisotropyOriented.taps log direction.1 direction.2
    for channel in [:4] do
      let mut numerator : Int := 0
      for tap in taps do
        let px := qx+tap.x
        let py := qy+tap.y
        let x := (px/256).toNat
        let y := (py/256).toNat
        let p := TextureAnisotropyOriented.plane tap.weight (px%256).toNat (py%256).toNat
        numerator := numerator+
          (p.1:Int)*values[(y*256+x)*4+channel]!+
          (p.2.1:Int)*values[(y*256+x+1)*4+channel]!+
          (p.2.2.1:Int)*values[((y+1)*256+x)*4+channel]!+
          (p.2.2.2:Int)*values[((y+1)*256+x+1)*4+channel]!
      let expected := dyadicBits numerator (8+groups.log2)
      let actual := word outputs (row*8+channel)
      if expected != actual then
        if bad<8 then IO.println s!"row={row} channel={channel} dx={dx0},{dx1} q={qx},{qy} expected={expected} actual={actual}"
        bad := bad+1
    if word outputs (row*8+4) != 1 then bad := bad+1
    for k in [5:8] do
      if word outputs (row*8+k) != 0xa5a5a5a5 then bad := bad+1
  for k in [:16] do
    if word outputs (count*8+k) != 0xa5a5a5a5 then bad := bad+1
  IO.println s!"rows={count} numeric_words={count*4} state_words={count*4} guard_words=16 differences={bad}"
  return if bad=0 then 0 else 1
