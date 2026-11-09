import GB300Math.TextureGradient3D
open GB300Math.TextureGradient3D

def word (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut n := 0
  for j in [:4] do n := n+data[offset+j]!.toNat*2^(8*j)
  return n

def main (args : List String) : IO UInt32 := do
  let [input,output] := args | throw (IO.userError "Expected inputs outputs")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 ∧ xs.size%64 = 0 ∧ ys.size = xs.size/2 do
    throw (IO.userError "Invalid capture sizes")
  let mut bad := 0
  for i in [:xs.size/64] do
    unless (Array.range 3).all (fun lane => word xs (i*64+lane*4)%2^31 < 0x7f800000) do
      throw (IO.userError "Unsupported 3D gradient domain")
    let expected := sample (word xs (i*64)) (word xs (i*64+4)) (word xs (i*64+8)) (word xs (i*64+16)) (word xs (i*64+20)) (word xs (i*64+24)) (word xs (i*64+32)) (word xs (i*64+36)) (word xs (i*64+40))
    for lane in [:8] do
      let value := if lane < 4 then expected else if lane = 4 then 1 else 0
      let actual := word ys (i*32+lane*4)
      if actual ≠ value then
        if bad < 3 then IO.println s!"Mismatch row={i} lane={lane} model={value} hardware={actual}"
        bad := bad+1
  IO.println s!"result count={xs.size/8} mismatches={bad}"
  return if bad = 0 then 0 else 1
