import GB300Math.MufuSm103

def readWord (bytes : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut word := 0
  for i in [:4] do word := word + bytes[offset + i]!.toNat * 2 ^ (8 * i)
  return word

def main (args : List String) : IO UInt32 := do
  let [op, input, output, inputStride, outputStride] := args |
    throw (IO.userError "Expected native SM103 operation input output input-stride output-stride")
  let some si := inputStride.toNat? | throw (IO.userError "Invalid input stride")
  let some so := outputStride.toNat? | throw (IO.userError "Invalid output stride")
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless si ≥ 4 ∧ so ≥ 4 ∧ xs.size > 0 ∧ xs.size % si = 0 ∧ ys.size = xs.size / si * so do
    throw (IO.userError "Invalid byte lengths or strides")
  for i in [:xs.size / si] do
    let x := readWord xs (i * si)
    let some expected := GB300Math.MufuSm103.evaluate op x |
      throw (IO.userError s!"Unsupported SM103 operation: {op}")
    let actual := readWord ys (i * so)
    unless actual = expected do
      throw (IO.userError s!"Mismatch: row={i} input={x} model={expected} hardware={actual}")
  IO.println s!"SM103 {op}: {xs.size / si} words match"
  return 0
