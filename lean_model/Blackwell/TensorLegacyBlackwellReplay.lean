import GB300Math.TensorLegacyBlackwell
open GB300Math.TensorLegacyBlackwell
open GB300Math.SparseTensorExtra

def word (data : ByteArray) (offset : Nat) : Nat := Id.run do
  let mut result := 0
  for j in [:4] do result := result + data[offset+j]!.toNat*2^(8*j)
  return result

def main (args : List String) : IO UInt32 := do
  let [name,modifier,zero,input,output,errors] := args | throw (IO.userError "Expected variant modifier inputs outputs errors")
  let some mods := modifier.toNat? | throw (IO.userError "Bad modifiers")
  let some zero := zero.toNat? | throw (IO.userError "Bad zero-register mask")
  unless zero < 4 do throw (IO.userError "Zero-register mask out of range")
  let some expected := errors.toNat? | throw (IO.userError "Bad expected errors")
  unless mods < 4 do throw (IO.userError "Invalid modifiers")
  let denseMode := name.startsWith "dense_"
  let (kind,acc,k,m) ← if denseMode then do
      let [_,kind,acc,ks] := name.splitOn "_" | throw (IO.userError "Bad dense mode")
      let some k := ks.toNat? | throw (IO.userError "Bad K")
      pure (kind,acc,k,none)
    else do
      let some m := parseMode name | throw (IO.userError "Bad sparse mode")
      unless m.bits ≥ 16 do throw (IO.userError "Integer negate is not modeled")
      pure (m.a,m.acc,m.k,some m)
  unless zero = 0 ∨ zero = 2 ∨ (denseMode ∧ k = (if kind = "tf32" then 4 else 8)) do
    throw (IO.userError "B RZ only modeled for short dense forms")
  let stride := 4*(if denseMode then 24*k+128 else 16*k+160)
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 && xs.size % stride == 0 && ys.size == xs.size/stride*512 do
    throw (IO.userError "Bad matrix extents")
  let mut failures := 0
  for sample in [:xs.size/stride] do
    let base := sample*stride
    let an := if denseMode then 16*k else 8*k
    let a := (Array.range an).map fun j => word xs (base+4*j)
    let b := (Array.range (8*k)).map fun j => if zero % 2 = 1 then 0 else word xs (base+4*(an+j))
    let c := (Array.range 128).map fun j => if zero / 2 % 2 = 1 then 0 else word xs (base+4*(an+8*k+j))
    let value := match m with
      | none => dense kind acc k mods a b c
      | some m => sparse m mods a b c ((Array.range 32).map fun j => word xs (base+4*(16*k+128+j)))
    let some values := value | throw (IO.userError "Out-of-domain input")
    for cell in [:128] do
      let actual := word ys (sample*512+4*cell)
      if values[cell]! != actual then
        if failures < 6 then IO.println s!"witness matrix={sample} cell={cell} model={values[cell]!} actual={actual}"
        failures := failures+1
  IO.println s!"result {name} modifiers={mods} cells={xs.size/stride*128} mismatches={failures} expected={expected}"
  return if failures == expected then 0 else 1
