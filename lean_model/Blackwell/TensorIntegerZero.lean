import GB300Math.TensorInteger
import GB300Math.SparseTensorExtra

namespace GB300Math.TensorIntegerZero

def select (mask bit : Nat) (values : Array Nat) : Array Nat :=
  if mask / 2^bit % 2 = 1 then values.map fun _ => 0 else values

theorem selected_is_zero (xs : Array Nat) : select 1 0 xs = xs.map (fun _ => 0) := by
  simp [select]

theorem binary_xor_zero_retains_other_operand :
    TensorInteger.dot ⟨1,8,128,false,false,false,true⟩ #[0,0,0,0] #[1,0,1,1] 7 = 10 := by
  decide +kernel

#print axioms selected_is_zero
#print axioms binary_xor_zero_retains_other_operand

end GB300Math.TensorIntegerZero

open GB300Math.TensorIntegerZero

def main (args : List String) : IO UInt32 := do
  let [name,zeroMask,input,output,errors] := args | throw (IO.userError "Expected mode zero input output errors")
  let some zeroMask := zeroMask.toNat? | throw (IO.userError "Bad zero mask")
  let some errors := errors.toNat? | throw (IO.userError "Bad expected errors")
  unless zeroMask > 0 ∧ zeroMask < 8 do throw (IO.userError "Invalid zero mask")
  let dense := name.startsWith "dense_"
  let (m,k,bits,dm,sm) ← if dense then do
      let some mode := GB300Math.TensorInteger.mode ((name.drop 6).toString) | throw (IO.userError "Bad dense mode")
      pure (mode.m,mode.k,mode.bits,some mode,none)
    else do
      let some mode := GB300Math.SparseTensorExtra.parseMode name | throw (IO.userError "Bad sparse mode")
      unless mode.bits = 4 ∨ mode.bits = 8 do throw (IO.userError "Only sparse integer forms")
      pure (16,mode.k,mode.bits,none,some mode)
  unless (zeroMask % 2 = 0 ∨ (dense ∧ m = 8)) ∧
      (zeroMask / 2 % 2 = 0 ∨ (dense ∧ k = 128/bits)) do
    throw (IO.userError "Unmodeled zero-register group")
  let an := if dense then m*k else 8*k
  let stride := 4*(an+8*k+m*8+(if dense then 0 else 32))
  let outStride := m*8*4
  let xs ← IO.FS.readBinFile input
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 ∧ xs.size % stride = 0 ∧ ys.size = xs.size/stride*outStride do
    throw (IO.userError "Bad full-matrix extents")
  let mut bad := 0
  for sample in [:xs.size/stride] do
    let base := sample*stride
    let a := select zeroMask 0 ((Array.range an).map fun j => GB300Math.TensorInteger.word xs (base+4*j))
    let b := select zeroMask 1 ((Array.range (8*k)).map fun j => GB300Math.TensorInteger.word xs (base+4*(an+j)))
    let c := select zeroMask 2 ((Array.range (m*8)).map fun j => GB300Math.TensorInteger.word xs (base+4*(an+8*k+j)))
    let expected ← match dm,sm with
      | some mode,none => do
        let mut result := #[]
        for row in [:m] do
          for col in [:8] do
            result := result.push (GB300Math.TensorInteger.dot mode
              ((Array.range k).map fun j => a[row*k+j]!)
              ((Array.range k).map fun j => b[col*k+j]!) c[row*8+col]!)
        pure result
      | none,some mode => do
        let e := (Array.range 32).map fun j => GB300Math.TensorInteger.word xs (base+4*(an+8*k+m*8+j))
        let some result := GB300Math.SparseTensorExtra.evaluate mode a b c e | throw (IO.userError "Out-of-domain sparse input")
        pure result
      | _,_ => throw (IO.userError "Invalid model selection")
    for cell in [:m*8] do
      let actual := GB300Math.TensorInteger.word ys (sample*outStride+4*cell)
      if expected[cell]! ≠ actual then
        if bad < 4 then IO.println s!"Mismatch {name} sample={sample} cell={cell} expected={expected[cell]!} actual={actual}"
        bad := bad+1
  IO.println s!"result {name} cells={xs.size/stride*m*8} mismatches={bad} expected={errors}"
  return if bad = errors then 0 else 1
