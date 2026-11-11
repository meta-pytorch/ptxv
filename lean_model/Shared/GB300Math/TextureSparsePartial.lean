import Std

namespace GB300Math.TextureSparsePartial

def word (bytes : ByteArray) (index : Nat) : Nat :=
  (List.range 4).foldl (fun v i => v + bytes[index*4+i]!.toNat * 2^(8*i)) 0

def signed (n : Nat) : Int := if n < 2^31 then n else (n : Int)-2^32

def floatFloor (bits extra : Nat) : Int :=
  let e := bits / 2^23 % 256
  let m := bits % 2^23 + if e = 0 then 0 else 2^23
  let shift := (if e = 0 then 1 else e) + extra
  let numerator := if shift ≥ 150 then m * 2^(shift-150) else m
  let denominator := if shift ≥ 150 then 1 else 2^(150-shift)
  if bits / 2^31 = 1 then -((numerator+denominator-1)/denominator : Nat)
  else numerator/denominator

def clampIndex (extent : Nat) (i : Int) : Nat := min (extent-1) i.toNat

theorem clamp_bounded (extent : Nat) (i : Int) (h : 0 < extent) :
    clampIndex extent i < extent := by
  have := Nat.min_le_left (extent-1) i.toNat
  simp only [clampIndex]
  omega

theorem negative_clamps (extent : Nat) (i : Int) (h : i < 0) :
    clampIndex extent i = 0 := by
  have hi : i.toNat = 0 := Int.toNat_eq_zero.mpr (by omega)
  simp [clampIndex, hi]

def mapped (x y : Nat) : Bool := x < 64 && y < 64

def resident (footprint : Array (Nat × Nat)) : Bool :=
  footprint.all (fun xy => mapped xy.1 xy.2)

theorem mixed_footprint : resident #[(63,63),(64,63),(64,62),(63,62)] = false := by simp [resident, mapped, Array.all]

theorem mapped_member : mapped 63 63 = true := by decide

end GB300Math.TextureSparsePartial

#print axioms GB300Math.TextureSparsePartial.clamp_bounded
#print axioms GB300Math.TextureSparsePartial.negative_clamps
#print axioms GB300Math.TextureSparsePartial.mixed_footprint
#print axioms GB300Math.TextureSparsePartial.mapped_member

open GB300Math.TextureSparsePartial

def main (args : List String) : IO UInt32 := do
  let [geometry, ctype, component, input, texels, output] := args |
    throw (IO.userError "Expected geometry, coordinate type, component, inputs, texels, outputs")
  unless geometry = "2d" && ctype ∈ ["s32","f32"] do
    throw (IO.userError "Unsupported partial sparse configuration")
  let some component := component.toInt? | throw (IO.userError "Invalid component")
  unless component ≥ -1 && component < 4 && (component < 0 || ctype = "f32") do
    throw (IO.userError "Unsupported gather")
  let xs ← IO.FS.readBinFile input
  let ts ← IO.FS.readBinFile texels
  let ys ← IO.FS.readBinFile output
  unless xs.size > 0 && xs.size % 32 = 0 && xs.size = ys.size && ts.size = 128*128*16 do
    throw (IO.userError "Invalid capture extent")
  let mut failures := 0
  for i in [:xs.size/32] do
    unless word xs (8*i+3) = 0 do throw (IO.userError "Layer outside tested domain")
    let mut coordinates : Array Int := #[]
    for axis in [:3] do
      let coordinate := word xs (8*i+axis)
      let offset := signed (word xs (8*i+4+axis))
      unless -8 ≤ offset && offset ≤ 7 do throw (IO.userError "Offset outside PTX domain")
      unless ctype = "s32" || coordinate / 2^23 % 256 < 255 do
        throw (IO.userError "Nonfinite coordinate outside this fixture")
      let pos := if ctype = "s32" then signed coordinate else
        if component ≥ 0 then (floatFloor coordinate 1 - 1)/2 else floatFloor coordinate 0
      coordinates := coordinates.push (pos+offset)
    let oob := ctype = "s32" && (coordinates[0]! < 0 || coordinates[0]! ≥ 128 ||
      coordinates[1]! < 0 || coordinates[1]! ≥ 128)
    let footprint := (List.range 4).toArray.map fun lane =>
      let dx := if component ≥ 0 && lane ∈ [1,2] then 1 else 0
      let dy := if component ≥ 0 && lane ∈ [0,1] then 1 else 0
      (clampIndex 128 (coordinates[0]!+dx), clampIndex 128 (coordinates[1]!+dy))
    for lane in [:8] do
      let expected := if lane ≥ 4 then
        if lane = 4 && (oob || resident footprint) then 1 else 0
        else
          let (x,y) := footprint[lane]!
          if oob || !mapped x y then 0
          else word ts ((x+128*y)*4+(if component < 0 then lane else component.toNat))
      let actual := word ys (8*i+lane)
      if expected != actual then
        if failures < 5 then IO.println s!"witness row={i} lane={lane} expected={expected} actual={actual}"
        failures := failures+1
  IO.println s!"result count={xs.size/4} mismatches={failures}"
  return if failures = 0 then 0 else 1
