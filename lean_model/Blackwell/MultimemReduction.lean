/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import BulkReductionSpec

set_option maxRecDepth 100000
set_option maxHeartbeats 0
set_option cbv.maxSteps 10000000
namespace GB300Math.MultimemReduction
open FloatLib.Floats.Formats.BinaryInterchange

def base (precision : String) : String :=
  match precision with
  | "f16x2" => "f16"
  | "bf16x2" => "bf16"
  | "e4m3x2" | "e4m3x4" => "e4m3"
  | "e5m2x2" | "e5m2x4" => "e5m2"
  | _ => precision

def bits (precision : String) : Nat :=
  if precision = "e4m3" || precision = "e5m2" then 8
  else if precision = "e4m3x2" || precision = "e5m2x2" then 16
  else if precision.endsWith "x2" || precision.endsWith "x4" then 32
  else BulkReduction.bits precision

def legal (op precision acc : String) (vec : Nat) : Bool :=
  let scalar := base precision
  if scalar ∈ ["f16","bf16","e4m3","e5m2","f32","f64"] then
    bits precision * vec ∈ [32,64,128] && vec ∈ [1,2,4,8] &&
    (scalar ≠ "f64" || vec = 1) &&
    (op = "add" || (scalar ≠ "f32" && scalar ≠ "f64" && op ∈ ["min","max"])) &&
    (acc = "default" || (op = "add" &&
      ((scalar ∈ ["f16","bf16"] && acc = "f32") ||
       (scalar ∈ ["e4m3","e5m2"] && acc = "f16"))))
  else vec = 1 && acc = "default" &&
    (if scalar ∈ ["b32","b64"] then op ∈ ["and","or","xor"]
     else if scalar = "s64" then op ∈ ["min","max"]
     else scalar ∈ ["u32","s32","u64"] && op ∈ ["add","min","max"])

def format (name : String) : Option FloatFormat :=
  match name with
  | "f16" => some FloatFormat.binary16
  | "bf16" => some FloatFormat.bfloat16
  | "f32" => some FloatFormat.binary32
  | "f64" => some FloatFormat.binary64
  | _ => none

def quiet (fmt : FloatFormat) (word : Nat) : Nat := word ||| 2^(fmt.fracWidth-1)

def cast (src dst : FloatFormat) (word : Nat) : Nat :=
  let value : Model src := Model.ofNatBits word
  if Model.isNaN value then
    let payload := (word % 2^src.fracWidth) * 2^dst.fracWidth / 2^src.fracWidth
    quiet dst ((word / 2^(src.bitWidth-1)) * 2^(dst.bitWidth-1) +
      (2^dst.expWidth-1) * 2^dst.fracWidth + payload)
  else Model.toNatBits (Model.castWithRounding src dst value .nearestEven)

def ordinaryAdd (fmt : FloatFormat) (a b : Nat) : Nat :=
  let x : Model fmt := Model.ofNatBits a
  let y : Model fmt := Model.ofNatBits b
  if Model.isNaN x then quiet fmt a
  else if Model.isNaN y then quiet fmt b
  else
    let value := Model.addWithRounding .nearestEven x y
    if Model.isNaN value then (2^fmt.expWidth-1) * 2^fmt.fracWidth + 2^(fmt.fracWidth-1)
    else if Model.toNatBits value % 2^(fmt.bitWidth-1) = 0 then 0
    else Model.toNatBits value

def ordinaryMinMax (fmt : FloatFormat) (maximum : Bool) (a b : Nat) : Nat :=
  let f : PackedMinMax.Format := ⟨fmt.bitWidth, fmt.fracWidth⟩
  if PackedMinMax.isNaN f a && PackedMinMax.isNaN f b then quiet fmt a
  else PackedMinMax.select f maximum false a b

def workType (precision acc : String) : String := if acc = "default" then precision else acc

def load (precision acc : String) (word : Nat) : Option Nat := do
  if acc = "default" then return word
  return cast (← format precision) (← format acc) word

def finish (precision acc : String) (word : Nat) : Option Nat := do
  if acc = "default" then return word
  return cast (← format acc) (← format precision) word

def combine (op precision : String) (a b : Nat) : Option Nat := do
  if let some fmt := format precision then
    if op = "add" then return ordinaryAdd fmt a b
    else if op = "min" || op = "max" then return ordinaryMinMax fmt (op = "max") a b
    else none
  else if precision = "e4m3" || precision = "e5m2" then
    if op ≠ "min" && op ≠ "max" then none else do
      let fraction := if precision = "e4m3" then 3 else 2
      let fmt : PackedMinMax.Format := ⟨8, fraction⟩
      if PackedMinMax.isNaN fmt a && PackedMinMax.isNaN fmt b then
        return a ||| 2^(fraction-1)
      return PackedMinMax.select fmt (op = "max") false a b
  else BulkReduction.evaluate op precision a b

def parenthesizations (op precision : String) (a b c d : Nat) : Option (List Nat) := do
  let ab ← combine op precision a b
  let bc ← combine op precision b c
  let cd ← combine op precision c d
  return [← combine op precision (← combine op precision ab c) d,
          ← combine op precision (← combine op precision a bc) d,
          ← combine op precision ab cd,
          ← combine op precision a (← combine op precision bc d),
          ← combine op precision a (← combine op precision b cd)]

def orders : List (List Nat) := Id.run do
  let mut result := []
  for a in [:4] do
    for b in [:4] do
      for c in [:4] do
        for d in [:4] do
          if a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d then
            result := [a,b,c,d] :: result
  return result

def insertOrdered (n : Nat) : List Nat → List Nat
  | [] => [n]
  | a :: rest => if n < a then n :: a :: rest
    else if n = a then a :: rest else a :: insertOrdered n rest

def uniqueSorted : List Nat → List Nat
  | [] => []
  | a :: rest => insertOrdered a (uniqueSorted rest)

def exactFiniteSum (precision : String) (inputs : List Nat) : Option Nat := do
  let fmt ← format precision
  let values := inputs.toArray.map fun word => (Model.ofNatBits word : Model fmt)
  unless values.all Model.isFinite do none
  let result := Model.toNatBits (Model.sum fmt values .nearestEven)
  return if result % 2^(fmt.bitWidth-1) = 0 then 0 else result

def candidates (op precision acc : String) (inputs : List Nat) : Option (List Nat) := do
  let values ← inputs.mapM (load precision acc)
  let work := workType precision acc
  let raw ← match values with
    | [a,b] => pure [← combine op work a b, ← combine op work b a]
    | [_,_,_,_] => do
        let cases ← orders.mapM fun order =>
          parenthesizations op work values[order[0]!]! values[order[1]!]!
            values[order[2]!]! values[order[3]!]!
        pure cases.flatten
    | _ => none
  let converted ← raw.mapM (finish precision acc)
  let additional := if op = "add" then (exactFiniteSum precision inputs).toList else []
  return uniqueSorted (converted ++ additional)

theorem exactly_24_orders : orders.length = 24 := by decide +kernel

theorem bf16_default_can_lose_unit :
    candidates "add" "bf16" "default" [0x4380, 0x3f80, 0xc380, 0] =
      some [0, 0x3f80] := by cbv

theorem bf16_wide_preserves_unit :
    candidates "add" "bf16" "f32" [0x4380, 0x3f80, 0xc380, 0] =
      some [0x3f80] := by decide +kernel

theorem f32_left_chain_loses_unit :
    (do let ab ← combine "add" "f32" 0x4b800000 0x3f800000
        combine "add" "f32" ab 0xcb800000) = some 0 := by decide +kernel

theorem f32_pairwise_preserves_unit :
    (do let ac ← combine "add" "f32" 0x4b800000 0xcb800000
        combine "add" "f32" ac 0x3f800000) = some 0x3f800000 := by decide +kernel

#print axioms exactly_24_orders
#print axioms bf16_default_can_lose_unit
#print axioms bf16_wide_preserves_unit
#print axioms f32_left_chain_loses_unit
#print axioms f32_pairwise_preserves_unit

theorem f32_negative_zeros_add_to_positive_zero :
    ordinaryAdd FloatFormat.binary32 0x80000000 0x80000000 = 0 := by decide +kernel

theorem f32_add_quiets_preserving_payload :
    ordinaryAdd FloatFormat.binary32 0x7f812345 0x3f800000 = 0x7fc12345 := by decide +kernel

theorem half_nan_payload_survives_widening :
    cast FloatFormat.binary32 FloatFormat.binary16
      (cast FloatFormat.binary16 FloatFormat.binary32 0xfd01) = 0xff01 := by decide +kernel

#print axioms f32_negative_zeros_add_to_positive_zero
#print axioms f32_add_quiets_preserving_payload
#print axioms half_nan_payload_survives_widening

theorem exact_f32_witness_sum :
    exactFiniteSum "f32" [2319920140,1732455079,1730440352,3213447358] = some 1739836323 := by decide +kernel

#print axioms exact_f32_witness_sum

def read (data : ByteArray) (offset size : Nat) : Nat := Id.run do
  let mut result := 0
  for j in [:size] do result := result + data[offset+j]!.toNat * 2^(8*j)
  return result
end GB300Math.MultimemReduction
open GB300Math.MultimemReduction

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected multicast reduction manifest")
  let lines := (← IO.FS.readFile manifest).splitOn "\n" |>.filter (! ·.isEmpty)
  unless !lines.isEmpty do throw (IO.userError "Empty multicast reduction manifest")
  let mut cache : Std.HashMap (String × String × String × List Nat) (List Nat) := {}
  let mut failed := 0
  for line in lines do
    let [op, precision, acc, vecText, teamText, input, output, expectedText] := line.splitOn "\t" |
      throw (IO.userError "Malformed multicast reduction row")
    let some vec := vecText.toNat? | throw (IO.userError "Invalid vector width")
    let some team := teamText.toNat? | throw (IO.userError "Invalid team size")
    let some expected := expectedText.toNat? | throw (IO.userError "Invalid expected count")
    unless legal op precision acc vec ∧ (team = 2 ∨ team = 4) do
      throw (IO.userError "Illegal reduction form")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 64 = 0 ∧ ys.size = xs.size/4 do
      throw (IO.userError "Invalid reduction record sizes")
    let scalar := base precision
    unless (scalar ≠ "e4m3" ∧ scalar ≠ "e5m2") ∨ (op = "min" ∨ op = "max") do
      throw (IO.userError "FP8 multicast arithmetic model is unresolved")
    let bytes := bits scalar / 8
    let lanes := bits precision * vec / bits scalar
    let mut errors := 0
    let mut ambiguous := 0
    for row in [:xs.size/64] do
      let mut matched := true
      let mut multiple := false
      for lane in [:16/bytes] do
        let actual := read ys (row*16+lane*bytes) bytes
        if lane ≥ lanes then
          if actual ≠ 0 then matched := false
        else
          let values := (List.range team).map fun rank => read xs (row*64+rank*16+lane*bytes) bytes
          let key := (op,scalar,acc,values.mergeSort (· ≤ ·))
          let targets ← match cache[key]? with
            | some result => pure result
            | none => do
              let some result := candidates op scalar acc values |
                throw (IO.userError "Unsupported reduction model")
              cache := cache.insert key result
              pure result
          if !(targets.contains actual) then
            if errors < 4 then IO.println s!"witness {op} {precision} {acc} row={row} lane={lane} input={values} candidates={targets} actual={actual}"
            matched := false
          if targets.length > 1 then multiple := true
      if !matched then errors := errors+1
      if multiple then ambiguous := ambiguous+1
    IO.println s!"ambiguity {output} records={ambiguous}"
    IO.println s!"result {output} records={xs.size/64} mismatches={errors} expected={expected}"
    if errors ≠ expected then failed := failed+1
  return if failed = 0 then 0 else 1
