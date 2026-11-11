/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std.Tactic

namespace GB300Math.SmallConversions

structure Format where
  exponentBits : Nat
  fractionBits : Nat
  bias : Nat
  limit : Nat
  deriving Repr

def format : String → Option Format
  | "e4m3" => some ⟨4, 3, 7, 126⟩
  | "e5m2" => some ⟨5, 2, 15, 123⟩
  | "e2m3" => some ⟨2, 3, 1, 31⟩
  | "e3m2" => some ⟨3, 2, 3, 31⟩
  | "e2m1" => some ⟨2, 1, 1, 7⟩
  | _ => none

def nearestEven (m d : Nat) : Nat :=
  let q := m / 2^d
  let r := m % 2^d
  q + if 2*r > 2^d ∨ (2*r = 2^d ∧ q % 2 = 1) then 1 else 0

def roundShift (m : Nat) (d : Int) : Nat :=
  if d ≤ 0 then m * 2^(-d).toNat else nearestEven m d.toNat

theorem nearest_even_grid (m d : Nat) : nearestEven (m * 2^d) d = m := by
  have hp : 0 < 2^d := Nat.two_pow_pos d
  have hn : 0 ≠ 2^d := by omega
  simp [nearestEven, Nat.mul_div_left m hp, hn]

theorem nearest_even_between (m d : Nat) :
    m / 2^d ≤ nearestEven m d ∧ nearestEven m d ≤ m / 2^d + 1 := by
  simp only [nearestEven]
  split <;> omega

theorem nearest_even_half_quantum (m d : Nat) :
    2*m ≤ 2*(nearestEven m d * 2^d) + 2^d ∧
    2*(nearestEven m d * 2^d) ≤ 2*m + 2^d := by
  have hp : 0 < 2^d := Nat.two_pow_pos d
  have hr := Nat.mod_lt m hp
  have hm := Nat.mod_add_div m (2^d)
  rw [Nat.mul_comm (2^d) (m / 2^d)] at hm
  simp only [nearestEven]
  split <;> simp only [Nat.add_mul, Nat.one_mul, Nat.add_zero] <;>
    omega

def widen16 (bf16 : Bool) (x : Nat) : Nat :=
  if bf16 then x % 65536 * 65536 else
    let sign := x / 32768 % 2 * 2^31
    let e := x / 1024 % 32
    let f := x % 1024
    if e = 31 then sign + 0x7f800000 + f * 8192
    else if e > 0 then sign + (e + 112)*2^23 + f*8192
    else if f = 0 then sign
    else sign + (f.log2 + 103)*2^23 + f*2^(23-f.log2) % 2^23

def downSmall (fmt : Format) (relu : Bool) (x : Nat) : Nat :=
  let mag := x % 2^31
  let nan := mag > 0x7f800000
  let negative := x / 2^31 % 2 = 1
  let sign := if negative then 2^(fmt.exponentBits + fmt.fractionBits) else 0
  if nan then if fmt.exponentBits ≥ 4 then 127 else fmt.limit
  else if relu ∧ negative then 0
  else if mag = 0x7f800000 then sign + fmt.limit
  else
    let e : Nat := mag / 2^23
    let m := mag % 2^23 + if e = 0 then 0 else 2^23
    let target : Int := (max e 1 : Nat) - 127 + (fmt.bias : Int)
    let d := 23 - (fmt.fractionBits : Int) + max (1-target) 0
    let value := (max (target-1) 0).toNat * 2^fmt.fractionBits + roundShift m d
    sign + min value fmt.limit

def downE8m0 (up sat : Bool) (x : Nat) : Nat :=
  let mag := x % 2^31
  if mag > 0x7f800000 then 255 else
    let e := mag / 2^23
    let value := e + if up ∧ mag % 2^23 ≠ 0 ∧ mag > 0x400000 ∧ e < 255 then 1 else 0
    min value (if sat then 254 else 255)

def downS2F6 (nearest relu scaled : Bool) (x scale : Nat) : Nat :=
  let mag := x % 2^31
  let negative := x / 2^31 % 2 = 1
  if mag > 0x7f800000 ∨ (scaled ∧ scale = 255) then 127
  else if relu ∧ negative then 0
  else
    let e := mag / 2^23
    let m := mag % 2^23 + if e = 0 then 0 else 2^23
    let shift : Int := 144 - (max e 1 : Nat) + if scaled then (scale : Int)-127 else 0
    let rounded := if shift ≤ 0 then m * 2^(-shift).toNat
      else if nearest then nearestEven m shift.toNat else m / 2^shift.toNat
    let magnitude := if mag = 0x7f800000 then 128 else rounded
    if negative then (256 - min magnitude 128) % 256 else min magnitude 127

def encodeFinite (fractionBits bias maxExponent : Nat) (negative : Bool)
    (mantissa : Nat) (power : Int) : Nat :=
  let sign := if negative then (maxExponent+1)*2^fractionBits else 0
  if mantissa = 0 then sign else
    let e : Int := (mantissa.log2 : Int) + power + (bias : Int)
    let shift := max e 1 - (bias : Int) - (fractionBits : Int) - power
    let code := (max (e-1) 0).toNat * 2^fractionBits + roundShift mantissa shift
    sign + min code (maxExponent*2^fractionBits)

def upSmall (name : String) (fmt : Format) (bf16 relu sat scaled : Bool)
    (word scale : Nat) : Nat :=
  let e := word / 2^fmt.fractionBits % 2^fmt.exponentBits
  let f := word % 2^fmt.fractionBits
  let negative := word / 2^(fmt.fractionBits+fmt.exponentBits) % 2 = 1
  let nan := (name = "e4m3" ∧ e = 15 ∧ f = 7) ∨
    (name = "e5m2" ∧ e = 31 ∧ f ≠ 0) ∨ (scaled ∧ scale = 255)
  let infinity := name = "e5m2" ∧ e = 31 ∧ f = 0
  let fractionBits := if bf16 then 7 else 10
  let bias := if bf16 then 127 else 15
  let maxExponent := if bf16 then 255 else 31
  let infBits := maxExponent * 2^fractionBits
  let sign := if negative then 32768 else 0
  if nan then infBits + 2^fractionBits - 1
  else if relu ∧ negative then 0
  else if infinity then sign + infBits - if sat then 1 else 0
  else
    let m := f + if e = 0 then 0 else 2^fmt.fractionBits
    let power := (max e 1 : Int) - (fmt.bias : Int) - (fmt.fractionBits : Int) +
      if scaled then (scale : Int) - 127 else 0
    let bits := encodeFinite fractionBits bias maxExponent negative m power
    if sat ∧ bits % 32768 = infBits then bits - 1 else bits

def upS2F6 (relu sat scaled : Bool) (word scale : Nat) : Nat :=
  let word := word % 256
  let negative := word ≥ 128
  if scaled ∧ scale = 255 then 0x7fff
  else if relu ∧ negative then 0
  else
    let magnitude := if negative then 256-word else word
    let power : Int := -6 + if scaled then (scale : Int)-127 else 0
    let bits := encodeFinite 7 127 255 negative magnitude power
    if sat ∧ bits % 32768 = 0x7f80 then bits-1 else bits

def evaluate (direction name other mode : String) (relu sat scaled : Bool)
    (a b scale : Nat) : Option Nat := do
  if direction = "down" then
    let (hi, lo) ← if other = "f32" then some (a, b)
      else if other = "f16x2" ∨ other = "bf16x2" then
        some (widen16 (other = "bf16x2") (a / 65536), widen16 (other = "bf16x2") a)
      else none
    if name = "s2f6" then
      if (mode ≠ "rn" ∧ mode ≠ "rz") ∨ !sat then none else
        return downS2F6 (mode = "rn") relu scaled hi (scale / 256 % 256) * 256 +
          downS2F6 (mode = "rn") relu scaled lo (scale % 256)
    if name = "ue8m0" then
      if mode ≠ "rz" ∧ mode ≠ "rp" then none else
        return downE8m0 (mode = "rp") sat hi * 256 + downE8m0 (mode = "rp") sat lo
    if mode ≠ "rn" ∨ !sat ∨ scaled then none else do
      let fmt ← format name
      let stride := if name = "e2m1" then 16 else 256
      return downSmall fmt relu hi * stride + downSmall fmt relu lo
  else if direction = "up" ∧ mode = "rn" then
    if name = "s2f6" then
      if other ≠ "bf16x2" then none else
        return upS2F6 relu sat scaled (a / 256 % 256) (scale / 256 % 256) * 65536 +
          upS2F6 relu sat scaled (a % 256) (scale % 256)
    else if name = "ue8m0" then
      if other ≠ "bf16x2" ∨ relu ∨ sat ∨ scaled then none else
        let convert := fun x => if x = 255 then 0x7fff else if x = 0 then 64 else x*128
        return convert (a / 256 % 256) * 65536 + convert (a % 256)
    else do
      if other ≠ "f16x2" ∧ other ≠ "bf16x2" then none else do
        let fmt ← format name
        let stride := if name = "e2m1" then 16 else 256
        let hi := upSmall name fmt (other = "bf16x2") relu sat scaled (a / stride) (scale / 256 % 256)
        let lo := upSmall name fmt (other = "bf16x2") relu sat scaled (a % stride) (scale % 256)
        return hi*65536 + lo
  else none

theorem fp4_midpoint_even : downSmall ⟨2, 1, 1, 7⟩ false 0x3fa00000 = 2 := by decide
theorem fp4_midpoint_odd : downSmall ⟨2, 1, 1, 7⟩ false 0x3fe00000 = 4 := by decide
theorem fp8_saturates : downSmall ⟨4, 3, 7, 126⟩ false 0x7f800000 = 126 := by decide
theorem e8m0_abs : downE8m0 true true 0xbf800001 = 128 := by decide
theorem scaled_s2f6_subnormal : downS2F6 true false true 0x00400000 0 = 64 := by decide
theorem s2f6_scaled_roundtrip : upS2F6 false false true 64 127 = 0x3f80 := by decide

#print axioms nearest_even_grid
#print axioms nearest_even_between
#print axioms nearest_even_half_quantum
#print axioms fp4_midpoint_even
#print axioms fp4_midpoint_odd
#print axioms fp8_saturates
#print axioms e8m0_abs
#print axioms scaled_s2f6_subnormal
#print axioms s2f6_scaled_roundtrip

def readWord (data : ByteArray) (offset bytes : Nat) : Nat := Id.run do
  let mut value := 0
  for j in [:bytes] do value := value + data[offset+j]!.toNat * 2^(8*j)
  return value

end GB300Math.SmallConversions

open GB300Math.SmallConversions

def main (args : List String) : IO UInt32 := do
  let [manifest] := args | throw (IO.userError "Expected a tab-separated manifest")
  let rows ← IO.FS.readFile manifest
  let mut failures := 0
  let mut cases := 0
  for row in rows.splitOn "\n" do
    if row.isEmpty then continue
    let [direction, fmt, other, mode, relu, sat, scaled, input, output, expected] := row.splitOn "\t" |
      throw (IO.userError s!"Invalid case {row}")
    unless [relu, sat, scaled].all (fun s => s = "0" ∨ s = "1") do
      throw (IO.userError "Invalid Boolean")
    let some expected := expected.toNat? | throw (IO.userError "Invalid mismatch count")
    let xs ← IO.FS.readBinFile input
    let ys ← IO.FS.readBinFile output
    unless xs.size > 0 ∧ xs.size % 16 = 0 ∧ ys.size = xs.size / 2 do
      throw (IO.userError "Invalid input/output size")
    let mut errors := 0
    for i in [:xs.size/16] do
      let a := readWord xs (16*i) 4
      let b := readWord xs (16*i+4) 4
      let scale := readWord xs (16*i+8) 4
      let some model := evaluate direction fmt other mode (relu = "1") (sat = "1") (scaled = "1") a b scale |
        throw (IO.userError s!"Unsupported {row}")
      let actual := readWord ys (8*i) 8
      if model ≠ actual then
        if errors < 3 then IO.println s!"witness {fmt} {direction} {other} row={i} a={a} b={b} scale={scale} model={model} actual={actual}"
        errors := errors + 1
    cases := cases + 1
    IO.println s!"result {output} records={xs.size/16} mismatches={errors} expected={expected}"
    if errors ≠ expected then failures := failures + 1
  unless cases > 0 do throw (IO.userError "Empty manifest")
  return if failures = 0 then 0 else 1
