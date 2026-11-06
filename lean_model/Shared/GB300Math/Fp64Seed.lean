import GB300Math.SeedTables

namespace GB300Math.Fp64Seed

def squareTruncated (k col : Nat) : Nat := Id.run do
  let mut dropped := 0
  for i in [:((col + 1) / 2)] do
    if (k >>> i) % 2 = 1 then
      if 2 * i < col then
        dropped := dropped + 2 ^ (2 * i)
      if col ≥ 2 * i + 2 then
        let hi := col - 2 - i
        if hi > i then
          let mask := 2 ^ (hi + 1) - 2 ^ (i + 1)
          dropped := dropped + (k &&& mask) * 2 ^ (i + 1)
  return k * k - dropped

def interpolate (rsq : Bool) (index : Nat) : Nat :=
  let n := if rsq then 17 else 16
  let col := if rsq then 19 else 17
  let table := if rsq then SeedTables.rsq else SeedTables.rcp
  let (c0, c1, c2) := table[index / 2 ^ n]!
  let k := index % 2 ^ n
  let offset := if rsq then 4294705152 else 1060634624
  let b := if rsq then 18 else 19
  let c := if rsq then 0 else 2
  ((c0 * 2 ^ 32 + offset + c1 * (k : Int) * 2 ^ b +
    c2 * (squareTruncated k col : Int) * 2 ^ c) / 2 ^ 34).toNat

def highWord (rsq : Bool) (input : Nat) : Nat :=
  let high := input % 2 ^ 32
  let sign := high &&& 0x80000000
  let exponent := (high / 2 ^ 20) % 2048
  let fraction := high % 2 ^ 20
  if exponent = 2047 ∧ fraction ≠ 0 then 0x7fffffff
  else if exponent = 0 then sign ||| 0x7ff00000
  else if rsq ∧ sign ≠ 0 then 0x7fffffff
  else if exponent = 2047 then sign
  else
    let parity : Nat := if rsq then (exponent + 1) % 2 else 0
    let index := parity * 2 ^ 23 + fraction * 8
    let mantissa := if rsq ∧ index = 0 then 2 ^ 24 else interpolate rsq index
    let outputExponent : Int :=
      (if rsq then 1023 - ((exponent : Int) - 1023 - (parity : Int)) / 2
       else 2046 - (exponent : Int)) - (if mantissa = 2 ^ 24 then 0 else 1)
    if outputExponent ≤ 0 then sign
    else sign ||| (outputExponent.toNat * 2 ^ 20) ||| ((mantissa % 2 ^ 23) / 8)

def evaluate (rsq : Bool) (word : Nat) : Nat := highWord rsq (word / 2 ^ 32) * 2 ^ 32

theorem ignores_low_word (rsq : Bool) (high low : Nat) (h : low < 2 ^ 32) :
    evaluate rsq (high * 2 ^ 32 + low) = highWord rsq high * 2 ^ 32 := by
  have hdiv : (high * 2 ^ 32 + low) / 2 ^ 32 = high := by omega
  simp only [evaluate, hdiv]

theorem zero_low_output (rsq : Bool) (word : Nat) :
    evaluate rsq word % 2 ^ 32 = 0 := by
  simp [evaluate]

#print axioms ignores_low_word
#print axioms zero_low_output

end GB300Math.Fp64Seed
