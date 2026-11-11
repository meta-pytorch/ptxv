/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

namespace GB300Math.WmmaArithmetic

def signed (n bits : Nat) : Int :=
  if n % 2^bits < 2^(bits-1) then n % 2^bits else (n % 2^bits : Nat)- (2^bits : Int)

def operand (typ : String) (word : Nat) : Int :=
  if typ = "s8" then signed word 8 else if typ = "u8" then word % 256
  else if typ = "s4" then signed word 4 else if typ = "u4" then word % 16
  else if typ = "b1" then word % 2 else signed word 32

def entry (m n k : Nat) (typ mode : String) (input : Array Nat) (r c : Nat) : Int := Id.run do
  let mut acc := signed input[m*k+k*n+r*n+c]! 32
  for j in [:k] do
    let a := operand typ input[r*k+j]!
    let b := operand typ input[m*k+j*n+c]!
    let product := if typ = "b1" then
      if mode = "xor" then (if a = b then 0 else 1) else a*b
      else a*b
    acc := acc+product
  if mode = "satfinite" then return max (-(2^31)) (min (2^31-1) acc)
  return acc

def encodeInteger (value : Int) (fraction exponent : Nat) : Nat :=
  if value = 0 then 0 else
  let magnitude := value.natAbs
  let top := magnitude.log2
  let mantissa := if top ≤ fraction then magnitude*2^(fraction-top)
    else magnitude/2^(top-fraction)
  (if value < 0 then 2^(fraction+exponent) else 0) +
    (top+2^(exponent-1)-1)*2^fraction + mantissa%2^fraction

def valueWord (dtype : String) (value : Int) : Nat :=
  if dtype = "f16" then encodeInteger value 10 5
  else if dtype = "f32" then encodeInteger value 23 8
  else if dtype = "f64" then encodeInteger value 52 11
  else (value % 2^32).toNat

def negativeContribution (m n k : Nat) (typ : String) (input : Array Nat) (r c : Nat) : Bool :=
  signed input[m*k+k*n+r*n+c]! 32 < 0 || (Array.range k).any fun j =>
    (operand typ input[r*k+j]! < 0) != (operand typ input[m*k+j*n+c]! < 0)

def cellWord (m n k : Nat) (typ dtype mode : String) (input : Array Nat) (r c : Nat) : Nat :=
  let value := entry m n k typ mode input r c
  if dtype = "f64" ∧ mode = "rm" ∧ value = 0 ∧ negativeContribution m n k typ input r c then
    0x8000000000000000
  else valueWord dtype value

def word (m n k : Nat) (typ dtype mode : String) (input : Array Nat) (i : Nat) : Nat :=
  let bits := if dtype = "f16" then 16 else if dtype = "f64" then 64 else 32
  if i*32 ≥ m*n*bits then 0xdeadbeef else
  if bits = 16 then
    cellWord m n k typ dtype mode input (i*2/n) (i*2%n) +
    cellWord m n k typ dtype mode input ((i*2+1)/n) ((i*2+1)%n)*65536
  else if bits = 64 then
    cellWord m n k typ dtype mode input (i/2/n) (i/2%n) / 2^(32*(i%2)) % 2^32
  else cellWord m n k typ dtype mode input (i/n) (i%n)

theorem negative_half : encodeInteger (-3) 10 5 = 0xc200 := by decide +kernel
theorem positive_single : encodeInteger 3 23 8 = 0x40400000 := by decide +kernel
theorem negative_double : encodeInteger (-3) 52 11 = 0xc008000000000000 := by decide +kernel
theorem signed_subbyte : operand "s4" 15 = -1 := by decide +kernel
theorem downward_cancellation_zero :
    cellWord 1 1 1 "f64" "f64" "rm" #[1,0xffffffff,1] 0 0 = 0x8000000000000000 := by decide +kernel
theorem downward_positive_zero :
    cellWord 1 1 1 "f64" "f64" "rm" #[0,0,0] 0 0 = 0 := by decide +kernel
#print axioms downward_cancellation_zero
#print axioms downward_positive_zero

#print axioms negative_half
#print axioms positive_single
#print axioms negative_double
#print axioms signed_subbyte
end GB300Math.WmmaArithmetic
