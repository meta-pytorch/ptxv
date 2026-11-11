/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FabricBF16

namespace FabricBF16V2
open FabricBF16Block

def coarseShift (words : List Nat) : Nat := 16*(maxExponent words/16)-17

def flush (shift : Nat) (value : Int) : Int :=
  if value.natAbs < 2^shift then 0 else value

def output (states : Array Accum) (lane shift : Nat) (sr : Bool) : Nat :=
  match states[lane]! with
  | .finite value =>
    if value.natAbs < 2^shift then 0 else FabricBF16Block.output states lane sr
  | _ => FabricBF16Block.output states lane sr

def predict (inputs : Array (Array Nat)) (sr : Bool) : Array Nat := Id.run do
  let mut result := #[]
  for group in [:2] do
    let words := (Array.range 4).map fun lane =>
      inputs.toList.map fun rank => rank[group*4+lane]!
    let states := words.map accumulate
    for lane in [:4] do result := result.push (output states lane (coarseShift words[lane]!) sr)
  return result

def cancellation : Array (Array Nat) :=
  #[#[0x3f80,0,0,0,0,0,0,0],#[0xbf80,0,0,0,0,0,0,0],
    #[0x2b80,0,0,0,0,0,0,0],#[0,0,0,0,0,0,0,0]]

theorem cancellation_old : (FabricBF16Block.predict cancellation false)[0]! = 0x2b80 := by
  decide +kernel

theorem cancellation_new : (predict cancellation false)[0]! = 0 ∧
    (predict cancellation true)[0]! = 0 := by decide +kernel

#print axioms cancellation_old
#print axioms cancellation_new
end FabricBF16V2
