/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionProgram.Code.Funs

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction.IEEEComparison

abbrev fractionBase : Nat := 8388608
abbrev signBase : Nat := 2147483648

def finiteUnits (exponent fraction : Nat) : Nat :=
  if exponent = 0 then fraction else (fractionBase + fraction) * 2 ^ (exponent - 1)

inductive Value where
  | negativeInfinity
  | finite (units : Int)
  | positiveInfinity
  | nan
  deriving DecidableEq

def decode (bits : U32) (flush : Bool) : Value :=
  let negative := bits.val ≥ signBase
  let magnitude := bits.val % signBase
  let exponent := magnitude / fractionBase
  let fraction := magnitude % fractionBase
  if exponent = 255 then
    if fraction ≠ 0 then .nan
    else if negative then .negativeInfinity else .positiveInfinity
  else
    let units := if flush && exponent == 0 then 0 else finiteUnits exponent fraction
    .finite (if negative then -(units : Int) else (units : Int))

def unordered : Value → Value → Bool
  | .nan, _ | _, .nan => true
  | _, _ => false

def less : Value → Value → Bool
  | .negativeInfinity, .finite _ | .negativeInfinity, .positiveInfinity => true
  | .finite _, .positiveInfinity => true
  | .finite first, .finite second => decide (first < second)
  | _, _ => false

def equal : Value → Value → Bool
  | .negativeInfinity, .negativeInfinity | .positiveInfinity, .positiveInfinity => true
  | .finite first, .finite second => decide (first = second)
  | _, _ => false

def truth (mode : PtxvSim.ptxv_ir.FCmp) (first second : Value) : Bool :=
  match mode with
  | .F => false
  | .T => true
  | .Lt => less first second
  | .Eq => equal first second
  | .Le => less first second || equal first second
  | .Gt => less second first
  | .Ne => !unordered first second && !equal first second
  | .Ge => less second first || equal first second
  | .Num => !unordered first second
  | .Nan => unordered first second
  | .Ltu => unordered first second || less first second
  | .Equ => unordered first second || equal first second
  | .Leu => unordered first second || less first second || equal first second
  | .Gtu => unordered first second || less second first
  | .Neu => unordered first second || !equal first second
  | .Geu => unordered first second || less second first || equal first second

def compare (mode : PtxvSim.ptxv_ir.FCmp) (first second : U32) (flush : Bool) : Bool :=
  truth mode (decode first flush) (decode second flush)

end Instruction.IEEEComparison
