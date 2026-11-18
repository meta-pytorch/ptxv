/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.Bounds

open Aeneas Aeneas.Std PtxvIr
set_option autoImplicit false
namespace Instruction

def dataReads (operation : Op) : Option (List Key) :=
  match operation with
  | .Ld _ _ _ _ _ _ _ _ | .Ld256 _ _ _ _ _ _ => some []
  | .St _ bytes _ value _ => some (storeValueReads false bytes value)
  | .StAsync _ _ value bytes => some (storeValueReads true bytes value)
  | .St256 lo hi _ _ => some (consecutive lo 4 ++ consecutive hi 4)
  | .Atom operation _ bytes _ value _ _ _ replacement =>
      some (atomicValueReads operation bytes value ++ replacementReads operation bytes value replacement)
  | _ => none

def controlReads (operation : Op) : Option (List Key) :=
  match operation with
  | .Ld _ _ _ _ address _ _ _ | .Ld256 _ _ address _ _ _ | .St _ _ address _ _ |
      .St256 _ _ address _ | .Atom _ _ _ address _ _ _ _ _ => some (addressReads address)
  | .StAsync address barrier _ _ => some (operandReads address ++ operandReads barrier)
  | _ => none

def optionBounded (registers : Option (List Key)) : Prop :=
  match registers with
  | some registers => Bounded registers
  | none => True

theorem atomicValueReads_length (operation : AtomOp) (bytes : U8) (value : Val) :
    (atomicValueReads operation bytes value).length ≤ 255 := by
  cases operation with
  | Exch => exact vectorReads_length value
  | FAddFtz =>
    cases value with
    | Reg first count =>
      simp only [atomicValueReads]
      split
      · simp only [consecutive_length]; scalar_tac
      · exact operandReads_length _
    | Imm _ | Const _ _ | SReg _ => simp [atomicValueReads, operandReads]
  | _ => exact operandReads_length value

theorem replacementReads_length (operation : AtomOp) (bytes : U8) (value : Val)
    (replacement : Option Val) :
    (replacementReads operation bytes value replacement).length ≤ 255 := by
  cases replacement with
  | some replacement => exact operandReads_length replacement
  | none => cases operation <;> first
    | exact casReplacement_length bytes value none
    | exact Nat.zero_le 255

end Instruction
