/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.AtomicCancelRefinement
open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 4000000
namespace AtomicCancel

def valueRead (value : Val) (register : Reg) : Prop :=
  match value with
  | .Reg source _ => source = register
  | _ => False

def optionRead (value : Option Val) (register : Reg) : Prop :=
  match value with
  | some value => valueRead value register
  | none => False

def addressRead (address : Addr) (register : Reg) : Prop :=
  optionRead address.base register ∨ optionRead address.base2 register

/-- Base-register read semantics; vector expansion requires the accepted scalar/private scope. -/
def baseRead (operation : Op) (register : Reg) : Prop :=
  match operation with
  | .Mov _ a | .CvtInt _ a _ _ _ => valueRead a register
  | .Int _ _ _ a b | .MulWide _ _ a b _ => valueRead a register ∨ valueRead b register
  | .Ld _ _ _ _ addr _ _ _ => addressRead addr register
  | .St _ _ addr value _ | .Atom _ _ _ addr value _ _ _ _ =>
    addressRead addr register ∨ valueRead value register
  | .Exit => False
  | _ => True

theorem source_reads_eq (value : Val) (register : Reg) :
    atomic_cancel.reads value register = .ok (by classical exact decide (valueRead value register)) := by
  classical
  cases value <;> simp [atomic_cancel.reads, valueRead, same_register_eq]
  all_goals first
    | exact decide_eq_decide.mpr Iff.rfl
    | exact decide_eq_false (fun h => h)
    | exact decide_eq_true True.intro

theorem source_optional_reads_eq (value : Option Val) (register : Reg) :
    atomic_cancel.optional_reads value register =
      .ok (by classical exact decide (optionRead value register)) := by
  classical
  cases value <;> simp [atomic_cancel.optional_reads, optionRead, source_reads_eq]
  all_goals first
    | exact decide_eq_decide.mpr Iff.rfl
    | exact decide_eq_false (fun h => h)
    | exact decide_eq_true True.intro

theorem source_address_reads_eq (address : Addr) (register : Reg) :
    atomic_cancel.address_reads address register =
      .ok (by classical exact decide (addressRead address register)) := by
  classical
  simp only [atomic_cancel.address_reads, source_optional_reads_eq, bind_ok]
  by_cases first : optionRead address.base register <;>
    simp [first, addressRead]

theorem source_base_reads_eq (operation : Op) (register : Reg) :
    atomic_cancel.reads_return operation register =
      .ok (by classical exact decide (baseRead operation register)) := by
  classical
  cases operation <;>
    simp [atomic_cancel.reads_return, baseRead, source_reads_eq, source_address_reads_eq,
      Bool.or_eq_true, Bool.and_eq_true]
  all_goals first
    | exact decide_eq_decide.mpr Iff.rfl
    | exact decide_eq_false (fun h => h)
    | exact decide_eq_true True.intro
    | (split <;> simp_all [Bool.decide_or])

theorem source_refusal_means_no_base_read (operation : Op) (register : Reg)
    (refuses : atomic_cancel.reads_return operation register = .ok false) :
    ¬baseRead operation register := by
  classical
  simpa [source_base_reads_eq] using refuses

end AtomicCancel
