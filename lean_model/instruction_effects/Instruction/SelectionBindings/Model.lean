/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionEvaluator.Composition

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction.SegBinding

open SegProgram

abbrev Binding := PtxvSim.ptxv_ir.Reg × PtxvSim.seg.SVal

def lookup (bindings : List Binding) (register : PtxvSim.ptxv_ir.Reg) : Option PtxvSim.seg.SVal :=
  match bindings with
  | [] => none
  | (candidate, source) :: rest =>
      if registerKey candidate = registerKey register then some source else lookup rest register

def svalBits (args : List U32) (prior : List U64) : PtxvSim.seg.SVal → Option U64
  | .Arg index => (args[index.val]?).map (UScalar.cast .U64)
  | .Const bits => some bits
  | .Op index => prior[index.val]?

def svalValid (args : List U32) (prior : List U64) : PtxvSim.seg.SVal → Prop
  | .Arg index => index.val < args.length
  | .Const _ => True
  | .Op index => index.val < prior.length

def predicateSource (predicate : PtxvSim.ptxv_ir.Pred) (bindings : List Binding) :
    Option PtxvSim.seg.SVal × Bool :=
  (predicate.reg.bind (lookup bindings), predicate.neg)

def registerBits (register : PtxvSim.ptxv_ir.Reg) (bindings : List Binding)
    (args : List U32) (prior : List U64) : Option U64 :=
  match lookup bindings register with
  | none => some 0#u64
  | some source => svalBits args prior source

def registerPredicateBits (register : PtxvSim.ptxv_ir.Reg) (bits : U64) : Bool :=
  match register.file with
  | .R | .UR => decide ((bits.val / 2 ^ 31) % 2 = 1)
  | _ => bits != 0#u64

def predicateBits (predicate : PtxvSim.ptxv_ir.Pred) (bindings : List Binding)
    (args : List U32) (prior : List U64) : Option Bool :=
  let (source, neg) := predicateSource predicate bindings
  match predicate.reg, source with
  | some register, some source =>
      (svalBits args prior source).map (fun bits => registerPredicateBits register bits != neg)
  | _, _ => some (!neg)

theorem sign_predicate_separates_nonzero (index : U16) :
    registerPredicateBits ⟨.R, index⟩ 1#u64 = false ∧
    registerPredicateBits ⟨.UR, index⟩ 2147483648#u64 = true := by
  constructor <;> rfl

theorem register_key_exact (left right : PtxvSim.ptxv_ir.Reg) :
    registerKey left = registerKey right ↔ left.file = right.file ∧ left.idx = right.idx := by
  rcases left with ⟨leftFile, leftIndex⟩
  rcases right with ⟨rightFile, rightIndex⟩
  have hindex : leftIndex.val = rightIndex.val ↔ leftIndex = rightIndex :=
    ⟨UScalar.eq_of_val_eq, UScalar.val_eq_of_eq⟩
  cases leftFile <;> cases rightFile <;> simp [registerKey, hindex]

theorem lookup_drop (bindings : List Binding) (register : PtxvSim.ptxv_ir.Reg)
    (index : Nat) (within : index < bindings.length) :
    lookup (bindings.drop index) register =
      if registerKey bindings[index].1 = registerKey register then some bindings[index].2
      else lookup (bindings.drop (index + 1)) register := by
  rw [List.drop_eq_getElem_cons within]
  rfl

end Instruction.SegBinding
