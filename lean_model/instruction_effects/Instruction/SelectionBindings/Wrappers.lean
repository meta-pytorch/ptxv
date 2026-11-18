/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.SelectionBindings.Lookup

open Aeneas Aeneas.Std Result
set_option autoImplicit false
namespace Instruction.SegBinding

theorem source_register_binding_exact (bindings : Slice Binding) (register : PtxvSim.ptxv_ir.Reg) :
    PtxvSim.seg.binding.register_binding bindings register = .ok (lookup bindings.val register) :=
  exact_eq_of_spec _ _ (source_register_binding bindings register)

theorem source_predicate_binding_exact (predicate : PtxvSim.ptxv_ir.Pred) (bindings : Slice Binding) :
    PtxvSim.seg.binding.predicate_binding predicate bindings = .ok (predicateSource predicate bindings.val) :=
  exact_eq_of_spec _ _ (source_predicate_binding predicate bindings)

theorem source_register_bits_exact (register : PtxvSim.ptxv_ir.Reg) (bindings : Slice Binding)
    (inputs : PtxvSim.seg.binding.Inputs) :
    PtxvSim.seg.binding.register_bits register bindings inputs =
      checked (registerBits register bindings.val inputs.args.val inputs.prior.val) := by
  unfold PtxvSim.seg.binding.register_bits
  rw [source_register_binding_exact]
  cases hlookup : lookup bindings.val register <;>
    simp [registerBits, hlookup, checked, source_read_sval_exact]

theorem source_read_predicate_exact (predicate : PtxvSim.ptxv_ir.Pred) (bindings : Slice Binding)
    (inputs : PtxvSim.seg.binding.Inputs) :
    PtxvSim.seg.binding.read_predicate predicate bindings inputs =
      checked (predicateBits predicate bindings.val inputs.args.val inputs.prior.val) := by
  unfold PtxvSim.seg.binding.read_predicate
  rw [source_predicate_binding_exact]
  simp only [bind_ok, predicateSource, source_read_sval_exact]
  cases hreg : predicate.reg with
  | none => simp [predicateBits, predicateSource, hreg, checked]
  | some register =>
    cases hsource : lookup bindings.val register with
    | none => simp [predicateBits, predicateSource, hreg, hsource, checked]
    | some source =>
      cases hbits : svalBits inputs.args.val inputs.prior.val source <;>
        simp [predicateBits, predicateSource, hreg, hsource, checked, hbits,
          source_predicate_register_exact]

end Instruction.SegBinding
