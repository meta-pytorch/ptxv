/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.Code.Funs

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 3000000
set_option maxRecDepth 10000

namespace Instruction.BarrierSeed

abbrev Bit := Fin 32

def Labels (operation : Op) (bit : Bit) : Prop :=
  match operation with
  | .BarSync (.Imm identifier) _ _ _ _ _ =>
      if identifier.val < 32 then bit.val = identifier.val else True
  | .BarSync _ _ _ _ _ _ | .CallAbs _ _ => True
  | _ => False

def SeedPost (operation : Op) (word : U32) : Prop :=
  ∀ bit, word.bv[bit.val] = true ↔ Labels operation bit

@[simp]
theorem zero_bit (bit : Bit) : (0#u32).bv[bit.val] = true ↔ False := by simp

@[simp]
theorem max_bit (bit : Bit) : core.num.U32.MAX.bv[bit.val] = true ↔ True := by
  change (BitVec.allOnes 32)[bit.val] = true ↔ True
  simp only [BitVec.getElem_allOnes]

@[step]
theorem rust_shifted_one (identifier : U64) (inside : identifier.val < 32) :
    (1#u32 <<< identifier) ⦃ word => ∀ bit : Bit, word.bv[bit.val] = true ↔ bit.val = identifier.val ⦄ := by
  step with U32.ShiftLeft_spec (1#u32) identifier inside as ⟨word, value, bits⟩
  rename_i bit
  rw [bits]
  change (1#32 <<< identifier.val)[bit.val] = true ↔ bit.val = identifier.val
  simp only [BitVec.getElem_shiftLeft, BitVec.getElem_one, Bool.and_eq_true,
    Bool.not_eq_true', decide_eq_false_iff_not, decide_eq_true_eq]
  omega

@[step]
theorem rust_barrier_seed (operation : Op) :
    control_flow.barrier_seed operation ⦃ word => SeedPost operation word ⦄ := by
  cases operation <;> simp only [control_flow.barrier_seed, WP.spec_ok, SeedPost, Labels]
  all_goals try exact fun bit => zero_bit bit
  all_goals try exact fun bit => max_bit bit
  case BarSync identifier count arrive packed aligned reduction =>
    cases identifier <;> simp only [WP.spec_ok]
    all_goals try exact fun bit => max_bit bit
    case Imm identifier =>
      by_cases inside : identifier.val < 32
      · have test : identifier < 32#u64 := by scalar_tac
        simp only [test, inside, ite_true]
        exact rust_shifted_one identifier inside
      · have test : ¬identifier < 32#u64 := by scalar_tac
        simp only [test, inside, ite_false, WP.spec_ok]
        exact fun bit => max_bit bit

end Instruction.BarrierSeed
