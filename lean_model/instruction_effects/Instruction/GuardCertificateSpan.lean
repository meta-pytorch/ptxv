/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.GuardCertificateWords
import Instruction.SpanRefinement

open Aeneas Aeneas.Std Result PtxvIr
set_option autoImplicit false
set_option maxHeartbeats 5000000
namespace Instruction.GuardCertificate

@[step] theorem span_loop_spec (initial : alloc.vec.Vec Reg) (first : Reg) (count : U16)
    (hindex : first.idx.val + count.val ≤ 65536)
    (hsize : initial.val.length + count.val ≤ Usize.max) :
    guard_certificate.span_loop first count initial 0#u16 ⦃ result =>
      result.val.map key = initial.val.map key ++ consecutive first count.val ∧
      result.val.length = initial.val.length + count.val ⦄ := by
  unfold guard_certificate.span_loop
  apply loop.spec_decr_nat
    (fun state : alloc.vec.Vec Reg × U16 => count.val - state.2.val)
    (fun state : alloc.vec.Vec Reg × U16 =>
      state.2.val ≤ count.val ∧
      state.1.val.map key = initial.val.map key ++ consecutive first state.2.val ∧
      state.1.val.length = initial.val.length + state.2.val)
  · rintro ⟨result, index⟩ ⟨hbound, hmap, hlength⟩
    dsimp only at hbound hmap hlength
    unfold guard_certificate.span_loop.body
    by_cases hlt : index.val < count.val
    · have htest : index < count := by scalar_tac
      simp only [htest, ite_true]
      step as ⟨position, hposition⟩; try scalar_tac
      step with alloc.vec.Vec.push_spec result { first with idx := position }
        (by scalar_tac) as ⟨pushed, hpushed⟩
      step as ⟨next, hnext⟩; try scalar_tac
      refine ⟨?_, ?_, ?_, ?_⟩
      · scalar_tac
      · rw [hpushed, List.map_append, hmap]
        have hnextval : next.val = index.val + 1 := by scalar_tac
        rw [hnextval, consecutive_succ, List.append_assoc]
        congr 2
        simp only [List.map_cons, List.map_nil, List.cons.injEq, and_true]
        have hpositionval : position.val = first.idx.val + index.val := by scalar_tac
        simp [key, hpositionval]
      · simp [hpushed, hlength]
        scalar_tac
      · scalar_tac
    · have htest : ¬index < count := by scalar_tac
      simp only [htest, ite_false, WP.spec_ok]
      have heq : index.val = count.val := by omega
      exact ⟨by simpa [heq] using hmap, by simpa [heq] using hlength⟩
  · simp [consecutive]


@[step] theorem span_spec (first : Reg) (count : U16)
    (hindex : first.idx.val + count.val ≤ 65536)
    (hsize : count.val ≤ Usize.max) :
    guard_certificate.span first count ⦃ result =>
      result.val.map key = consecutive first count.val ∧ result.val.length = count.val ⦄ := by
  unfold guard_certificate.span
  apply WP.spec_mono (span_loop_spec (alloc.vec.Vec.new Reg) first count hindex (by simpa using hsize))
  intro result property
  simpa only [empty_register_vector, List.map_nil, List.nil_append, List.length_nil, Nat.zero_add]
    using property

def moveCount (destination : Reg) (operand : Val) : Nat :=
  if (match operand with | .Const _ bytes => decide (bytes.val = 16) | _ => false) then 4
  else if destination.file = .V then 1
  else match operand with
    | .Reg _ count => max count.val 1
    | .Const _ bytes => max (bytes.val / 4) 1
    | _ => 1

@[step] theorem move_count_spec (destination : Reg) (operand : Val) :
    guard_certificate.move_count destination operand
      ⦃ result => result.val = moveCount destination operand ⦄ := by
  have literal : (16#8#uscalar : U8).val = 16 := rfl
  cases destination with
  | mk file index =>
    cases file <;> cases operand <;>
      simp [guard_certificate.move_count, moveCount, File.Insts.CoreCmpPartialEqFile.eq,
        File.read_discriminant, lift, UScalar.cast_val_eq]
    all_goals first | (split <;> simp_all [UScalar.eq_equiv, literal]) | scalar_tac
    all_goals step <;> scalar_tac

end Instruction.GuardCertificate
