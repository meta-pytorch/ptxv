/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.ControlFlowRange

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace Instruction.ControlFlow

theorem next_length (context : Context) : (next context).length ≤ 1 := by
  unfold next
  split <;> simp

@[step]
theorem rust_next (context : control_flow.Context) :
    control_flow.append_next (alloc.vec.Vec.new Usize) context
      ⦃ result => RowPost (next (contextView context)) result ⦄ := by
  have bound := next_length (contextView context)
  have source := rust_append_next (alloc.vec.Vec.new Usize) context (by
    simp only [alloc.vec.Vec.new, alloc.vec.Vec.from_val, List.length_nil, Nat.zero_add]
    have large : 1 ≤ Usize.max := by scalar_tac
    omega)
  simpa only [alloc.vec.Vec.new, alloc.vec.Vec.from_val, List.map_nil, List.nil_append] using source

@[step]
theorem rust_target (target : Usize) :
    alloc.vec.Vec.push (alloc.vec.Vec.new Usize) target ⦃ result => RowPost [target.val] result ⦄ := by
  apply WP.spec_mono (alloc.vec.Vec.push_spec (alloc.vec.Vec.new Usize) target (by
    simp only [alloc.vec.Vec.new, alloc.vec.Vec.from_val, List.length_nil]; scalar_tac))
  intro result resultEq
  simp only [RowPost, resultEq, alloc.vec.Vec.new, alloc.vec.Vec.from_val,
    List.nil_append, List.map_cons, List.map_nil]

@[step]
theorem rust_target_next (target : Usize) (context : control_flow.Context) :
    Std.bind (alloc.vec.Vec.push (alloc.vec.Vec.new Usize) target)
      (fun result => control_flow.append_next result context)
      ⦃ result => RowPost (target.val :: next (contextView context)) result ⦄ := by
  step with rust_target target as ⟨initial, initialPost⟩
  have initialLength := initialPost.length
  have bound := next_length (contextView context)
  step with rust_append_next initial context (by
    simp only [List.length_singleton] at initialLength
    have large : 2 ≤ Usize.max := by scalar_tac
    omega) as ⟨result, resultPost⟩
  change initial.val.map UScalar.val = [target.val] at initialPost
  simpa only [initialPost, List.singleton_append] using resultPost


@[step]
theorem rust_optional_next (initial : alloc.vec.Vec Usize) (context : control_flow.Context) (always : Bool)
    (space : initial.val.length + (if always then [] else next (contextView context)).length ≤ Usize.max) :
    (if always then .ok initial else control_flow.append_next initial context) ⦃ result =>
      RowPost (initial.val.map UScalar.val ++ if always then [] else next (contextView context)) result ⦄ := by
  cases always with
  | true => simp only [ite_true, WP.spec_ok, RowPost, List.append_nil]
  | false => simpa only [Bool.false_eq_true, ite_false] using rust_append_next initial context space


@[step]
theorem rust_copy_optional_next (values : Slice Usize) (context : control_flow.Context) (always : Bool)
    (space : values.val.length + (if always then [] else next (contextView context)).length ≤ Usize.max) :
    Std.bind (control_flow.append (alloc.vec.Vec.new Usize) values)
      (fun result => if always then .ok result else control_flow.append_next result context)
      ⦃ result => RowPost (values.val.map UScalar.val ++ if always then [] else next (contextView context)) result ⦄ := by
  step with rust_append (alloc.vec.Vec.new Usize) values (by
    simp only [alloc.vec.Vec.new, alloc.vec.Vec.from_val, List.length_nil, Nat.zero_add]
    exact values.property) as ⟨initial, copied⟩
  have initialValue : initial.val = values.val := by
    simpa only [alloc.vec.Vec.new, alloc.vec.Vec.from_val, List.nil_append] using copied
  step with rust_optional_next initial context always (by simpa only [initialValue] using space)
    as ⟨result, post⟩
  simpa only [initialValue] using post

@[step]
theorem rust_range_optional_next (context : control_flow.Context) (always : Bool)
    (space : context.count.val + (if always then [] else next (contextView context)).length ≤ Usize.max) :
    Std.bind (control_flow.successors_loop context.count (alloc.vec.Vec.new Usize) 0#usize)
      (fun result => if always then .ok result else control_flow.append_next result context)
      ⦃ result => RowPost (List.range context.count.val ++ if always then [] else next (contextView context)) result ⦄ := by
  step with rust_range (alloc.vec.Vec.new Usize) context.count (by
    simp only [alloc.vec.Vec.new, alloc.vec.Vec.from_val, List.length_nil, Nat.zero_add]
    scalar_tac) as ⟨initial, copied⟩
  have initialValue : initial.val.map UScalar.val = List.range context.count.val := by
    simpa only [RowPost, alloc.vec.Vec.new, alloc.vec.Vec.from_val, List.map_nil, List.nil_append] using copied
  have initialLength : initial.val.length = context.count.val := by
    simpa only [List.length_map, List.length_range] using congrArg List.length initialValue
  step with rust_optional_next initial context always (by simpa only [initialLength] using space)
    as ⟨result, post⟩
  simpa only [initialValue] using post

end Instruction.ControlFlow
