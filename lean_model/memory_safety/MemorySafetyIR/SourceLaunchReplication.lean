/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyIR.LaunchModel
import MemorySafety.SourceVectors

open Aeneas Aeneas.Std Result ControlFlow PtxvMemorySafety PtxvMemorySafety.ptxv_memory_write
open MemorySafety.Source

set_option autoImplicit false
set_option backward.isDefEq.respectTransparency false
set_option maxHeartbeats 4000000
set_option maxRecDepth 4096
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyIR

theorem source_instruction_clone (instruction : memory_safety.Instruction) :
    memory_safety.Instruction.Insts.CoreCloneClone.clone instruction = .ok instruction := by
  cases instruction with
  | Access request => simp only [memory_safety.Instruction.Insts.CoreCloneClone.clone, memory_safety.Request.Insts.CoreCloneClone.clone, bind_ok]
  | Free index => simp only [memory_safety.Instruction.Insts.CoreCloneClone.clone, core.clone.impls.CloneUsize.clone, lift, bind_ok]
  | Choice targets =>
    simp only [memory_safety.Instruction.Insts.CoreCloneClone.clone]
    rw [vec_clone_exact core.clone.CloneUsize (by intro value; rfl)]
    simp only [bind_ok]
  | Skip => rfl
  | Halt => rfl
  | Unsupported => rfl

theorem source_replicate_copy (instructions : Slice memory_safety.Instruction)
    (code result : alloc.vec.Vec memory_safety.Instruction) (pc : Usize)
    (computed : strict_memory_launch.replicate_code_loop0_loop0 instructions code pc = .ok result) :
    result.val = code.val ++ instructions.val.drop pc.val := by
  generalize fuelEq : instructions.val.length - pc.val = fuel
  induction fuel using Nat.strong_induction_on generalizing code pc with
  | h fuel ih =>
    have unroll : strict_memory_launch.replicate_code_loop0_loop0 instructions code pc = (do
        let result ← strict_memory_launch.replicate_code_loop0_loop0.body instructions code pc
        match result with
        | .cont pair => strict_memory_launch.replicate_code_loop0_loop0 instructions pair.1 pair.2
        | .done result => .ok result) := by
      unfold strict_memory_launch.replicate_code_loop0_loop0
      rw [loop.eq_def]
      congr 1
      funext result
      cases result with
      | cont pair => cases pair; rfl
      | done output => rfl
    rw [unroll] at computed
    by_cases inside : pc.val < instructions.val.length
    · have bound : pc.val + 1 < 2 ^ UScalarTy.Usize.numBits := by
        have := Slice.length_ineq instructions
        scalar_tac
      let after : Usize := Usize.ofNatCore (pc.val + 1) bound
      have afterVal : after.val = pc.val + 1 := by simp [after]
      have increment := source_usize_increment pc bound
      have lookup : Slice.index_usize instructions pc = .ok instructions.val[pc.val] := by
        simp [Slice.index_usize, List.getElem?_eq_getElem inside]
      have test : pc < Slice.len instructions := by scalar_tac
      have body : strict_memory_launch.replicate_code_loop0_loop0.body instructions code pc = (do
          let next ← alloc.vec.Vec.push code instructions.val[pc.val]
          .ok (.cont (next, after))) := by
        simp only [strict_memory_launch.replicate_code_loop0_loop0.body, test, ite_true,
          lookup, bind_ok, source_instruction_clone]
        congr 1
        funext next
        rw [increment]
        simp only [bind_ok]
        rfl
      rw [body] at computed
      simp only [Std.bind_assoc, bind_ok] at computed
      obtain ⟨next, pushed, finished⟩ := source_bind_success _ _ _ computed
      have tail := ih (instructions.val.length - after.val) (by omega) next after finished rfl
      rw [tail, vec_push_success code next instructions.val[pc.val] pushed]
      rw [List.drop_eq_getElem_cons inside]
      simp only [afterVal, List.append_assoc, List.singleton_append]
    · have test : ¬pc < Slice.len instructions := by scalar_tac
      simp only [strict_memory_launch.replicate_code_loop0_loop0.body, test, ite_false,
        bind_ok, ok.injEq] at computed
      rw [← computed]
      simp [List.drop_eq_nil_of_le (show instructions.val.length ≤ pc.val by omega)]

theorem source_replicate_loop (instructions : Slice memory_safety.Instruction) (count : Usize)
    (threads result : alloc.vec.Vec (alloc.vec.Vec memory_safety.Instruction)) (thread : Usize)
    (computed : strict_memory_launch.replicate_code_loop0 instructions count threads thread = .ok result) :
    result.val.map (fun code => code.val) = threads.val.map (fun code => code.val) ++
      List.replicate (count.val - thread.val) instructions.val := by
  generalize fuelEq : count.val - thread.val = fuel
  induction fuel using Nat.strong_induction_on generalizing threads thread with
  | h fuel ih =>
    have unroll : strict_memory_launch.replicate_code_loop0 instructions count threads thread = (do
        let result ← strict_memory_launch.replicate_code_loop0.body instructions count threads thread
        match result with
        | .cont pair => strict_memory_launch.replicate_code_loop0 instructions count pair.1 pair.2
        | .done result => .ok result) := by
      unfold strict_memory_launch.replicate_code_loop0
      rw [loop.eq_def]
      congr 1
      funext result
      cases result with
      | cont pair => cases pair; rfl
      | done output => rfl
    rw [unroll] at computed
    by_cases inside : thread.val < count.val
    · have bound : thread.val + 1 < 2 ^ UScalarTy.Usize.numBits := by scalar_tac
      let after : Usize := Usize.ofNatCore (thread.val + 1) bound
      have afterVal : after.val = thread.val + 1 := by simp [after]
      have increment := source_usize_increment thread bound
      have test : thread < count := by scalar_tac
      simp only [strict_memory_launch.replicate_code_loop0.body, test, ite_true] at computed
      simp only [Std.bind_assoc, bind_ok] at computed
      obtain ⟨code, copied, finished⟩ := source_bind_success _ _ _ computed
      have codeEq : code.val = instructions.val := by
        simpa using source_replicate_copy instructions (alloc.vec.Vec.new _) code 0#usize copied
      obtain ⟨next, pushed, finished⟩ := source_bind_success _ _ _ finished
      rw [increment] at finished
      simp only [bind_ok] at finished
      have tail := ih (count.val - after.val) (by omega) next after finished rfl
      rw [tail, vec_push_success threads next code pushed, List.map_append, List.map_cons,
        List.map_nil, codeEq]
      have countEq : count.val - thread.val = (count.val - after.val) + 1 := by omega
      rw [← fuelEq, countEq, List.replicate_succ]
      simp [List.append_assoc]
    · have test : ¬thread < count := by scalar_tac
      simp only [strict_memory_launch.replicate_code_loop0.body, test, ite_false,
        bind_ok, ok.injEq] at computed
      rw [← computed]
      have fuelZero : fuel = 0 := by omega
      simp [fuelZero]

theorem source_replicate_code (instructions : Slice memory_safety.Instruction) (count : Usize)
    (result : alloc.vec.Vec (alloc.vec.Vec memory_safety.Instruction))
    (computed : strict_memory_launch.replicate_code instructions count = .ok result) :
    result.val.map (fun code => code.val.map instructionModel) =
      List.replicate count.val (instructions.val.map instructionModel) := by
  have exact : result.val.map (fun code => code.val) = List.replicate count.val instructions.val := by
    simpa using source_replicate_loop instructions count (alloc.vec.Vec.new _) result 0#usize computed
  have mapped := congrArg (List.map (List.map instructionModel)) exact
  simpa only [List.map_map, List.map_replicate, Function.comp_def] using mapped

end MemorySafetyIR
