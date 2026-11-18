/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MachineActionBasic
import MemoryWrite.MachineRegistersUniform
import MemoryWrite.MachineRegistersPoison
import MemoryWrite.MachineWarp

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 4000000

namespace MemoryWrite.MachineActionMutations

open MachineRegisters MachineActionModel

structure Registers where
  words : List U64
  written : List U64
  predicates : U8
  defined : U8

def registersView (registers : machine_registers.WarpRegisters) : Registers :=
  ⟨registers.ur.val, registers.ur_written.val, registers.up, registers.up_def⟩

def uniformValid (registers : machine_registers.WarpRegisters)
    (request : machine_action.UniformRequest) : Prop :=
  match request.op with
  | .Word write => write.slots.val ≤ 2 ∧ 255 ≤ registers.ur.val.length
  | .Predicate write => write.index.val < 8
  | .PoisonWord _ => True
  | .PoisonPredicate index => index.val < 8

def uniformMutation (registers : Registers) (request : machine_action.UniformRequest) :
    Option machine_recorder.Access × Registers :=
  match request.op with
  | .Word write =>
    let (words, written) := scalarResult registers.words registers.written write
    (if 255 ≤ write.index.val then none else some (uniformAccess request.warp),
      { registers with words, written })
  | .Predicate write =>
    (some (uniformAccess request.warp),
      { registers with
        predicates := predicateBit registers.predicates write.index write.value,
        defined := predicateBit registers.defined write.index write.defined })
  | .PoisonWord index =>
    if 255 ≤ index.val then (none, registers)
    else (some (uniformAccess request.warp),
      { registers with written := poisonWords registers.written index.val })
  | .PoisonPredicate index =>
    (some (uniformAccess request.warp),
      { registers with defined := predicateBit registers.defined index false })

def dispatchUniform (registers : machine_registers.WarpRegisters)
    (request : machine_action.UniformRequest) :
    Result (Option machine_recorder.Access × machine_registers.WarpRegisters) :=
  match request.op with
  | .Word write => machine_registers.write_uniform registers request.warp write
  | .Predicate write => machine_registers.write_uniform_predicate registers request.warp write
  | .PoisonWord index => machine_registers.poison_uniform registers request.warp index
  | .PoisonPredicate index => machine_registers.poison_uniform_predicate registers request.warp index

theorem source_dispatch_uniform (registers : machine_registers.WarpRegisters)
    (request : machine_action.UniformRequest) (valid : uniformValid registers request) :
    dispatchUniform registers request ⦃ emitted updated =>
      (emitted, registersView updated) = uniformMutation (registersView registers) request ⦄ := by
  cases op : request.op with
  | Word write =>
    have bounds : write.slots.val ≤ 2 ∧ 255 ≤ registers.ur.val.length := by
      simpa [uniformValid, op] using valid
    simp only [dispatchUniform, op]
    step with source_write_uniform registers request.warp write bounds.1 bounds.2 as
      ⟨emitted, updated, words, written, predicates, defined, emission⟩
    simp [uniformMutation, op, registersView, words, written, predicates, defined, emission]
  | Predicate write =>
    have bound : write.index.val < 8 := by simpa [uniformValid, op] using valid
    simp only [dispatchUniform, op]
    step with source_write_uniform_predicate registers request.warp write bound as
      ⟨emitted, updated, emission, update⟩
    simp [uniformMutation, op, registersView, emission, update]
  | PoisonWord index =>
    simp only [dispatchUniform, op]
    by_cases constant : 255 ≤ index.val
    · rw [source_uniform_poison_constant_frame registers request.warp index constant]
      simp [uniformMutation, op, constant]
    · step with source_poison_uniform registers request.warp index (by omega) as
        ⟨emitted, updated, words, predicates, defined, written, emission⟩
      simp [uniformMutation, op, constant, registersView, words, written, predicates,
        defined, emission, poisonWords]
  | PoisonPredicate index =>
    have bound : index.val < 8 := by simpa [uniformValid, op] using valid
    simp only [dispatchUniform, op]
    step with source_poison_uniform_predicate registers request.warp index bound as
      ⟨emitted, updated, emission, update⟩
    simp [uniformMutation, op, registersView, emission, update]

def routeEmission (state : State) (slice : Bool) (event : Option machine_recorder.Access) :
    Result (Option machine_recorder.Access × State) :=
  match event with
  | none => .ok (none, state)
  | some event => recordAccess state slice event

def uniformWrite (registers : Registers) (state : State) (slice : Bool)
    (request : machine_action.UniformRequest) :
    Result (Option machine_recorder.Access × Registers × State) := do
  let (emitted, updated) := uniformMutation registers request
  let (delivery, next) ← routeEmission state slice emitted
  .ok (delivery, updated, next)

def mapUniform (run : Result (Option machine_recorder.Access ×
    machine_registers.WarpRegisters × machine_action.ActionState)) :
    Result (Option machine_recorder.Access × Registers × State) := do
  let (delivery, registers, state) ← run
  .ok (delivery, registersView registers, view state)

theorem project_bind {α : Type}
    (run : Result (Option machine_recorder.Access × machine_action.ActionState)) (data : α) :
    Aeneas.Std.bind run (fun pair => .ok (pair.1, data, view pair.2)) =
      Aeneas.Std.bind (MachineAction.mapPair run) (fun pair => .ok (pair.1, data, pair.2)) := by
  unfold MachineAction.mapPair
  change Aeneas.Std.bind run (fun pair => .ok (pair.1, data, view pair.2)) =
    Aeneas.Std.bind (Aeneas.Std.bind run (fun pair => .ok (pair.1, view pair.2)))
      (fun pair => .ok (pair.1, data, pair.2))
  rw [Std.bind_assoc]
  congr 1
  funext pair
  rcases pair with ⟨delivery, next⟩
  simp only [bind_ok]

theorem source_write_uniform_unroll (registers : machine_registers.WarpRegisters)
    (state : machine_action.ActionState) (slice : Bool)
    (request : machine_action.UniformRequest) :
    machine_action.write_uniform registers state slice request = (do
      let (emitted, updated) ← dispatchUniform registers request
      match emitted with
      | none => .ok (none, updated, state)
      | some access =>
        let (delivery, next) ← machine_action.record_access state slice access
        .ok (delivery, updated, next)) := by
  cases op : request.op <;>
    simp [machine_action.write_uniform, dispatchUniform, op, Std.bind_assoc]
  all_goals
    congr 1
    funext pair
    rcases pair with ⟨emitted, updated⟩
    change Aeneas.Std.bind (.ok (updated, emitted)) _ = _
    rw [bind_ok]
    rfl

theorem source_write_uniform (registers : machine_registers.WarpRegisters)
    (state : machine_action.ActionState) (slice : Bool)
    (request : machine_action.UniformRequest) (valid : uniformValid registers request)
    (room : MachineRecorder.TraceRoom state.recorder) :
    mapUniform (machine_action.write_uniform registers state slice request) =
      uniformWrite (registersView registers) (view state) slice request := by
  obtain ⟨⟨emitted, updated⟩, executed, same⟩ := MachineAction.spec_result
    (source_dispatch_uniform registers request valid)
  simp only [WP.uncurry'_pair] at same
  rw [source_write_uniform_unroll, executed]
  unfold uniformWrite
  rw [← same]
  cases emitted with
  | none => simp [mapUniform, routeEmission]
  | some event =>
    unfold mapUniform
    conv_lhs =>
      arg 1
      rw [bind_ok]
      change (do
        let (delivery, next) ← machine_action.record_access state slice event
        .ok (delivery, updated, next))
    change Aeneas.Std.bind
      (Aeneas.Std.bind (machine_action.record_access state slice event)
        (fun pair => .ok (pair.1, updated, pair.2)))
      (fun triple => .ok (triple.1, registersView triple.2.1, view triple.2.2)) =
      Aeneas.Std.bind (recordAccess (view state) slice event)
        (fun pair => .ok (pair.1, registersView updated, pair.2))
    rw [Std.bind_assoc]
    trans (do
      let (delivery, next) ← machine_action.record_access state slice event
      .ok (delivery, registersView updated, view next))
    · congr 1
      funext pair
      rcases pair with ⟨delivery, next⟩
      rw [bind_ok]
      rfl
    · change Aeneas.Std.bind (machine_action.record_access state slice event)
        (fun pair => .ok (pair.1, registersView updated, view pair.2)) = _
      rw [project_bind, MachineAction.source_record_access state slice event room]

structure Warp where
  registers : Registers
  masks : List U32
  sequence : U32

def warpView (warp : machine_warp.WarpCore) : Warp :=
  ⟨registersView warp.registers, warp.bmask.val, warp.cp_seq⟩

def maskMutation (warp : Warp) (write : machine_warp.MaskWrite) : Warp :=
  { warp with masks := (warp.masks.set write.barrier.val
      (MachineWarp.maskValue warp.masks[write.barrier.val]! write.op)) }

def maskWrite (warp : Warp) (state : State) (slice : Bool) (write : machine_warp.MaskWrite) :
    Result (Option machine_recorder.Access × Warp × State) := do
  let (delivery, next) ← recordAccess state slice
    (MachineWarp.maskAccess write.warp write.barrier)
  .ok (delivery, maskMutation warp write, next)

def mapMask (run : Result (Option machine_recorder.Access ×
    machine_warp.WarpCore × machine_action.ActionState)) :
    Result (Option machine_recorder.Access × Warp × State) := do
  let (delivery, warp, state) ← run
  .ok (delivery, warpView warp, view state)

theorem source_write_mask (warp : machine_warp.WarpCore)
    (state : machine_action.ActionState) (slice : Bool) (write : machine_warp.MaskWrite)
    (bounded : write.barrier.val < 16) (room : MachineRecorder.TraceRoom state.recorder) :
    mapMask (machine_action.write_mask warp state slice write) =
      maskWrite (warpView warp) (view state) slice write := by
  obtain ⟨⟨emitted, updated⟩, executed, emission, registers, sequence, masks⟩ :=
    MachineAction.spec_result (MachineWarp.rust_write_mask warp write bounded)
  subst emitted
  have updatedView : warpView updated = maskMutation (warpView warp) write := by
    simp [warpView, maskMutation, registers, sequence, masks]
  simp only [machine_action.write_mask, executed, bind_ok]
  unfold mapMask
  conv_lhs =>
    arg 1
    change (do
      let (delivery, next) ← machine_action.record_access state slice
        (MachineWarp.maskAccess write.warp write.barrier)
      .ok (delivery, updated, next))
  unfold maskWrite
  rw [← updatedView]
  change Aeneas.Std.bind
    (Aeneas.Std.bind (machine_action.record_access state slice
        (MachineWarp.maskAccess write.warp write.barrier))
      (fun pair => .ok (pair.1, updated, pair.2)))
    (fun triple => .ok (triple.1, warpView triple.2.1, view triple.2.2)) =
    Aeneas.Std.bind (recordAccess (view state) slice
      (MachineWarp.maskAccess write.warp write.barrier))
      (fun pair => .ok (pair.1, warpView updated, pair.2))
  rw [Std.bind_assoc]
  trans (do
    let (delivery, next) ← machine_action.record_access state slice
      (MachineWarp.maskAccess write.warp write.barrier)
    .ok (delivery, warpView updated, view next))
  · congr 1
    funext pair
    rcases pair with ⟨delivery, next⟩
    rw [bind_ok]
    rfl
  · change Aeneas.Std.bind (machine_action.record_access state slice
        (MachineWarp.maskAccess write.warp write.barrier))
      (fun pair => .ok (pair.1, warpView updated, view pair.2)) = _
    rw [project_bind, MachineAction.source_record_access state slice
      (MachineWarp.maskAccess write.warp write.barrier) room]

end MemoryWrite.MachineActionMutations
