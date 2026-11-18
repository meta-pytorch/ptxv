/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.SymbolicState
import MemoryWrite.CheckedControlSafety
import MemoryWrite.Code.Funs

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 4000000

namespace MemoryWrite.Symbolic

abbrev SourceMetadata := symbolic.Metadata

def metadataView (metadata : SourceMetadata) : Metadata :=
  ⟨metadata.captured, metadata.history, metadata.tainted, metadata.defined⟩

def operationView : symbolic.BinaryOp → BinaryOp
  | .Add => .add
  | .And => .bitAnd
  | .Or => .bitOr
  | .Xor => .bitXor

def missingView : symbolic.MissingHistory → MissingHistory
  | .Constant => .constant
  | .Opaque => .opaque

@[step]
theorem rust_prepare_spec (bits : U32) (history : Option U32) (tainted defined : Bool) :
    symbolic.prepare bits history tainted defined
      ⦃ result => metadataView result = ⟨bits, history, tainted, defined⟩ ⦄ := by
  simp [symbolic.prepare, metadataView]

@[step]
theorem rust_missing_history_spec (inputDerived : Bool) :
    symbolic.classify_missing inputDerived
      ⦃ result => missingView result = classifyMissing inputDerived ⦄ := by
  cases inputDerived <;> simp [symbolic.classify_missing, missingView, classifyMissing]

@[step]
theorem rust_assign_metadata_spec (enabled dependent : Bool) (source old : SourceMetadata) :
    symbolic.assign_metadata enabled dependent source old
      ⦃ result => metadataView result = assignMetadata enabled dependent (metadataView source) (metadataView old) ⦄ := by
  cases enabled <;> cases dependent <;>
    simp [symbolic.assign_metadata, assignMetadata, metadataView]

@[step]
theorem rust_binary_bits_spec (op : symbolic.BinaryOp) (left right : U32) :
    symbolic.binary_bits op left right
      ⦃ result => result = binaryBits (operationView op) left right ⦄ := by
  cases op <;> simp [symbolic.binary_bits, binaryBits, operationView,
    core.num.U32.wrapping_add, UScalar.wrapping_add] <;>
    apply UScalar.eq_of_val_eq <;>
    change _ = (UScalar.bv _).toNat <;>
    simp only [BitVec.toNat_and, BitVec.toNat_or, BitVec.toNat_xor,
      UScalar.val_and, UScalar.val_or, UScalar.val_xor] <;> rfl

@[step]
theorem rust_binary_spec (op : symbolic.BinaryOp) (left right : SourceMetadata) :
    symbolic.binary op left right
      ⦃ result => metadataView result = binaryMetadata (operationView op) (metadataView left) (metadataView right) ⦄ := by
  unfold symbolic.binary
  step as ⟨bits, hbits⟩
  cases hl : left.tainted <;> cases hd : left.defined <;>
    simp [hl, hd, hbits, metadataView, binaryMetadata]

@[step]
theorem rust_checked_node_id_spec (index : Usize) :
    symbolic.checked_node_id index
      ⦃ result => match result with
        | .Ok id => id.val = index.val ∧ index.val ≤ U32.max
        | .Err .NodeIdExhausted => index.val > U32.max ⦄ := by
  have hmax : (UScalar.cast .Usize core.num.U32.MAX).val = U32.max := by
    rcases System.Platform.numBits_eq with h | h <;>
      simp [core.num.U32.MAX, UScalar.cast_val_eq, UScalarTy.numBits, U32.max, U32.numBits, U32.rMax, h]
  unfold symbolic.checked_node_id
  simp only [lift, bind_ok]
  split
  · simp only [WP.spec_ok]
    change index.val > (UScalar.cast .Usize core.num.U32.MAX).val at *
    simpa [hmax] using ‹index.val > (UScalar.cast .Usize core.num.U32.MAX).val›
  · simp only [WP.spec_ok]
    have hb : index.val ≤ U32.max := by
      change ¬index.val > (UScalar.cast .Usize core.num.U32.MAX).val at *
      omega
    constructor
    · rw [UScalar.cast_val_eq]
      apply Nat.mod_eq_of_lt
      simpa [U32.max, U32.numBits, UScalarTy.numBits] using Nat.lt_succ_of_le hb
    · exact hb

theorem rust_node_ids_injective (left right : Usize) (id : U32)
    (hl : symbolic.checked_node_id left = .ok (.Ok id))
    (hr : symbolic.checked_node_id right = .ok (.Ok id)) : left = right := by
  have hlp := rust_checked_node_id_spec left
  have hrp := rust_checked_node_id_spec right
  simp only [hl, hr, WP.spec_ok] at hlp hrp
  apply UScalar.eq_of_val_eq
  exact hlp.1.symm.trans hrp.1

theorem rust_fresh_allocation_preserves_old_meaning {Input : Type} (meaning : Meaning Input)
    (count : Usize) (fresh old : U32) (values : Input → U32)
    (hfresh : symbolic.checked_node_id count = .ok (.Ok fresh)) (hold : old.val < count.val) :
    extendMeaning meaning fresh values old = meaning old := by
  have hcount := rust_checked_node_id_spec count
  simp only [hfresh, WP.spec_ok] at hcount
  apply extendMeaning_preserves_old
  intro h
  have heq := congrArg UScalar.val h
  omega

theorem rust_register_key_preserves_namespace (thread : U32) (file : U8) (index : U16) :
    symbolic.register_key thread file index = .ok ⟨thread, file, index⟩ := by
  rfl

theorem rust_predicate_namespaces_distinct (thread : U32) (index : U16) :
    symbolic.register_key thread 3#u8 index ≠ symbolic.register_key thread 4#u8 index := by
  simp [symbolic.register_key]

@[step]
theorem rust_control_spec (guard low high : SourceMetadata) :
    symbolic.control guard low high
      ⦃ result => result.enabled = decide (guard.captured ≠ 0#u32) ∧
        result.address_dependent = (low.tainted || high.tainted) ∧
        result.guard_dependent = guard.tainted ⦄ := by
  cases hl : low.tainted <;> simp [symbolic.control, hl] <;>
    simp [bne, beq_eq_decide]
  all_goals exact ⟨fun h => by rw [h]; rfl, fun h => UScalar.eq_of_val_eq h⟩

theorem rust_controlled_write_derives_stable_execution {Input : Type} (sem : Semantics Input)
    (meaning : Meaning Input) (guard low high : SourceMetadata)
    (guardValues lowValues highValues : Input → Value)
    (hg : Sound sem meaning (metadataView guard) guardValues)
    (hl : Sound sem meaning (metadataView low) lowValues)
    (hh : Sound sem meaning (metadataView high) highValues)
    (control : symbolic.Control) (hc : symbolic.control guard low high = .ok control)
    (cells next : Slice Cell) (write : checked.PreparedWrite)
    (hrun : checked.checked_write cells
      { write with
        enabled := control.enabled
        address_dependent := control.address_dependent
        guard_dependent := control.guard_dependent } = .ok (.Ok (), next))
    (space : U8) (allocation : U64) :
    Producers.StableStoreExecution sem (predicateGuard guardValues)
      (addressFromWords space allocation lowValues highValues) ∧
    control.enabled = predicateGuard guardValues sem.capture := by
  have hcontrol := rust_control_spec guard low high
  simp only [hc, WP.spec_ok] at hcontrol
  have hindependent := rust_accepted_write_requires_independent_control cells next _ hrun
  have ha : low.tainted = false ∧ high.tainted = false := by
    simpa [hcontrol.2.1] using hindependent.1
  have hgfalse : guard.tainted = false := by
    simpa [hcontrol.2.2] using hindependent.2
  constructor
  · exact derives_stable_store_execution sem meaning (metadataView guard) (metadataView low)
      (metadataView high) guardValues lowValues highValues hg hl hh hgfalse ha.1 ha.2 space allocation
  · have hcaptured : guard.captured = (guardValues sem.capture).bits := hg.captured
    simpa [predicateGuard, hcaptured] using hcontrol.1

theorem rust_prepared_assignment_sound {Input : Type} (sem : Semantics Input)
    (meaning : Meaning Input) (source old : SourceMetadata)
    (sourceValues oldValues : Input → Value)
    (hs : Sound sem meaning (metadataView source) sourceValues)
    (ho : Sound sem meaning (metadataView old) oldValues)
    (guard : Input → Bool) (dependent : Bool) (hg : GuardSound sem guard dependent) :
    symbolic.assign_metadata (guard sem.capture) dependent source old
      ⦃ result => Sound sem meaning (metadataView result) (guardedValues guard sourceValues oldValues) ⦄ := by
  step as ⟨result, hresult⟩
  rw [hresult]
  exact assign_sound sem meaning _ _ _ _ hs ho guard dependent hg

theorem rust_binary_preserves_soundness {Input : Type} (sem : Semantics Input)
    (meaning : Meaning Input) (op : symbolic.BinaryOp) (left right : SourceMetadata)
    (leftValues rightValues : Input → Value)
    (hl : Sound sem meaning (metadataView left) leftValues)
    (hr : Sound sem meaning (metadataView right) rightValues) :
    symbolic.binary op left right
      ⦃ result => Sound sem meaning (metadataView result)
        (binaryValues (operationView op) leftValues rightValues) ⦄ := by
  step as ⟨result, hresult⟩
  rw [hresult]
  exact binary_preserves_soundness sem meaning _ _ _ _ _ hl hr

theorem rust_prepared_assignment_derives_register_word {Input : Type} (sem : Semantics Input)
    (meaning : Meaning Input) (source old : SourceMetadata)
    (sourceValues oldValues : Input → Value)
    (hs : Sound sem meaning (metadataView source) sourceValues)
    (ho : Sound sem meaning (metadataView old) oldValues)
    (guard : Input → Bool) (dependent : Bool) (hg : GuardSound sem guard dependent)
    (hm : ∀ id input lane, sem.word id input lane = byteAt (meaning id input) lane) :
    symbolic.assign_metadata (guard sem.capture) dependent source old
      ⦃ result => Producers.RegisterWordSound sem (registerWord (metadataView result))
        (byteValues (guardedValues guard sourceValues oldValues)) ⦄ := by
  step with rust_prepared_assignment_sound sem meaning source old sourceValues oldValues hs ho guard dependent hg
    as ⟨result, hresult⟩
  exact derives_register_word_sound sem meaning _ _ hresult hm

structure SourceState (Input : Type) where
  metadata : RegisterKey → SourceMetadata
  values : RegisterKey → Input → Value
  meaning : Meaning Input

def sourceView {Input : Type} (state : SourceState Input) : State Input :=
  ⟨fun key => metadataView (state.metadata key), state.values, state.meaning⟩

def sourceCopy {Input : Type} (sem : Semantics Input) (state : SourceState Input)
    (source destination predicate : RegisterKey) : Result (SourceState Input) := do
  let operand := state.metadata source
  let frozen ← symbolic.prepare operand.captured operand.history operand.tainted operand.defined
  let guard := predicateGuard (state.values predicate)
  let assigned ← symbolic.assign_metadata (guard sem.capture)
    (state.metadata predicate).tainted frozen (state.metadata destination)
  .ok { state with
    metadata := fun key => if key = destination then assigned else state.metadata key
    values := fun key => if key = destination then
      guardedValues guard (state.values source) (state.values key) else state.values key }

theorem rust_prepare_copy_preserves_state {Input : Type} (sem : Semantics Input)
    (state : SourceState Input) (hs : (sourceView state).Sound sem)
    (source destination predicate : RegisterKey) :
    sourceCopy sem state source destination predicate
      ⦃ result => (sourceView result).Sound sem ⦄ := by
  have hsource := hs source
  have hold := hs destination
  have hpredicate := hs predicate
  have hg := derives_guard_sound sem state.meaning _ _ hpredicate
  unfold sourceCopy
  step as ⟨frozen, hfrozen⟩
  have hf : Sound sem state.meaning (metadataView frozen) (state.values source) := by
    rw [hfrozen]
    exact hsource
  step with rust_prepared_assignment_sound sem state.meaning frozen (state.metadata destination)
    (state.values source) (state.values destination) hf hold
    (predicateGuard (state.values predicate)) (state.metadata predicate).tainted hg
    as ⟨assigned, hassigned⟩
  intro key
  by_cases hk : key = destination
  · subst key
    simpa [sourceView] using hassigned
  · simpa [sourceView, hk] using hs key

theorem rust_prepared_copy_derives_payload {Input : Type} (sem : Semantics Input)
    (state : SourceState Input) (hs : (sourceView state).Sound sem)
    (source destination predicate : RegisterKey) (keys : Nat → RegisterKey)
    (registers : List Nat) (hsize : 4 * registers.length ≤ Usize.max)
    (hm : ∀ id input lane, sem.word id input lane = byteAt (state.meaning id input) lane) :
    sourceCopy sem state source destination predicate
      ⦃ result => PayloadSound sem
        (Producers.registerPayload
          (fun index => registerWord (metadataView (result.metadata (keys index)))) registers hsize)
        (Producers.registerValues (fun index => byteValues (result.values (keys index))) registers) ⦄ := by
  have hmeaning : ∀ result,
      sourceCopy sem state source destination predicate = .ok result → result.meaning = state.meaning := by
    intro result hresult
    cases hg : predicateGuard (state.values predicate) sem.capture <;>
      cases hd : (state.metadata predicate).tainted <;>
      simp [sourceCopy, symbolic.prepare, symbolic.assign_metadata, hg, hd] at hresult <;>
      exact (congrArg SourceState.meaning hresult).symm
  obtain ⟨result, hactual, hsound⟩ := spec_returns _ _
    (rust_prepare_copy_preserves_state sem state hs source destination predicate)
  rw [hactual]
  simp only [WP.spec_ok]
  apply derives_payload_sound sem (sourceView result) hsound keys registers hsize
  simpa [sourceView, hmeaning result hactual] using hm

end MemoryWrite.Symbolic
