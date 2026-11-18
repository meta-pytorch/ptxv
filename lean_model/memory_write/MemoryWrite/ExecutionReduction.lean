/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.AtomicComponentOrder
import MemoryWrite.Theorem

open Aeneas Aeneas.Std

set_option autoImplicit false

namespace MemoryWrite.ExecutionReduction

structure RegisterState where
  scalar : List U64
  predicates : List U64
  predicateDefined : List U64
  scalarDefined : List U64
  vectorUndefined : List (U16 × U64)
  uniform : List U64
  uniformDefined : List U64
  uniformPredicates : U8
  uniformPredicateDefined : U8
  deriving DecidableEq

structure MemoryByte where
  bits : U8
  defined : Bool
  deriving DecidableEq

structure StoreSample where
  initial : List MemoryByte
  current : List MemoryByte
  deriving DecidableEq

structure PollEvidence where
  before : RegisterState
  after : RegisterState
  stores : List StoreSample
  deriving DecidableEq

def storesUnchanged (stores : List StoreSample) : Prop :=
  ∀ sample ∈ stores, sample.current = sample.initial

instance (stores : List StoreSample) : Decidable (storesUnchanged stores) :=
  inferInstanceAs (Decidable (∀ sample ∈ stores, sample.current = sample.initial))

def accepts (e : PollEvidence) : Bool :=
  decide (e.before = e.after ∧ storesUnchanged e.stores)

theorem accepts_iff (e : PollEvidence) :
    accepts e = true ↔ e.before = e.after ∧ storesUnchanged e.stores := by
  simp [accepts]

structure Observation (Event : Type) where
  registers : RegisterState
  events : List Event

structure PollSegment (Event : Type) where
  evidence : PollEvidence
  reads : List Event

def executeSegment {Event : Type} (segment : PollSegment Event)
    (priorEvents : List Event) : Observation Event :=
  ⟨segment.evidence.after, priorEvents ++ segment.reads⟩

def suppressSegment {Event : Type} (segment : PollSegment Event)
    (priorEvents : List Event) : Observation Event :=
  ⟨segment.evidence.before, priorEvents ++ segment.reads⟩

theorem checked_poll_preserves_observation {Event : Type}
    (segment : PollSegment Event) (priorEvents : List Event)
    (h : accepts segment.evidence = true) :
    executeSegment segment priorEvents = suppressSegment segment priorEvents ∧
      ∀ sample ∈ segment.evidence.stores, sample.current = sample.initial := by
  obtain ⟨hstate, hstores⟩ := (accepts_iff segment.evidence).mp h
  exact ⟨by simp [executeSegment, suppressSegment, hstate], hstores⟩

def auditStores : List StoreSample → Bool
  | [] => true
  | sample :: rest => decide (sample.current = sample.initial) && auditStores rest

theorem auditStores_iff (samples : List StoreSample) :
    auditStores samples = true ↔ storesUnchanged samples := by
  induction samples with
  | nil => simp [auditStores, storesUnchanged]
  | cons sample rest ih =>
    simp [auditStores, storesUnchanged, ih]

theorem audit_append (first second : List StoreSample) :
    auditStores (first ++ second) = (auditStores first && auditStores second) := by
  induction first with
  | nil => simp [auditStores]
  | cons sample rest ih => simp [auditStores, ih, Bool.and_assoc]

theorem changed_store_cannot_be_restored (first second : List StoreSample)
    (h : auditStores first = false) : auditStores (first ++ second) = false := by
  simp [audit_append, h]

def executePolls {Event : Type} : RegisterState → List Event → List (PollSegment Event) →
    Observation Event
  | state, events, [] => ⟨state, events⟩
  | _, events, segment :: rest =>
      executePolls segment.evidence.after (events ++ segment.reads) rest

def suppressPolls {Event : Type} (state : RegisterState) (events : List Event)
    (segments : List (PollSegment Event)) : Observation Event :=
  ⟨state, events ++ segments.flatMap PollSegment.reads⟩

def Connected {Event : Type} : RegisterState → List (PollSegment Event) → Prop
  | _, [] => True
  | state, segment :: rest =>
      segment.evidence.before = state ∧ accepts segment.evidence = true ∧
        Connected segment.evidence.after rest

theorem checked_poll_trace_preserves_observation {Event : Type}
    (segments : List (PollSegment Event)) (state : RegisterState) (events : List Event)
    (h : Connected state segments) :
    executePolls state events segments = suppressPolls state events segments := by
  induction segments generalizing state events with
  | nil => simp [executePolls, suppressPolls]
  | cons segment rest ih =>
    obtain ⟨hbefore, haccept, hrest⟩ := h
    have hafter : segment.evidence.after = state :=
      ((accepts_iff segment.evidence).mp haccept).1.symm.trans hbefore
    rw [executePolls, ih _ _ hrest]
    simp [suppressPolls, hafter, List.append_assoc]

theorem checked_poll_trace_hides_no_counterexample {Event : Type}
    (segments : List (PollSegment Event)) (state : RegisterState) (events : List Event)
    (h : Connected state segments) (safe : Observation Event → Prop)
    (hreduced : safe (suppressPolls state events segments)) :
    safe (executePolls state events segments) := by
  rw [checked_poll_trace_preserves_observation segments state events h]
  exact hreduced

private def emptyRegisters : RegisterState :=
  ⟨[], [], [], [], [], [], [], 0#u8, 0#u8⟩

private def uniformChange : PollEvidence :=
  ⟨emptyRegisters, { emptyRegisters with uniform := [1#u64] }, []⟩

private def definednessChange : PollEvidence :=
  ⟨emptyRegisters, { emptyRegisters with scalarDefined := [1#u64] }, []⟩

private def changedByte : StoreSample :=
  ⟨[⟨0#u8, true⟩], [⟨1#u8, true⟩]⟩

private def restoredByte : StoreSample :=
  ⟨[⟨0#u8, true⟩], [⟨0#u8, true⟩]⟩

theorem historical_uniform_change_rejected : accepts uniformChange = false := by decide

theorem historical_definedness_change_rejected : accepts definednessChange = false := by decide

theorem historical_write_restore_rejected :
    auditStores [changedByte, restoredByte] = false := by decide

theorem historical_undefined_store_rejected :
    auditStores [⟨[⟨0#u8, true⟩], [⟨0#u8, false⟩]⟩] = false := by decide

theorem component_order_requires_return_observations :
    (∀ order, AtomicComponentOrder.mixedOrder.final =
      (AtomicComponentOrder.jointOrder order).final) ∧
    (∀ order, AtomicComponentOrder.mixedOrder ≠ AtomicComponentOrder.jointOrder order) :=
  AtomicComponentOrder.final_alone_cannot_certify_atomic_returns

end MemoryWrite.ExecutionReduction
