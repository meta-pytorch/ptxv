/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.WellFormed

open Aeneas Aeneas.Std PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace MemoryWrite
namespace AsyncSafety

structure QueueWellFormed (state : CentralState) : Prop where
  unique : (state.pending.map PendingChecked.ticket).Nodup
  bounded : ∀ pending ∈ state.pending, pending.ticket < state.nextTicket

structure StateWellFormed (state : CentralState) : Prop where
  storage : Storage.WellFormed state.storage
  queue : QueueWellFormed state

theorem initial_queue_wellFormed (storage : Storage) (markers : WrittenMarkers) (tracking : WrittenTracking) :
    QueueWellFormed { storage := storage, markers := markers, tracking := tracking } := by
  constructor <;> simp

theorem append_operation_wellFormed (state : CentralState) (operation : PendingCheckedOperation)
    (hstate : QueueWellFormed state) :
    QueueWellFormed { state with
      pending := state.pending ++ [⟨state.nextTicket, operation⟩]
      nextTicket := state.nextTicket + 1 } := by
  constructor
  · simp only [List.map_append, List.map_cons, List.map_nil]
    apply List.nodup_append.mpr
    refine ⟨hstate.unique, by simp, ?_⟩
    intro left hleft right hright heq
    obtain ⟨pending, hpending, rfl⟩ := List.mem_map.mp hleft
    have hright : right = state.nextTicket := by simpa using hright
    have hbound := hstate.bounded pending hpending
    omega
  · intro pending hpending
    change pending.ticket < state.nextTicket + 1
    rcases List.mem_append.mp hpending with hbefore | hnew
    · have := hstate.bounded pending hbefore
      omega
    · have heq : pending = ⟨state.nextTicket, operation⟩ := by simpa using hnew
      subst pending
      simp

theorem issueChecked_wellFormed (state : CentralState) (requests : List CheckedWrite)
    (hstate : QueueWellFormed state) : QueueWellFormed (issueChecked state requests).1 :=
  append_operation_wellFormed state (.captured requests) hstate

theorem issueDeferredTransfer_wellFormed (state : CentralState) (transfers : List CheckedTransfer)
    (hstate : QueueWellFormed state) : QueueWellFormed (issueDeferredTransfer state transfers).1 :=
  append_operation_wellFormed state (.deferred transfers) hstate

theorem issueChecked_fresh (state : CentralState) (requests : List CheckedWrite)
    (hstate : QueueWellFormed state) :
    ∀ pending ∈ state.pending, pending.ticket ≠ (issueChecked state requests).2 := by
  intro pending hpending
  have := hstate.bounded pending hpending
  change pending.ticket ≠ state.nextTicket
  omega

theorem issueTransfer_wellFormed (layout : MemoryLayout) (before after : CentralState)
    (transfers : List CheckedTransfer) (ticket : Nat) (hstate : QueueWellFormed before)
    (hrun : issueTransfer layout before transfers = .ok (after, ticket)) : QueueWellFormed after := by
  unfold issueTransfer at hrun
  cases hcapture : captureTransfers layout before.storage transfers with
  | error error => simp [hcapture, Bind.bind, Except.bind] at hrun
  | ok captured =>
    have heq : (after, ticket) = issueChecked before (captured.map CapturedCheckedTransfer.request) := by
      simpa [hcapture, Bind.bind, Except.bind] using hrun.symm
    have hafter : after = (issueChecked before (captured.map CapturedCheckedTransfer.request)).1 :=
      congrArg Prod.fst heq
    rw [hafter]
    exact issueChecked_wellFormed before _ hstate

theorem write_queue_frame (layout : MemoryLayout) (before after : CentralState) (request : CheckedWrite)
    (hrun : write layout before request = .ok after) :
    after.pending = before.pending ∧ after.nextTicket = before.nextTicket := by
  unfold write at hrun
  cases hstep : applyChecked layout before.storage request with
  | error error => simp [hstep] at hrun
  | ok storage =>
    have heq : after = { before with
        storage := storage
        markers := before.markers.record before.tracking request.normalizedEffect } := by
      simpa [hstep] using hrun.symm
    subst after
    exact ⟨rfl, rfl⟩

theorem runChecked_queue_frame (layout : MemoryLayout) (before after : CentralState)
    (requests : List CheckedWrite) (hrun : runChecked layout before requests = .ok after) :
    after.pending = before.pending ∧ after.nextTicket = before.nextTicket := by
  induction requests generalizing before with
  | nil =>
    have heq : after = before := by simpa [runChecked] using hrun.symm
    subst after
    exact ⟨rfl, rfl⟩
  | cons request requests ih =>
    cases hstep : write layout before request with
    | error error => simp [runChecked, hstep, Bind.bind, Except.bind] at hrun
    | ok middle =>
      have hrest : runChecked layout middle requests = .ok after := by
        simpa [runChecked, hstep, Bind.bind, Except.bind] using hrun
      have hfirst := write_queue_frame layout before middle request hstep
      have hfinish := ih middle hrest
      exact ⟨hfinish.1.trans hfirst.1, hfinish.2.trans hfirst.2⟩

theorem finishChecked_queue_frame (layout : MemoryLayout) (before after : CentralState) (ticket : Nat)
    (hrun : finishChecked layout before ticket = .ok after) :
    after.pending = before.pending.filter (fun pending => pending.ticket != ticket) ∧
      after.nextTicket = before.nextTicket := by
  unfold finishChecked at hrun
  cases hfind : before.pending.find? (fun pending => pending.ticket == ticket) with
  | none => simp [hfind, Bind.bind, Except.bind] at hrun
  | some pending =>
    cases hprepare : preparePendingChecked layout before.storage pending.operation with
    | error error => simp [hfind, hprepare, Bind.bind, Except.bind] at hrun
    | ok requests =>
      cases hstep : runChecked layout before requests with
      | error error => simp [hfind, hprepare, hstep, Bind.bind, Except.bind] at hrun
      | ok next =>
        have heq : after = { next with
            pending := before.pending.filter (fun pending => pending.ticket != ticket) } := by
          simpa [hfind, hprepare, hstep, Bind.bind, Except.bind] using hrun.symm
        subst after
        exact ⟨rfl, (runChecked_queue_frame layout before next requests hstep).2⟩

theorem finishChecked_wellFormed (layout : MemoryLayout) (before after : CentralState) (ticket : Nat)
    (hstate : QueueWellFormed before) (hrun : finishChecked layout before ticket = .ok after) :
    QueueWellFormed after := by
  obtain ⟨hpending, hnext⟩ := finishChecked_queue_frame layout before after ticket hrun
  constructor
  · rw [hpending]
    exact (List.Sublist.map PendingChecked.ticket List.filter_sublist).nodup hstate.unique
  · intro pending hmem
    rw [hpending] at hmem
    rw [hnext]
    exact hstate.bounded pending (List.mem_filter.mp hmem).1

theorem runChecked_state_wellFormed (layout : MemoryLayout) (before after : CentralState)
    (requests : List CheckedWrite) (hstate : StateWellFormed before)
    (hrun : runChecked layout before requests = .ok after) : StateWellFormed after := by
  have hframe := runChecked_queue_frame layout before after requests hrun
  refine ⟨runChecked_wellFormed layout before after requests hstate.storage hrun, ?_⟩
  constructor
  · rw [hframe.1]
    exact hstate.queue.unique
  · intro pending hpending
    rw [hframe.1] at hpending
    rw [hframe.2]
    exact hstate.queue.bounded pending hpending

theorem finishChecked_state_wellFormed (layout : MemoryLayout) (before after : CentralState)
    (ticket : Nat) (hstate : StateWellFormed before)
    (hrun : finishChecked layout before ticket = .ok after) : StateWellFormed after := by
  refine ⟨?_, finishChecked_wellFormed layout before after ticket hstate.queue hrun⟩
  unfold finishChecked at hrun
  cases hfind : before.pending.find? (fun pending => pending.ticket == ticket) with
  | none => simp [hfind, Bind.bind, Except.bind] at hrun
  | some pending =>
    cases hprepare : preparePendingChecked layout before.storage pending.operation with
    | error error => simp [hfind, hprepare, Bind.bind, Except.bind] at hrun
    | ok requests =>
      cases hstep : runChecked layout before requests with
      | error error => simp [hfind, hprepare, hstep, Bind.bind, Except.bind] at hrun
      | ok next =>
        have heq : after = { next with
            pending := before.pending.filter (fun pending => pending.ticket != ticket) } := by
          simpa [hfind, hprepare, hstep, Bind.bind, Except.bind] using hrun.symm
        subst after
        exact runChecked_wellFormed layout before next requests hstate.storage hstep

theorem finishChecked_membership (layout : MemoryLayout) (before after : CentralState)
    (ticket : Nat) (pending : PendingChecked) (hrun : finishChecked layout before ticket = .ok after) :
    pending ∈ after.pending ↔ pending ∈ before.pending ∧ pending.ticket ≠ ticket := by
  rw [(finishChecked_queue_frame layout before after ticket hrun).1]
  simp

theorem finishChecked_at_most_once (layout nextLayout : MemoryLayout) (before after : CentralState)
    (ticket : Nat) (hrun : finishChecked layout before ticket = .ok after) :
    finishChecked nextLayout after ticket = .error .missingPending := by
  have hnone : after.pending.find? (fun pending => pending.ticket == ticket) = none := by
    apply List.find?_eq_none.mpr
    intro pending hmem
    have hne := ((finishChecked_membership layout before after ticket pending hrun).mp hmem).2
    simpa using hne
  simp [finishChecked, hnone, Bind.bind, Except.bind]

theorem finishChecked_ticket_retired (layout : MemoryLayout) (before after : CentralState)
    (ticket : Nat) (hstate : QueueWellFormed before) (hrun : finishChecked layout before ticket = .ok after) :
    ticket < after.nextTicket := by
  have hnext := (finishChecked_queue_frame layout before after ticket hrun).2
  cases hfind : before.pending.find? (fun pending => pending.ticket == ticket) with
  | none => simp [finishChecked, hfind, Bind.bind, Except.bind] at hrun
  | some pending =>
    have hmem := List.mem_of_find?_eq_some hfind
    have heq : pending.ticket = ticket := by simpa using List.find?_some hfind
    rw [hnext, ← heq]
    exact hstate.bounded pending hmem

theorem issueChecked_does_not_reuse (state : CentralState) (requests : List CheckedWrite) (ticket : Nat)
    (hretired : ticket < state.nextTicket) : (issueChecked state requests).2 ≠ ticket := by
  change state.nextTicket ≠ ticket
  omega

theorem finishChecked_progress (layout : MemoryLayout) (before : CentralState)
    (ticket : Nat) (pending : PendingChecked) (requests : List CheckedWrite)
    (hfind : before.pending.find? (fun pending => pending.ticket == ticket) = some pending)
    (hprepare : preparePendingChecked layout before.storage pending.operation = .ok requests)
    (hsupported : ∀ request ∈ requests, SupportedWrite layout before.storage request) :
    ∃ after, finishChecked layout before ticket = .ok after := by
  obtain ⟨next, hrun, _⟩ := runChecked_progress layout before requests hsupported
  refine ⟨{ next with pending := before.pending.filter (fun pending => pending.ticket != ticket) }, ?_⟩
  simp [finishChecked, hfind, hprepare, hrun, Bind.bind, Except.bind]

abbrev Generations := Key → Nat

def bumpSpan (generations : Generations) (start : Key) (width : Usize) : Generations :=
  fun key => if Touches start width.val key then generations key + 1 else generations key

structure ByteLease where
  key : Key
  generation : Nat
  deriving DecidableEq

def leaseKeys (generations : Generations) (keys : List Key) : List ByteLease :=
  keys.map fun key => ⟨key, generations key⟩

def LeaseCurrent (generations : Generations) (leases : List ByteLease) : Prop :=
  ∀ lease ∈ leases, generations lease.key = lease.generation

instance (generations : Generations) (leases : List ByteLease) : Decidable (LeaseCurrent generations leases) :=
  inferInstanceAs (Decidable (∀ lease ∈ leases, _))

theorem leaseKeys_current (generations : Generations) (keys : List Key) :
    LeaseCurrent generations (leaseKeys generations keys) := by
  intro lease hlease
  obtain ⟨key, _, rfl⟩ := List.mem_map.mp hlease
  rfl

theorem leaseKeys_stale_after_bump (generations : Generations) (keys : List Key)
    (start key : Key) (width : Usize) (hkey : key ∈ keys) (htouched : Touches start width.val key) :
    ¬LeaseCurrent (bumpSpan generations start width) (leaseKeys generations keys) := by
  intro hcurrent
  have hlease : ByteLease.mk key (generations key) ∈ leaseKeys generations keys :=
    List.mem_map.mpr ⟨key, hkey, rfl⟩
  have h := hcurrent _ hlease
  simp only [bumpSpan, ite_eq_left htouched] at h
  omega

theorem leaseCurrent_bump_disjoint (generations : Generations) (leases : List ByteLease)
    (start : Key) (width : Usize) (hdisjoint : ∀ lease ∈ leases, ¬Touches start width.val lease.key) :
    LeaseCurrent (bumpSpan generations start width) leases ↔ LeaseCurrent generations leases := by
  unfold LeaseCurrent
  simp only [bumpSpan]
  constructor <;> intro h lease hlease
  · simpa only [ite_eq_right (hdisjoint lease hlease)] using h lease hlease
  · simpa only [ite_eq_right (hdisjoint lease hlease)] using h lease hlease

def LeaseExpired (generations : Generations) (leases : List ByteLease) : Prop :=
  ∃ lease ∈ leases, lease.generation < generations lease.key

theorem bumpSpan_monotone (generations : Generations) (start key : Key) (width : Usize) :
    generations key ≤ bumpSpan generations start width key := by
  unfold bumpSpan
  split <;> omega

theorem leaseKeys_expired_after_bump (generations : Generations) (keys : List Key)
    (start key : Key) (width : Usize) (hkey : key ∈ keys) (htouched : Touches start width.val key) :
    LeaseExpired (bumpSpan generations start width) (leaseKeys generations keys) := by
  refine ⟨⟨key, generations key⟩, List.mem_map.mpr ⟨key, hkey, rfl⟩, ?_⟩
  simp [bumpSpan, htouched]

theorem LeaseExpired.monotone (before after : Generations) (leases : List ByteLease)
    (hgenerations : ∀ key, before key ≤ after key) (hexpired : LeaseExpired before leases) :
    LeaseExpired after leases := by
  obtain ⟨lease, hlease, hstale⟩ := hexpired
  exact ⟨lease, hlease, hstale.trans_le (hgenerations lease.key)⟩

theorem LeaseExpired.not_current (generations : Generations) (leases : List ByteLease)
    (hexpired : LeaseExpired generations leases) : ¬LeaseCurrent generations leases := by
  obtain ⟨lease, hlease, hstale⟩ := hexpired
  intro hcurrent
  have := hcurrent lease hlease
  omega

def requestKeys (request : CheckedWrite) : List Key :=
  if request.effect.enabled then
    Storage.spanKeys request.location.key (Slice.len request.effect.payload) ++
      request.effect.words.flatMap (fun update => Storage.spanKeys update.key 4#usize)
  else []

def operationKeys : PendingCheckedOperation → List Key
  | .captured requests => requests.flatMap requestKeys
  | .deferred transfers => transfers.flatMap fun transfer =>
      Storage.spanKeys transfer.destination.key transfer.width ++
        match transfer.source with
        | none => []
        | some source => Storage.spanKeys source.key transfer.width

structure PendingLease where
  ticket : Nat
  bytes : List ByteLease

structure LifetimeState where
  central : CentralState
  generations : Generations := fun _ => 0
  leases : List PendingLease := []

inductive LifetimeError where
  | write (error : WriteError)
  | missingLease
  | staleAllocation
  deriving DecidableEq, Repr

def issueOperation (state : LifetimeState) (operation : PendingCheckedOperation) : LifetimeState × Nat :=
  let issued := match operation with
    | .captured requests => issueChecked state.central requests
    | .deferred transfers => issueDeferredTransfer state.central transfers
  ({ state with
      central := issued.1
      leases := ⟨issued.2, leaseKeys state.generations (operationKeys operation)⟩ :: state.leases }, issued.2)

def issueCapturedTransfer (layout : MemoryLayout) (state : LifetimeState) (transfers : List CheckedTransfer) :
    Except WriteError (LifetimeState × Nat) := do
  let captured ← captureTransfers layout state.central.storage transfers
  .ok (issueOperation state (.captured (captured.map CapturedCheckedTransfer.request)))

def invalidateSpan (state : LifetimeState) (start : Key) (width : Usize) : LifetimeState :=
  { state with generations := bumpSpan state.generations start width }

def finish (layout : MemoryLayout) (state : LifetimeState) (ticket : Nat) : Except LifetimeError LifetimeState :=
  match state.leases.find? (fun lease => lease.ticket == ticket) with
  | none => .error .missingLease
  | some lease =>
    if LeaseCurrent state.generations lease.bytes then
      match finishChecked layout state.central ticket with
      | .error error => .error (.write error)
      | .ok central => .ok { state with
          central := central
          leases := state.leases.filter (fun lease => lease.ticket != ticket) }
    else .error .staleAllocation

theorem finish_rejects_expired (layout : MemoryLayout) (state : LifetimeState)
    (ticket : Nat) (lease : PendingLease)
    (hfind : state.leases.find? (fun lease => lease.ticket == ticket) = some lease)
    (hexpired : LeaseExpired state.generations lease.bytes) :
    finish layout state ticket = .error .staleAllocation := by
  simp only [finish, hfind]
  exact ite_eq_right (LeaseExpired.not_current state.generations lease.bytes hexpired)

theorem issueOperation_lease (state : LifetimeState) (operation : PendingCheckedOperation) :
    (issueOperation state operation).1.leases.find?
      (fun lease => lease.ticket == (issueOperation state operation).2) =
      some ⟨(issueOperation state operation).2, leaseKeys state.generations (operationKeys operation)⟩ := by
  cases operation <;> simp [issueOperation, issueChecked, issueDeferredTransfer]

theorem issueOperation_generations (state : LifetimeState) (operation : PendingCheckedOperation) :
    (issueOperation state operation).1.generations = state.generations := by
  cases operation <;> rfl

theorem issued_then_invalidated_rejected (layout : MemoryLayout) (state : LifetimeState)
    (operation : PendingCheckedOperation) (start key : Key) (width : Usize)
    (hkey : key ∈ operationKeys operation) (htouched : Touches start width.val key) :
    finish layout (invalidateSpan (issueOperation state operation).1 start width)
      (issueOperation state operation).2 = .error .staleAllocation := by
  have hstale := leaseKeys_stale_after_bump state.generations (operationKeys operation)
    start key width hkey htouched
  unfold finish
  simp only [invalidateSpan, issueOperation_generations, issueOperation_lease]
  exact ite_eq_right hstale

def freeTmem (layout : MemoryLayout) (state : LifetimeState) (cta : U64)
    (lane : Fin 128) (column : Fin 512) (width : Usize) : MemoryLayout × LifetimeState :=
  let freed := freeTmemSpan layout state.central.storage cta lane column width
  (freed.1, invalidateSpan { state with central := { state.central with storage := freed.2 } }
    (MemoryLocation.tmem cta lane column 0).key width)

def allocateTmem (layout : MemoryLayout) (state : LifetimeState) (cta : U64)
    (lane : Fin 128) (column : Fin 512) (width : Usize) :
    Except WriteError (MemoryLayout × LifetimeState) := do
  let allocated ← MemoryWrite.allocateTmem layout state.central.storage cta lane column width
  .ok (allocated.1, invalidateSpan { state with central := { state.central with storage := allocated.2 } }
    (MemoryLocation.tmem cta lane column 0).key width)

theorem freeTmem_generations_monotone (layout : MemoryLayout) (state : LifetimeState) (cta : U64)
    (lane : Fin 128) (column : Fin 512) (width : Usize) (key : Key) :
    state.generations key ≤ (freeTmem layout state cta lane column width).2.generations key :=
  bumpSpan_monotone state.generations _ key width

theorem allocateTmem_generations (layout nextLayout : MemoryLayout) (before after : LifetimeState)
    (cta : U64) (lane : Fin 128) (column : Fin 512) (width : Usize)
    (hrun : allocateTmem layout before cta lane column width = .ok (nextLayout, after)) :
    after.generations = bumpSpan before.generations (MemoryLocation.tmem cta lane column 0).key width := by
  unfold allocateTmem at hrun
  cases hallocate : MemoryWrite.allocateTmem layout before.central.storage cta lane column width with
  | error error => simp [hallocate, Bind.bind, Except.bind] at hrun
  | ok allocated =>
    have heq : (nextLayout, after) =
        (allocated.1, invalidateSpan { before with central := { before.central with storage := allocated.2 } }
          (MemoryLocation.tmem cta lane column 0).key width) := by
      simpa [hallocate, Bind.bind, Except.bind] using hrun.symm
    have hafter : after =
        invalidateSpan { before with central := { before.central with storage := allocated.2 } }
          (MemoryLocation.tmem cta lane column 0).key width := congrArg Prod.snd heq
    rw [hafter]
    rfl

theorem allocateTmem_generations_monotone (layout nextLayout : MemoryLayout) (before after : LifetimeState)
    (cta : U64) (lane : Fin 128) (column : Fin 512) (width : Usize) (key : Key)
    (hrun : allocateTmem layout before cta lane column width = .ok (nextLayout, after)) :
    before.generations key ≤ after.generations key := by
  rw [allocateTmem_generations layout nextLayout before after cta lane column width hrun]
  exact bumpSpan_monotone before.generations _ key width

theorem allocateTmem_leases (layout nextLayout : MemoryLayout) (before after : LifetimeState)
    (cta : U64) (lane : Fin 128) (column : Fin 512) (width : Usize)
    (hrun : allocateTmem layout before cta lane column width = .ok (nextLayout, after)) :
    after.leases = before.leases := by
  unfold allocateTmem at hrun
  cases hallocate : MemoryWrite.allocateTmem layout before.central.storage cta lane column width with
  | error error => simp [hallocate, Bind.bind, Except.bind] at hrun
  | ok allocated =>
    have heq : (nextLayout, after) =
        (allocated.1, invalidateSpan { before with central := { before.central with storage := allocated.2 } }
          (MemoryLocation.tmem cta lane column 0).key width) := by
      simpa [hallocate, Bind.bind, Except.bind] using hrun.symm
    have hafter : after =
        invalidateSpan { before with central := { before.central with storage := allocated.2 } }
          (MemoryLocation.tmem cta lane column 0).key width := congrArg Prod.snd heq
    rw [hafter]
    rfl

theorem freeTmem_rejects_issued (layout : MemoryLayout) (state : LifetimeState)
    (operation : PendingCheckedOperation) (cta : U64) (lane : Fin 128) (column : Fin 512)
    (width : Usize) (key : Key) (hkey : key ∈ operationKeys operation)
    (htouched : Touches (MemoryLocation.tmem cta lane column 0).key width.val key) :
    finish (freeTmem layout (issueOperation state operation).1 cta lane column width).1
      (freeTmem layout (issueOperation state operation).1 cta lane column width).2
      (issueOperation state operation).2 = .error .staleAllocation := by
  have hstale := leaseKeys_stale_after_bump state.generations (operationKeys operation)
    (MemoryLocation.tmem cta lane column 0).key key width hkey htouched
  unfold finish
  simp only [freeTmem, invalidateSpan, issueOperation_generations, issueOperation_lease]
  exact ite_eq_right hstale

theorem free_and_reallocate_rejects_issued (layout nextLayout : MemoryLayout) (state after : LifetimeState)
    (operation : PendingCheckedOperation) (cta : U64) (lane : Fin 128) (column : Fin 512)
    (width : Usize) (key : Key) (hkey : key ∈ operationKeys operation)
    (htouched : Touches (MemoryLocation.tmem cta lane column 0).key width.val key)
    (hrun : allocateTmem
      (freeTmem layout (issueOperation state operation).1 cta lane column width).1
      (freeTmem layout (issueOperation state operation).1 cta lane column width).2
      cta lane column width = .ok (nextLayout, after)) :
    finish nextLayout after (issueOperation state operation).2 = .error .staleAllocation := by
  let freed := freeTmem layout (issueOperation state operation).1 cta lane column width
  have hexpired : LeaseExpired freed.2.generations (leaseKeys state.generations (operationKeys operation)) := by
    simpa only [freed, freeTmem, invalidateSpan, issueOperation_generations] using
      leaseKeys_expired_after_bump state.generations (operationKeys operation)
        (MemoryLocation.tmem cta lane column 0).key key width hkey htouched
  have hmonotone : ∀ key, freed.2.generations key ≤ after.generations key := by
    intro key
    exact allocateTmem_generations_monotone freed.1 nextLayout freed.2 after cta lane column width key hrun
  have hafter := LeaseExpired.monotone freed.2.generations after.generations _ hmonotone hexpired
  have hleases : after.leases = (issueOperation state operation).1.leases :=
    allocateTmem_leases freed.1 nextLayout freed.2 after cta lane column width hrun
  apply finish_rejects_expired nextLayout after (issueOperation state operation).2
    ⟨(issueOperation state operation).2, leaseKeys state.generations (operationKeys operation)⟩
  · rw [hleases]
    exact issueOperation_lease state operation
  · exact hafter

theorem finish_delegates (layout : MemoryLayout) (before after : LifetimeState) (ticket : Nat)
    (hrun : finish layout before ticket = .ok after) :
    finishChecked layout before.central ticket = .ok after.central ∧
      after.generations = before.generations ∧
      after.leases = before.leases.filter (fun lease => lease.ticket != ticket) := by
  unfold finish at hrun
  cases hfind : before.leases.find? (fun lease => lease.ticket == ticket) with
  | none => simp [hfind] at hrun
  | some lease =>
    simp only [hfind] at hrun
    split at hrun
    · cases hstep : finishChecked layout before.central ticket with
      | error error => simp [hstep] at hrun
      | ok central =>
        have heq : after = { before with
            central := central
            leases := before.leases.filter (fun lease => lease.ticket != ticket) } := by
          simpa [hstep] using hrun.symm
        subst after
        exact ⟨rfl, rfl, rfl⟩
    · simp at hrun

theorem finish_preserves_representation {Input : Type} (sem : Semantics Input)
    (memory : Memory Input) (layout : MemoryLayout) (before after : LifetimeState)
    (ticket : Nat) (pending : PendingChecked) (descriptions : List (DescribedChecked Input))
    (hfind : before.central.pending.find? (fun pending => pending.ticket == ticket) = some pending)
    (hprepare : preparePendingChecked layout before.central.storage pending.operation =
      .ok (descriptions.map DescribedChecked.request))
    (hstate : Storage.Represents sem memory before.central.storage)
    (hsound : EffectsSound sem memory (normalizedDescriptions descriptions))
    (hrun : finish layout before ticket = .ok after) :
    Storage.Represents sem (effectsMemory memory (normalizedDescriptions descriptions)) after.central.storage ∧
      after.central.markers = recordChecked before.central.markers before.central.tracking
        (descriptions.map DescribedChecked.request) ∧
      after.generations = before.generations := by
  have hdelegate := finish_delegates layout before after ticket hrun
  have h := finishChecked_preserves_representation sem memory layout before.central after.central
    ticket pending descriptions hfind hprepare hstate hsound hdelegate.1
  exact ⟨h.1, h.2.1, hdelegate.2.1⟩

theorem finish_at_most_once (layout nextLayout : MemoryLayout) (before after : LifetimeState)
    (ticket : Nat) (hrun : finish layout before ticket = .ok after) :
    finish nextLayout after ticket = .error .missingLease := by
  have hleases := (finish_delegates layout before after ticket hrun).2.2
  have hnone : after.leases.find? (fun lease => lease.ticket == ticket) = none := by
    apply List.find?_eq_none.mpr
    intro lease hlease
    rw [hleases] at hlease
    simpa using (List.mem_filter.mp hlease).2
  simp [finish, hnone]

end AsyncSafety
end MemoryWrite
