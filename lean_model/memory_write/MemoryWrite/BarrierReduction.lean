/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Aeneas

/-!
Barrier count and release semantics, with parity-indexed completed clocks and sticky findings.
`canSwapPlain` certifies one adjacent swap. `finite_prefix_budget` additionally bounds every
arrival before the selected action; this obligation is absent from the former production
local-ample rule. The scheduler and its learned dependency sets are separate callers.
-/

set_option autoImplicit false
set_option maxHeartbeats 2000000

namespace MemoryWrite.BarrierReduction

abbrev Clock := Nat → Nat

def join (left right : Clock) : Clock := fun actor => max (left actor) (right actor)

 theorem join_commute (base left right : Clock) :
    join (join base left) right = join (join base right) left := by
  funext actor
  simp only [join]
  omega

structure Barrier where
  expected : Nat
  pending : Nat
  phase : Nat
  txZero : Bool
  currentRelease : Clock
  completedRelease : Nat → Clock

structure State where
  barrier : Barrier
  tokens : Nat → Option Nat
  positions : Nat → Nat
  failed : Bool
  findings : List Nat
  acquired : Nat → Clock

theorem State.ext (s t : State) (hb : s.barrier = t.barrier)
    (ht : s.tokens = t.tokens) (hp : s.positions = t.positions) (hf : s.failed = t.failed) (hfind : s.findings = t.findings) (hacq : s.acquired = t.acquired) : s = t := by
  cases s
  cases t
  simp_all

def update {α : Type} (f : Nat → α) (actor : Nat) (value : α) : Nat → α :=
  fun other => if other = actor then value else f other

theorem update_commute {α : Type} (f : Nat → α) (a b : Nat) (x y : α)
    (different : a ≠ b) : update (update f a x) b y = update (update f b y) a x := by
  funext other
  simp only [update]
  by_cases ha : other = a <;> by_cases hb : other = b <;> simp_all

def complete (b : Barrier) : Barrier :=
  if b.pending = 0 ∧ b.txZero = true then
    { b with phase := b.phase + 1, pending := b.expected, currentRelease := fun _ => 0, completedRelease := update b.completedRelease (b.phase % 2) b.currentRelease }
  else b

def decrement (b : Barrier) (count : Nat) (release : Clock) : Barrier :=
  complete { b with pending := b.pending - count, currentRelease := join b.currentRelease release }

def arrive (s : State) (actor count : Nat) (release : Clock) : State :=
  if s.barrier.pending < count then { s with failed := true, findings := s.findings ++ [actor] } else
    { s with barrier := decrement s.barrier count release, tokens := update s.tokens actor (some s.barrier.phase), positions := update s.positions actor (s.positions actor + 1) }

def canSwapPlain (hasCounted : Bool) (pending arrivals : Nat) : Bool :=
  !hasCounted && decide (0 < arrivals ∧ arrivals < pending)

 theorem swap_condition_positive (hasCounted : Bool) (pending arrivals : Nat)
    (accepted : canSwapPlain hasCounted pending arrivals = true) :
    hasCounted = false ∧ 0 < arrivals ∧ arrivals < pending := by
  simpa [canSwapPlain] using accepted

 theorem decrement_noncompleting (b : Barrier) (n : Nat) (r : Clock) (h : n < b.pending) :
    decrement b n r = { b with pending := b.pending - n, currentRelease := join b.currentRelease r } := by
  simp [decrement, complete, Nat.sub_ne_zero_of_lt h]

 theorem decrement_commute (b : Barrier) (n m : Nat) (r q : Clock)
    (hn : n < b.pending) (hm : m < b.pending) (htotal : n + m ≤ b.pending) :
    decrement (decrement b n r) m q = decrement (decrement b m q) n r := by
  rw [decrement_noncompleting b n r hn, decrement_noncompleting b m q hm]
  simp only [decrement, complete]
  have hsub : b.pending - n - m = b.pending - m - n := by omega
  have hmax := join_commute b.currentRelease r q
  simp only [hsub, hmax]

 theorem arrive_commute (s : State) (a b n m : Nat) (r q : Clock)
    (different : a ≠ b) (hn : n < s.barrier.pending) (hm : m < s.barrier.pending)
    (htotal : n + m ≤ s.barrier.pending) :
    arrive (arrive s a n r) b m q = arrive (arrive s b m q) a n r := by
  have firstN : ¬s.barrier.pending < n := by omega
  have firstM : ¬s.barrier.pending < m := by omega
  have secondN : ¬s.barrier.pending - m < n := by omega
  have secondM : ¬s.barrier.pending - n < m := by omega
  have bn := decrement_noncompleting s.barrier n r hn
  have bm := decrement_noncompleting s.barrier m q hm
  have barrierEq := decrement_commute s.barrier n m r q hn hm htotal
  simp only [arrive, ite_false, firstN, firstM]
  simp only [bn, bm, secondN, secondM, ite_false]
  apply State.ext
  · simpa only [bn, bm] using barrierEq
  · exact update_commute s.tokens a b _ _ different
  · simp only [update, ite_eq_right different, ite_eq_right (Ne.symm different)]
    exact update_commute s.positions a b _ _ different
  · rfl
  · rfl
  · rfl

 theorem plain_swap_commutes (s : State) (a b n : Nat) (r q : Clock)
    (different : a ≠ b) (accepted : canSwapPlain false s.barrier.pending n = true) :
    arrive (arrive s a n r) b 1 q = arrive (arrive s b 1 q) a n r := by
  obtain ⟨_, hnpos, hn⟩ := swap_condition_positive false s.barrier.pending n accepted
  exact arrive_commute s a b n 1 r q different hn (by omega) (by omega)

inductive Action where
  | arrival (actor count : Nat) (release : Clock)
  | wait (actor phase : Nat)

def enabled (s : State) : Action → Bool
  | .arrival _ _ _ => true
  | .wait _ phase => decide (phase % 2 ≠ s.barrier.phase % 2)

def step (s : State) (action : Action) : Option State :=
  if enabled s action then
    match action with
    | .arrival actor count release => some (arrive s actor count release)
    | .wait actor phase => some { s with positions := update s.positions actor (s.positions actor + 1), acquired := update s.acquired actor (join (s.acquired actor) (s.barrier.completedRelease (phase % 2))) }
  else none

def run (s : State) : List Action → Option State
  | [] => some s
  | action :: rest => (step s action).bind (fun next => run next rest)

 theorem failure_sticky (s : State) (a : Action) (next : State)
    (failed : s.failed = true) (transition : step s a = some next) : next.failed = true := by
  cases a with
  | arrival actor count release =>
    simp only [step, enabled, ite_true, Option.some.injEq] at transition
    subst next
    unfold arrive
    split <;> simp [failed]
  | wait actor phase =>
    simp only [step] at transition
    split at transition
    · simp only [Option.some.injEq] at transition
      subst next
      exact failed
    · simp at transition

 theorem arrival_failure_sticky (s : State) (actor count : Nat) (release : Clock)
    (failed : s.failed = true) : (arrive s actor count release).failed = true := by
  unfold arrive
  split <;> simp [failed]

 theorem run_append (s : State) (left right : List Action) :
    run s (left ++ right) = (run s left).bind (fun next => run next right) := by
  induction left generalizing s with
  | nil => rfl
  | cons a rest ih => simp only [List.cons_append, run, ih, Option.bind_assoc]

 theorem plain_swap_preserves_suffix (s : State) (a b n : Nat) (r q : Clock) (suffix : List Action)
    (different : a ≠ b) (accepted : canSwapPlain false s.barrier.pending n = true) :
    run s (.arrival a n r :: .arrival b 1 q :: suffix) =
      run s (.arrival b 1 q :: .arrival a n r :: suffix) := by
  simp only [run, step, enabled, ite_true, Option.bind_some]
  rw [plain_swap_commutes s a b n r q different accepted]

 theorem swap_preserves_future_enabledness (s : State) (a b n : Nat) (r q : Clock) (future : Action)
    (different : a ≠ b) (accepted : canSwapPlain false s.barrier.pending n = true) :
    enabled (arrive (arrive s a n r) b 1 q) future =
      enabled (arrive (arrive s b 1 q) a n r) future := by
  rw [plain_swap_commutes s a b n r q different accepted]

inductive Status where
  | complete
  | blocked
  | failed
  deriving DecidableEq, Repr

structure Execution where
  state : State
  status : Status

def execute (s : State) (trace : List Action) : Execution :=
  match trace with
  | [] => ⟨s, if s.failed then .failed else .complete⟩
  | action :: rest =>
    match step s action with
    | none => ⟨s, .blocked⟩
    | some next => execute next rest

 theorem arrive_preserves_findings_prefix (s : State) (actor count : Nat) (release : Clock) :
    ∃ extra, (arrive s actor count release).findings = s.findings ++ extra := by
  unfold arrive
  split
  · exact ⟨[actor], rfl⟩
  · exact ⟨[], by simp⟩

 theorem step_preserves_findings_prefix (s next : State) (action : Action)
    (transition : step s action = some next) :
    ∃ extra, next.findings = s.findings ++ extra := by
  cases action with
  | arrival actor count release =>
    simp only [step] at transition
    split at transition
    · simp only [Option.some.injEq] at transition
      subst next
      exact arrive_preserves_findings_prefix s actor count release
    · simp at transition
  | wait actor phase =>
    simp only [step] at transition
    split at transition
    · simp only [Option.some.injEq] at transition
      subst next
      exact ⟨[], by simp⟩
    · simp at transition

 theorem execute_preserves_findings_prefix (s : State) (trace : List Action) :
    ∃ extra, (execute s trace).state.findings = s.findings ++ extra := by
  induction trace generalizing s with
  | nil => exact ⟨[], by simp [execute]⟩
  | cons action rest ih =>
    cases transition : step s action with
    | none => exact ⟨[], by simp [execute, transition]⟩
    | some next =>
      obtain ⟨first, hfirst⟩ := step_preserves_findings_prefix s next action transition
      obtain ⟨last, hlast⟩ := ih next
      refine ⟨first ++ last, ?_⟩
      simpa [execute, transition, hfirst, List.append_assoc] using hlast

 theorem plain_swap_preserves_execution (s : State) (a b n : Nat) (r q : Clock)
    (suffix : List Action) (different : a ≠ b)
    (accepted : canSwapPlain false s.barrier.pending n = true) :
    execute s (.arrival a n r :: .arrival b 1 q :: suffix) =
      execute s (.arrival b 1 q :: .arrival a n r :: suffix) := by
  simp only [execute, step, enabled, ite_true]
  rw [plain_swap_commutes s a b n r q different accepted]

structure Allocator where
  bitmap : Nat
  permit : Bool
  relinquished : Bool
  data : Nat → Nat

structure AllocationState where
  allocator : Allocator
  pc : Nat
  findings : List Nat

inductive AllocationRequest where
  | disabled
  | invalid
  | valid (available : Bool)
  deriving DecidableEq, Repr

def allocationReady : AllocationRequest → Bool
  | .disabled | .invalid => true
  | .valid available => available

def allocationStep (s : AllocationState) (request : AllocationRequest)
    (commit : Allocator → Allocator) : Option AllocationState :=
  match request with
  | .disabled => some { s with pc := s.pc + 1 }
  | .invalid => some { s with findings := s.findings ++ [s.pc] }
  | .valid false => none
  | .valid true => some { s with allocator := commit s.allocator, pc := s.pc + 1 }

 theorem disabled_allocation_enabled (s : AllocationState) (commit : Allocator → Allocator) :
    allocationReady .disabled = true ∧
      allocationStep s .disabled commit = some { s with pc := s.pc + 1 } := by
  exact ⟨rfl, rfl⟩

 theorem disabled_allocation_progress_and_frame (s : AllocationState)
    (commit : Allocator → Allocator) :
    ∃ next, allocationStep s .disabled commit = some next ∧
      next.pc = s.pc + 1 ∧ next.allocator = s.allocator ∧ next.findings = s.findings := by
  exact ⟨{ s with pc := s.pc + 1 }, rfl, rfl, rfl, rfl⟩

 def initial : State :=
  ⟨⟨3, 3, 0, true, fun _ => 0, fun _ _ => 0⟩, fun _ => none, fun _ => 0, false, [], fun _ _ => 0⟩

 theorem historical_plain_first_faults :
    (arrive (arrive initial 0 1 (fun _ => 0)) 1 3 (fun _ => 0)).failed = true := by decide

 theorem historical_count_first_succeeds :
    (arrive (arrive initial 1 3 (fun _ => 0)) 0 1 (fun _ => 0)).failed = false := by decide

 theorem historical_guard_rejects_counted_kernel :
    canSwapPlain true 3 1 = false := by decide

def oversubscribedInitial : State :=
  ⟨⟨2, 2, 0, true, fun _ => 0, fun _ _ => 0⟩,
    fun _ => none, fun _ => 0, false, [], fun _ _ => 0⟩

 theorem plain_guard_accepts_oversubscribed_initial :
    canSwapPlain false oversubscribedInitial.barrier.pending 1 = true := by decide

 theorem plain_arrival_generation_membership_changes_acquire :
    (execute oversubscribedInitial
      [.arrival 0 1 (fun _ => 1), .arrival 1 1 (fun _ => 0),
       .arrival 2 1 (fun _ => 0), .wait 3 0]).state.acquired 3 0 = 1 ∧
    (execute oversubscribedInitial
      [.arrival 1 1 (fun _ => 0), .arrival 2 1 (fun _ => 0),
       .arrival 0 1 (fun _ => 1), .wait 3 0]).state.acquired 3 0 = 0 := by decide

def plainActions (earlier : List (Nat × Clock)) : List Action :=
  earlier.map (fun item => .arrival item.1 1 item.2)

 theorem execute_arrival (s : State) (actor count : Nat) (release : Clock)
    (tail : List Action) :
    execute s (.arrival actor count release :: tail) = execute (arrive s actor count release) tail := by
  simp only [execute, step, enabled, ite_true]

 theorem finite_prefix_budget (earlier : List (Nat × Clock)) (s : State)
    (actor count : Nat) (release : Clock) (suffix : List Action)
    (positive : 0 < count) (budget : earlier.length + count ≤ s.barrier.pending)
    (separate : ∀ item ∈ earlier, item.1 ≠ actor) :
    execute s (plainActions earlier ++ .arrival actor count release :: suffix) =
      execute s (.arrival actor count release :: plainActions earlier ++ suffix) := by
  induction earlier generalizing s with
  | nil => rfl
  | cons item rest ih =>
    have two : 1 < s.barrier.pending := by simp only [List.length_cons] at budget; omega
    have hcount : count < s.barrier.pending := by simp only [List.length_cons] at budget; omega
    have hfirst : ¬s.barrier.pending < 1 := by omega
    have pendingAfter : (arrive s item.1 1 item.2).barrier.pending = s.barrier.pending - 1 := by
      simp only [arrive, ite_false, hfirst,
        decrement_noncompleting s.barrier 1 item.2 two]
    have restBudget : rest.length + count ≤ (arrive s item.1 1 item.2).barrier.pending := by
      rw [pendingAfter]
      simp only [List.length_cons] at budget
      omega
    have restSeparate : ∀ entry ∈ rest, entry.1 ≠ actor := by
      intro entry member
      exact separate entry (List.mem_cons_of_mem item member)
    have different : actor ≠ item.1 := Ne.symm (separate item (by simp))
    have accepted : canSwapPlain false s.barrier.pending count = true := by
      simp [canSwapPlain, positive, hcount]
    change execute s (.arrival item.1 1 item.2 ::
      (plainActions rest ++ .arrival actor count release :: suffix)) =
      execute s (.arrival actor count release :: .arrival item.1 1 item.2 ::
        (plainActions rest ++ suffix))
    rw [execute_arrival s item.1 1 item.2 _]
    rw [ih (arrive s item.1 1 item.2) restBudget restSeparate]
    rw [← execute_arrival s item.1 1 item.2 _]
    exact (plain_swap_preserves_execution s actor item.1 count release item.2
      (plainActions rest ++ suffix) different accepted).symm

end MemoryWrite.BarrierReduction
