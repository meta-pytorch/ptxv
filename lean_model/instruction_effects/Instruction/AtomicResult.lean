/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.OwnerPredicate

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false

namespace AtomicResult

def predicateObserved (front : Front) (predicate : Option Reg) : Bool :=
  match predicate with
  | none => false
  | some register => !RegisterOwner.immutablePredicate front register

def outputObserved (front : Front) (word predicate : Option Reg) : Bool :=
  word.isSome || predicateObserved front predicate

def compareExchangeMemory (operation : AtomOp) : Bool :=
  match operation with
  | .Cas | .CasSpin => true
  | _ => false

def exchangeMemory (operation : AtomOp) : Bool :=
  match operation with
  | .Exch => true
  | _ => false

theorem exchangeMemory_iff (operation : AtomOp) :
    exchangeMemory operation = true ↔ operation = .Exch := by
  cases operation <;> simp [exchangeMemory]

def localPolicy (front : Front) (operation : AtomOp) (word predicate : Option Reg) : Bool :=
  compareExchangeMemory operation || exchangeMemory operation || outputObserved front word predicate


theorem predicateObserved_iff (front : Front) (predicate : Option Reg) :
    predicateObserved front predicate = true ↔
      ∃ register, predicate = some register ∧
        RegisterOwner.immutablePredicate front register = false := by
  cases predicate <;> simp [predicateObserved]

theorem outputObserved_iff (front : Front) (word predicate : Option Reg) :
    outputObserved front word predicate = true ↔
      word.isSome = true ∨
      ∃ register, predicate = some register ∧
        RegisterOwner.immutablePredicate front register = false := by
  simp [outputObserved, predicateObserved_iff]

theorem compareExchangeMemory_iff (operation : AtomOp) :
    compareExchangeMemory operation = true ↔ operation = .Cas ∨ operation = .CasSpin := by
  cases operation <;> simp [compareExchangeMemory]

theorem source_observed (front : Front) (word predicate : Option Reg) :
    atomic_result.observed front word predicate = .ok (outputObserved front word predicate) := by
  cases word <;> cases predicate <;>
    simp [atomic_result.observed, core.option.Option.is_some,
      RegisterOwner.source_normalize_predicate, RegisterOwner.normalizedPredicate,
      outputObserved, predicateObserved]
  all_goals split <;> simp_all

theorem source_local_order_monitored (front : Front) (operation : AtomOp)
    (word predicate : Option Reg) :
    atomic_result.local_order_monitored front operation word predicate =
      .ok (localPolicy front operation word predicate) := by
  cases operation <;>
    simp [atomic_result.local_order_monitored, source_observed,
      localPolicy, compareExchangeMemory, exchangeMemory]

theorem source_unused_CAS_is_monitored (front : Front) (predicate : Option Reg) :
    atomic_result.local_order_monitored front .Cas none predicate = .ok true := by
  simp [source_local_order_monitored, localPolicy, compareExchangeMemory, exchangeMemory]

theorem source_predicate_only_CAS_SPIN_is_monitored (front : Front)
    (predicate : Option Reg) :
    atomic_result.local_order_monitored front .CasSpin none predicate = .ok true := by
  simp [source_local_order_monitored, localPolicy, compareExchangeMemory, exchangeMemory]

theorem source_word_output_is_monitored (front : Front) (operation : AtomOp)
    (word : Reg) (predicate : Option Reg) :
    atomic_result.local_order_monitored front operation (some word) predicate = .ok true := by
  simp [source_local_order_monitored, localPolicy, outputObserved]

theorem source_writable_predicate_is_monitored (front : Front) (operation : AtomOp)
    (word : Option Reg) (predicate : Reg)
    (writable : RegisterOwner.immutablePredicate front predicate = false) :
    atomic_result.local_order_monitored front operation word (some predicate) = .ok true := by
  simp [source_local_order_monitored, localPolicy, outputObserved,
    predicateObserved, writable]


theorem localPolicy_iff (front : Front) (operation : AtomOp) (word predicate : Option Reg) :
    localPolicy front operation word predicate = true ↔
      (operation = .Cas ∨ operation = .CasSpin) ∨ operation = .Exch ∨
      word.isSome = true ∨
      ∃ register, predicate = some register ∧
        RegisterOwner.immutablePredicate front register = false := by
  simp [localPolicy, compareExchangeMemory_iff, exchangeMemory_iff, outputObserved_iff, or_assoc]

theorem source_local_order_monitored_iff (front : Front) (operation : AtomOp)
    (word predicate : Option Reg) :
    atomic_result.local_order_monitored front operation word predicate = .ok true ↔
      (operation = .Cas ∨ operation = .CasSpin) ∨ operation = .Exch ∨
      word.isSome = true ∨
      ∃ register, predicate = some register ∧
        RegisterOwner.immutablePredicate front register = false := by
  rw [source_local_order_monitored]
  simpa using localPolicy_iff front operation word predicate

theorem unobserved_predicate_reads_agree (front : Front) (predicate : Option Reg)
    (absent : predicateObserved front predicate = false) (left right : Reg → Bool) :
    RegisterOwner.predicateRead front left { reg := predicate, neg := false } =
      RegisterOwner.predicateRead front right { reg := predicate, neg := false } := by
  cases predicate with
  | none => rfl
  | some register =>
    have constant : RegisterOwner.immutablePredicate front register = true := by
      simpa [predicateObserved] using absent
    simp [RegisterOwner.predicateRead, constant]


theorem predicateObserved_iff_read_can_vary (front : Front) (predicate : Option Reg) :
    predicateObserved front predicate = true ↔
      ∃ left right : Reg → Bool,
        RegisterOwner.predicateRead front left { reg := predicate, neg := false } ≠
          RegisterOwner.predicateRead front right { reg := predicate, neg := false } := by
  constructor
  · intro present
    obtain ⟨register, same, writable⟩ := (predicateObserved_iff front predicate).mp present
    subst predicate
    refine ⟨fun _ => false, fun _ => true, ?_⟩
    simp [RegisterOwner.predicateRead, writable]
  · rintro ⟨left, right, different⟩
    cases absent : predicateObserved front predicate with
    | false => exact False.elim (different (unobserved_predicate_reads_agree front predicate absent left right))
    | true => rfl

end AtomicResult

namespace AtomicResult
theorem source_unused_EXCH_is_monitored (front : PtxvIr.Front) :
    PtxvIr.atomic_result.local_order_monitored front .Exch none none = .ok true := by
  simp [source_local_order_monitored,localPolicy,compareExchangeMemory,exchangeMemory]
end AtomicResult
