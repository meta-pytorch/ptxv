/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.PredicateKnowledge
import Instruction.OwnerRefinement

open Aeneas Aeneas.Std Result PtxvIr

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace Instruction.OwnerTemporal

open RegisterOwner PredicateRead

def privateKnowledge (known : Reg → Option Bool) (register : Reg) : Option Bool :=
  if shared register.file then none else known register

def PrivateAgree (before after : Reg → Bool) : Prop :=
  ∀ register, shared register.file = false → before register = after register

theorem private_knowledge_survives_shared_interference
    (known : Reg → Option Bool) (before after : Reg → Bool)
    (sound : KnowledgeSound known before) (unchanged : PrivateAgree before after) :
    KnowledgeSound (privateKnowledge known) after := by
  intro register value found
  by_cases isShared : shared register.file
  · simp [privateKnowledge, isShared] at found
  · have same := unchanged register (Bool.eq_false_iff.mpr isShared)
    exact (sound register value (by simpa [privateKnowledge, isShared] using found)).trans same

theorem source_thread_private (register : Reg) :
    register_owner.is_thread_private register = .ok (!shared register.file) := by
  rcases register with ⟨file, index⟩
  cases file <;> rfl

theorem private_key_keeps_thread (thread : Nat) (owners : List Nat) (register : Reg)
    (isPrivate : shared register.file = false) :
    canonicalKey thread owners register =
      ⟨thread % 2 ^ 32, fileTag register.file, register.idx.val⟩ := by
  simp [canonicalKey, isPrivate]

theorem private_keys_do_not_alias_peer (left right : Nat) (owners : List Nat)
    (register : Reg) (isPrivate : shared register.file = false)
    (different : left % 2 ^ 32 ≠ right % 2 ^ 32) :
    canonicalKey left owners register ≠ canonicalKey right owners register := by
  rw [private_key_keeps_thread left owners register isPrivate,
    private_key_keeps_thread right owners register isPrivate]
  intro equal
  exact different (congrArg Key.owner equal)

@[step]
theorem rust_thread_private (register : Reg) :
    register_owner.is_thread_private register ⦃ result => result = !shared register.file ⦄ := by
  rw [source_thread_private, WP.spec_ok]

end Instruction.OwnerTemporal
