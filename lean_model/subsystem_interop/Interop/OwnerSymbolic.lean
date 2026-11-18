/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.OwnerRefinement
import MemoryWrite.SymbolicStateRefinement

open Aeneas Aeneas.Std PtxvIr

set_option autoImplicit false

namespace RegisterOwner.Symbolic

open MemoryWrite

def semanticFile : File → MemoryWrite.Symbolic.RegisterFile
  | .V => .virtual
  | .R => .lane
  | .UR => .uniform
  | .P => .predicate
  | .UP => .uniformPredicate

def resolvedKey (thread : Nat) (owners : List Nat) (register : Reg) :
    MemoryWrite.Symbolic.RegisterKey :=
  ⟨(canonicalKey thread owners register).owner, semanticFile register.file, register.idx.val, 0⟩

def resolvedResult (register : Reg) (result : U32 × U8 × U16) :
    MemoryWrite.Symbolic.RegisterKey :=
  ⟨result.1.val, semanticFile register.file, result.2.2.val, 0⟩

theorem resolved_result_exact (thread : Nat) (owners : List Nat) (register : Reg)
    (result : U32 × U8 × U16) (source : keyView result = canonicalKey thread owners register) :
    resolvedResult register result = resolvedKey thread owners register := by
  have owner := congrArg Key.owner source
  have index := congrArg Key.index source
  simp only [keyView, canonicalKey] at owner index
  simp [resolvedResult, resolvedKey, canonicalKey, owner, index]

theorem resolved_key_alias_iff (left right : Nat) (owners : List Nat) (a b : Reg) :
    resolvedKey left owners a = resolvedKey right owners b ↔
      canonicalKey left owners a = canonicalKey right owners b := by
  rcases a with ⟨af, ai⟩
  rcases b with ⟨bf, bi⟩
  cases af <;> cases bf <;>
    simp [resolvedKey, canonicalKey, semanticFile, shared, fileTag,
      MemoryWrite.Symbolic.RegisterKey.mk.injEq, Key.mk.injEq]

def resolvedCopy {Input : Type} (sem : Semantics Input)
    (state : MemoryWrite.Symbolic.SourceState Input) (thread : Usize) (owners : Slice U32)
    (source destination predicate : Reg) : Result (MemoryWrite.Symbolic.SourceState Input) := do
  let sourceKey ← register_owner.canonical_key thread owners source
  let destinationKey ← register_owner.canonical_key thread owners destination
  let predicateKey ← register_owner.canonical_key thread owners predicate
  MemoryWrite.Symbolic.sourceCopy sem state (resolvedResult source sourceKey)
    (resolvedResult destination destinationKey) (resolvedResult predicate predicateKey)

theorem source_resolved_copy_preserves_symbolic_state {Input : Type} (sem : Semantics Input)
    (state : MemoryWrite.Symbolic.SourceState Input)
    (sound : (MemoryWrite.Symbolic.sourceView state).Sound sem)
    (thread : Usize) (owners : Slice U32) (source destination predicate : Reg) :
    resolvedCopy sem state thread owners source destination predicate
      ⦃ result => (MemoryWrite.Symbolic.sourceView result).Sound sem ⦄ := by
  unfold resolvedCopy
  step with rust_canonical_key thread owners source as ⟨sourceKey, sourceExact⟩
  step with rust_canonical_key thread owners destination as ⟨destinationKey, destinationExact⟩
  step with rust_canonical_key thread owners predicate as ⟨predicateKey, predicateExact⟩
  exact MemoryWrite.Symbolic.rust_prepare_copy_preserves_state sem state sound _ _ _

end RegisterOwner.Symbolic
