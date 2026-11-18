/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.InputState
import MemorySafetyExecution.Native.ByteEffects
import ValueFlow.SourceDependencyMemory

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option maxHeartbeats 4000000
set_option Elab.async false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

open ValueFlow.Source

theorem input_memory_store (metadata afterMetadata : MetadataVector)
    (left right leftAfter rightAfter : AllocationMemory) (offset count : Nat) (stored : Metadata)
    (related : MemoryInputsRelated metadata left right)
    (leftFrame : ByteFrame left leftAfter offset count)
    (rightFrame : ByteFrame right rightAfter offset count)
    (shadowFrame : MetadataFrame metadata afterMetadata offset count stored)
    (defined : stored.defined = true)
    (payload : stored.tainted = false → ∀ index, offset ≤ index → index < offset + count →
      leftAfter.bytes.val[index]? = rightAfter.bytes.val[index]?) :
    MemoryInputsRelated afterMetadata leftAfter rightAfter := by
  have definitions : leftAfter.defined = rightAfter.defined := by
    apply alloc.vec.Vec.ext
    apply List.ext_getElem?
    intro index
    by_cases lower : offset ≤ index
    · by_cases upper : index < offset + count
      · rw [leftFrame.2.2.2.2 index lower upper, rightFrame.2.2.2.2 index lower upper]
      · rw [(leftFrame.2.2.2.1 index (Or.inr (by omega))).2,
          (rightFrame.2.2.2.1 index (Or.inr (by omega))).2, related.definitions]
    · rw [(leftFrame.2.2.2.1 index (Or.inl (by omega))).2,
        (rightFrame.2.2.2.1 index (Or.inl (by omega))).2, related.definitions]
  constructor
  · rw [leftFrame.2.1, rightFrame.2.1]
    exact related.bytesLength
  · exact definitions
  · rw [leftFrame.1, rightFrame.1, related.live]
  · have shadowLength := shadowFrame.1
    have memoryLength := leftFrame.2.1
    have beforeLength := related.metadataLength
    omega
  · have byteLength := leftFrame.2.1
    have definedLength := leftFrame.2.2.1
    have beforeShape := related.storageShape
    omega
  · intro index entry found
    by_cases lower : offset ≤ index
    · by_cases upper : index < offset + count
      · have entryEq : entry = stored := by
          simpa only [shadowFrame.2.2 index lower upper, Option.some.injEq] using found.symm
        subst entry
        exact ⟨by rw [defined]; exact leftFrame.2.2.2.2 index lower upper,
          fun _ clean => payload clean index lower upper⟩
      · have beyond : offset + count ≤ index := by omega
        have priorFound : metadata.val[index]? = some entry := by
          simpa only [shadowFrame.2.1 index (Or.inr beyond)] using found
        have prior := related.bytes index entry priorFound
        exact ⟨by rw [(leftFrame.2.2.2.1 index (Or.inr beyond)).2]; exact prior.1,
          fun entryDefined clean => by
            rw [(leftFrame.2.2.2.1 index (Or.inr beyond)).1,
              (rightFrame.2.2.2.1 index (Or.inr beyond)).1]
            exact prior.2 entryDefined clean⟩
    · have below : index < offset := by omega
      have priorFound : metadata.val[index]? = some entry := by
        simpa only [shadowFrame.2.1 index (Or.inl below)] using found
      have prior := related.bytes index entry priorFound
      exact ⟨by rw [(leftFrame.2.2.2.1 index (Or.inl below)).2]; exact prior.1,
        fun entryDefined clean => by
          rw [(leftFrame.2.2.2.1 index (Or.inl below)).1,
            (rightFrame.2.2.2.1 index (Or.inl below)).1]
          exact prior.2 entryDefined clean⟩

theorem input_memory_loaded_equal (metadata : MetadataVector) (left right : AllocationMemory)
    (related : MemoryInputsRelated metadata left right) (offset count : Nat)
    (defined : ∀ index entry, offset ≤ index → index < offset + count →
      metadata.val[index]? = some entry → entry.defined = true ∧ entry.tainted = false)
    (fits : offset + count ≤ metadata.val.length) :
    ∀ index, index < count → left.bytes.val[offset + index]? = right.bytes.val[offset + index]? := by
  intro index inside
  have shadowInside : offset + index < metadata.val.length := by omega
  let entry := metadata.val[offset + index]
  have found : metadata.val[offset + index]? = some entry := List.getElem?_eq_getElem shadowInside
  obtain ⟨entryDefined, untainted⟩ := defined (offset + index) entry (by omega) (by omega) found
  exact (related.bytes (offset + index) entry found).2 entryDefined untainted

end MemorySafetyExecution.Native
