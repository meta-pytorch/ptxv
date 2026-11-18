/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.Code.Funs

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false

namespace MemoryWrite.ActionEffects

def resourceTag : effects.ResourceSpace → U8
  | .Shared => 0#u8
  | .Global => 1#u8
  | .Tmem => 2#u8
  | .TmemAllocator => 3#u8
  | .CtaExit => 4#u8
  | .BulkGroup => 5#u8
  | .MbarPhase => 6#u8
  | .MbarArrivalPhase => 7#u8
  | .WarpPc => 8#u8
  | .GridGate => 9#u8
  | .StrongShared => 10#u8
  | .StrongGlobal => 11#u8
  | .MbarPhaseKey => 12#u8
  | .ClusterBarrier => 13#u8
  | .NamedBarrier => 14#u8
  | .MbarInitialization => 15#u8
  | .MbarPendingCount => 16#u8
  | .UniformRegisters => 17#u8
  | .MbarMembership => 18#u8
  | .ConvergenceBarrier => 19#u8
  | .L1Line => 20#u8
  | .L1Sm => 21#u8
  | .TensorCoreACollector => 24#u8
  | .MbarOverArrival => 25#u8
  | .NicQueue => 26#u8
  | .CtaAll => 98#u8

theorem source_resource_tag (space : effects.ResourceSpace) :
    effects.resource_tag space = .ok (resourceTag space) := by
  cases space <;> rfl

theorem resource_tag_injective (left right : effects.ResourceSpace)
    (same : resourceTag left = resourceTag right) : left = right := by
  cases left <;> cases right <;> simp_all [resourceTag]

theorem source_resource_range (resource : effects.Resource) :
    effects.resource_range resource =
      .ok (resourceTag resource.space, resource.owner, resource.start, resource.end) := by
  simp [effects.resource_range, source_resource_tag]

theorem resource_ranges_disjoint (left right : effects.Resource)
    (different : left.space ≠ right.space) :
    resourceTag left.space ≠ resourceTag right.space := by
  exact fun same => different (resource_tag_injective _ _ same)

theorem source_tensor_core_collector_range (cta : U32) :
    effects.resource_range
      { space := .TensorCoreACollector, owner := cta, start := 0#u64, «end» := 1#u64 } =
      .ok (24#u8, cta, 0#u64, 1#u64) := by
  simp [source_resource_range, resourceTag]

theorem source_tensor_core_collector_visibility :
    effects.included .Full .All .TensorCoreACollector = .ok true ∧
    effects.included .SyncOnly .All .TensorCoreACollector = .ok true ∧
    effects.included .Full .SyncOnly .TensorCoreACollector = .ok false ∧
    effects.included .SyncOnly .SyncOnly .TensorCoreACollector = .ok true := by
  simp [effects.included, effects.FoldMode.Insts.CoreCmpPartialEqFoldMode.eq,
    effects.Visibility.Insts.CoreCmpPartialEqVisibility.eq,
    effects.FoldMode.read_discriminant, effects.Visibility.read_discriminant]

end MemoryWrite.ActionEffects
