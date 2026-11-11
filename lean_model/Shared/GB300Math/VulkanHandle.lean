/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.VulkanHandle

def pack (texture sampler : Nat) : Nat := texture + sampler * 1048576
def texture (handle : Nat) : Nat := handle % 1048576
def sampler (handle : Nat) : Nat := handle / 1048576

theorem extract_texture (t s : Nat) (ht : t < 1048576) :
    texture (pack t s) = t := by
  simp [texture, pack, Nat.add_mod, Nat.mod_eq_of_lt ht]

theorem extract_sampler (t s : Nat) (ht : t < 1048576) :
    sampler (pack t s) = s := by
  unfold sampler pack
  omega

theorem reconstruct (handle : Nat) :
    pack (texture handle) (sampler handle) = handle := by
  unfold pack texture sampler
  omega

theorem carrier_fits (t s : Nat) (ht : t < 1048576) (hs : s < 4096) :
    pack t s < 4294967296 := by
  unfold pack
  omega

theorem injective (t u s v : Nat) (ht : t < 1048576) (hu : u < 1048576)
    (h : pack t s = pack u v) : t = u ∧ s = v := by
  have h₁ := congrArg texture h
  have h₂ := congrArg sampler h
  simpa [extract_texture t s ht, extract_texture u v hu,
    extract_sampler t s ht, extract_sampler u v hu] using And.intro h₁ h₂

theorem discarded_sampler (t s : Nat) (ht : t < 1048576) (hs : s ≠ 0) :
    sampler (pack t s) ≠ 0 := by
  simpa [extract_sampler t s ht] using hs

def pack32 (texture sampler : Nat) : Nat := texture + sampler * 4294967296
def texture32 (handle : Nat) : Nat := handle % 4294967296
def sampler32 (handle : Nat) : Nat := handle / 4294967296

theorem extract_texture32 (t s : Nat) (ht : t < 4294967296) :
    texture32 (pack32 t s) = t := by
  simp [texture32, pack32, Nat.add_mod, Nat.mod_eq_of_lt ht]

theorem extract_sampler32 (t s : Nat) (ht : t < 4294967296) :
    sampler32 (pack32 t s) = s := by
  unfold sampler32 pack32
  omega

theorem reconstruct32 (handle : Nat) :
    pack32 (texture32 handle) (sampler32 handle) = handle := by
  unfold pack32 texture32 sampler32
  omega

theorem carrier32_fits (t s : Nat) (ht : t < 4294967296) (hs : s < 4294967296) :
    pack32 t s < 18446744073709551616 := by
  unfold pack32
  omega

theorem injective32 (t u s v : Nat) (ht : t < 4294967296) (hu : u < 4294967296)
    (h : pack32 t s = pack32 u v) : t = u ∧ s = v := by
  have h₁ := congrArg texture32 h
  have h₂ := congrArg sampler32 h
  simpa [extract_texture32 t s ht, extract_texture32 u v hu,
    extract_sampler32 t s ht, extract_sampler32 u v hu] using And.intro h₁ h₂

theorem layouts_differ : sampler (pack32 8 1) = 4096 ∧ sampler32 (pack32 8 1) = 1 := by
  decide +kernel

end GB300Math.VulkanHandle
