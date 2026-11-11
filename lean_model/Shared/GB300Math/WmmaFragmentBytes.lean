/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.WmmaFragmentMap
namespace GB300Math.WmmaFragmentMap

universe u v
 theorem fold1 {α : Type u} {β : Type v} (f : β → α → β) (z : β) (a : α) :
    #[a].foldl f z = f z a := by rw [← Array.foldl_toList]; rfl
 theorem fold2 {α : Type u} {β : Type v} (f : β → α → β) (z : β) (a b : α) :
    #[a,b].foldl f z = f (f z a) b := by rw [← Array.foldl_toList]; rfl
 theorem fold4 {α : Type u} {β : Type v} (f : β → α → β) (z : β) (a b c d : α) :
    #[a,b,c,d].foldl f z = f (f (f (f z a) b) c) d := by rw [← Array.foldl_toList]; rfl


def byte (word lane : Nat) : Nat := word / 2^(8*lane) % 256

theorem pack16_byte (a b lane : Nat) (ha : a < 65536) (_hb : b < 65536) (hl : lane < 4) :
    byte (a + b*65536) lane = if lane < 2 then byte a lane else byte b (lane-2) := by
  rcases (show lane=0 ∨ lane=1 ∨ lane=2 ∨ lane=3 by omega) with rfl | rfl | rfl | rfl
  all_goals simp [byte]
  all_goals omega

theorem pack8_byte (a b c d lane : Nat) (ha : a < 256) (hb : b < 256)
    (hc : c < 256) (hd : d < 256) (hl : lane < 4) :
    byte (a + b*256 + c*65536 + d*16777216) lane =
      if lane=0 then a else if lane=1 then b else if lane=2 then c else d := by
  rcases (show lane=0 ∨ lane=1 ∨ lane=2 ∨ lane=3 by omega) with rfl | rfl | rfl | rfl
  all_goals simp [byte]
  all_goals omega

theorem input_field8 (input : Array Nat) (offset : Nat)
    (h : offset % 32 = 0 ∨ offset % 32 = 8 ∨ offset % 32 = 16 ∨ offset % 32 = 24) :
    field input offset 8 = byte input[offset/32]! ((offset%32)/8) := by
  unfold field byte
  rcases h with h | h | h | h
  all_goals simp [h]
  all_goals omega

theorem input_field16 (input : Array Nat) (offset lane : Nat)
    (h : offset % 32 = 0 ∨ offset % 32 = 16) (hl : lane < 2) :
    byte (field input offset 16) lane =
      byte input[offset/32]! ((offset%32)/8+lane) := by
  rcases h with h | h
  all_goals rcases (show lane=0 ∨ lane=1 by omega) with rfl | rfl
  all_goals simp [field, byte, h]
  all_goals omega

theorem input_field32 (input : Array Nat) (offset lane : Nat)
    (h : offset % 32 = 0) (hl : lane < 4) :
    byte (field input offset 32) lane = byte input[offset/32]! lane := by
  rcases (show lane=0 ∨ lane=1 ∨ lane=2 ∨ lane=3 by omega) with rfl | rfl | rfl | rfl
  all_goals simp [field, byte, h]
  all_goals omega

inductive ByteLayout where
  | constant (value : Nat)
  | whole (source : Nat)
  | halves (lo hi : Nat)
  | bytes (a b c d : Nat)
  deriving Inhabited, DecidableEq

deriving instance DecidableEq for Segment
deriving instance DecidableEq for Layout

def ByteLayout.layout : ByteLayout → Layout
  | .constant v => ⟨v,#[]⟩
  | .whole s => ⟨0,#[⟨s,0,32⟩]⟩
  | .halves a b => ⟨0,#[⟨a,0,16⟩,⟨b,16,16⟩]⟩
  | .bytes a b c d => ⟨0,#[⟨a,0,8⟩,⟨b,8,8⟩,⟨c,16,8⟩,⟨d,24,8⟩]⟩

def ByteLayout.aligned : ByteLayout → Prop
  | .constant _ => True
  | .whole s => s % 32 = 0
  | .halves a b => (a % 32 = 0 ∨ a % 32 = 16) ∧ (b % 32 = 0 ∨ b % 32 = 16)
  | .bytes a b c d => a % 8 = 0 ∧ b % 8 = 0 ∧ c % 8 = 0 ∧ d % 8 = 0

instance (l : ByteLayout) : Decidable l.aligned := by cases l <;> unfold ByteLayout.aligned <;> infer_instance

def ByteLayout.source : ByteLayout → Nat → Option Nat
  | .constant _, _ => none
  | .whole s, lane => some (s/8+lane)
  | .halves a b, lane => some (if lane<2 then a/8+lane else b/8+lane-2)
  | .bytes a b c d, lane => some ((if lane=0 then a else if lane=1 then b else if lane=2 then c else d)/8)

def inputByte (input : Array Nat) (offset : Nat) : Nat := byte input[offset/4]! (offset%4)

theorem inputByte_field (input : Array Nat) (offset : Nat) (h : offset % 8 = 0) :
    inputByte input (offset/8) = field input offset 8 := by
  have hr : offset%32 = 0 ∨ offset%32 = 8 ∨ offset%32 = 16 ∨ offset%32 = 24 := by omega
  rw [input_field8 input offset hr]
  unfold inputByte
  congr 2 <;> omega

theorem byte_whole (input : Array Nat) (s lane : Nat) (h : s%32=0) (hl : lane<4) :
    byte (word input (ByteLayout.whole s).layout) lane = inputByte input (s/8+lane) := by
  simp only [ByteLayout.layout, word]
  rewrite [fold1]
  simp only [Nat.pow_zero, Nat.mul_one, Nat.zero_add]
  rewrite [input_field32 input s lane h hl]
  unfold inputByte
  rewrite [show (s/8+lane)/4 = s/32 by omega, show (s/8+lane)%4 = lane by omega]
  rfl

theorem byte_halves (input : Array Nat) (a b lane : Nat)
    (ha : a%32=0 ∨ a%32=16) (hb : b%32=0 ∨ b%32=16) (hl : lane<4) :
    byte (word input (ByteLayout.halves a b).layout) lane =
      inputByte input (if lane<2 then a/8+lane else b/8+lane-2) := by
  simp only [ByteLayout.layout, word]
  rewrite [fold2]
  simp only [Nat.pow_zero, Nat.mul_one, Nat.zero_add]
  rewrite [pack16_byte _ _ _ (field_bound input a 16) (field_bound input b 16) hl]
  split <;> rename_i h
  · rewrite [input_field16 input a lane ha h]
    unfold inputByte
    rewrite [show (a/8+lane)/4 = a/32 by omega, show (a/8+lane)%4 = (a%32)/8+lane by omega]
    rfl
  · rewrite [input_field16 input b (lane-2) hb (by omega)]
    unfold inputByte
    rewrite [show (b/8+lane-2)/4 = b/32 by omega, show (b/8+lane-2)%4 = (b%32)/8+(lane-2) by omega]
    rfl

theorem byte_bytes (input : Array Nat) (a b c d lane : Nat)
    (ha : a%8=0) (hb : b%8=0) (hc : c%8=0) (hd : d%8=0) (hl : lane<4) :
    byte (word input (ByteLayout.bytes a b c d).layout) lane =
      inputByte input ((if lane=0 then a else if lane=1 then b else if lane=2 then c else d)/8) := by
  simp only [ByteLayout.layout, word]
  rewrite [fold4]
  simp only [Nat.pow_zero, Nat.mul_one, Nat.zero_add]
  rewrite [pack8_byte _ _ _ _ _ (field_bound input a 8) (field_bound input b 8) (field_bound input c 8) (field_bound input d 8) hl]
  rcases (show lane=0 ∨ lane=1 ∨ lane=2 ∨ lane=3 by omega) with rfl | rfl | rfl | rfl
  all_goals simp [inputByte_field, ha, hb, hc, hd]

theorem recovered_byte (input : Array Nat) (layout : ByteLayout) (lane offset : Nat)
    (ha : layout.aligned) (hl : lane < 4) (hs : layout.source lane = some offset) :
    byte (word input layout.layout) lane = inputByte input offset := by
  cases layout with
  | constant v => simp [ByteLayout.source] at hs
  | whole s =>
    simp only [ByteLayout.source, Option.some.injEq] at hs
    subst offset
    exact byte_whole input s lane ha hl
  | halves a b =>
    simp only [ByteLayout.source, Option.some.injEq] at hs
    subst offset
    exact byte_halves input a b lane ha.1 ha.2 hl
  | bytes a b c d =>
    simp only [ByteLayout.source, Option.some.injEq] at hs
    subst offset
    exact byte_bytes input a b c d lane ha.1 ha.2.1 ha.2.2.1 ha.2.2.2 hl

#print axioms fold1
#print axioms fold2
#print axioms fold4
#print axioms input_field16
#print axioms input_field32
#print axioms inputByte_field
#print axioms byte_whole
#print axioms byte_halves
#print axioms byte_bytes
#print axioms recovered_byte
#print axioms pack16_byte
#print axioms pack8_byte
#print axioms input_field8
end GB300Math.WmmaFragmentMap
