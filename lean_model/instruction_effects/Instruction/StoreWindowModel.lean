/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Instruction.Model

open Aeneas Aeneas.Std PtxvIr
set_option autoImplicit false
namespace Instruction

structure ByteWindow where
  source : Key
  first : Nat
  count : Nat

def byteWindow (entry : Reg × U32 × U32) : ByteWindow :=
  ⟨key entry.1, entry.2.1.val, entry.2.2.val⟩

def sourceWindows (first : Reg) (count stride total : Nat) : List ByteWindow :=
  (List.range count).map fun index =>
    ⟨⟨first.file, first.idx.val + index⟩, index * stride, min (total - index * stride) stride⟩

@[simp] theorem source_windows_length (first : Reg) (count stride total : Nat) :
    (sourceWindows first count stride total).length = count := by simp [sourceWindows]

theorem source_windows_succ (first : Reg) (count stride total : Nat) :
    sourceWindows first (count + 1) stride total = sourceWindows first count stride total ++
      [⟨⟨first.file, first.idx.val + count⟩, count * stride, min (total - count * stride) stride⟩] := by
  simp [sourceWindows, List.range_succ, List.map_append]

theorem source_window_keys (first : Reg) (count stride total : Nat) :
    (sourceWindows first count stride total).map ByteWindow.source = consecutive first count := by
  simp [sourceWindows, consecutive, List.map_map]

def layoutCount (file : File) (elements bytes : U8) (exchange : Bool) : Nat :=
  if exchange then scalarSlots file elements.val else storeSlots file bytes.val elements.val false

def layoutStride (file : File) (elements bytes : U8) (exchange : Bool) : Nat :=
  if file = .V then
    if exchange then bytes.val else if elements.val ≥ 2 then bytes.val / elements.val
    else if bytes.val > 8 then 4 else bytes.val
  else 4

def layoutWindows (first : Reg) (elements bytes : U8) (exchange : Bool) : List ByteWindow :=
  sourceWindows first (layoutCount first.file elements bytes exchange)
    (layoutStride first.file elements bytes exchange) bytes.val

def storeDescriptor (operation : Op) : Option (Reg × U8 × U8 × Bool) :=
  match operation with
  | .St space bytes _ (.Reg first elements) _ =>
      match space with
      | .Param => none
      | _ => some (first, elements, bytes, false)
  | .Atom .Exch space bytes _ (.Reg first elements) _ _ _ _ =>
      match space with
      | .Param => none
      | _ => some (first, elements, bytes, true)
  | _ => none

def repeatedWindows (first : Reg) (count total : Nat) : List ByteWindow :=
  (List.range count).map fun index =>
    ⟨key first, index * 8, min (total - index * 8) 8⟩

theorem repeated_windows_succ (first : Reg) (count total : Nat) :
    repeatedWindows first (count + 1) total = repeatedWindows first count total ++
      [⟨key first, count * 8, min (total - count * 8) 8⟩] := by
  simp [repeatedWindows, List.range_succ, List.map_append]

def storeWindows (front : Front) (operation : Op) : List ByteWindow :=
  match storeDescriptor operation with
  | none => []
  | some (first, elements, bytes, exchange) =>
      if first.file = .V then
        match front with
        | .Sass =>
            if !exchange && elements.val == 1 && bytes.val > 8 then
              repeatedWindows first ((bytes.val + 7) / 8) bytes.val
            else []
        | .Ptx => layoutWindows first elements bytes exchange
      else layoutWindows first elements bytes exchange

def windowsBounded (windows : List ByteWindow) : Prop :=
  Bounded (windows.map ByteWindow.source)

theorem source_window_byte_in_payload (first : Reg) (count stride total : Nat)
    (window : ByteWindow) (hwindow : window ∈ sourceWindows first count stride total)
    (offset : Nat) (hbyte : offset < window.count) : window.first + offset < total := by
  simp only [sourceWindows, List.mem_map, List.mem_range] at hwindow
  rcases hwindow with ⟨index, hindex, rfl⟩
  dsimp only at hbyte ⊢
  have hremaining : offset < total - index * stride :=
    lt_of_lt_of_le hbyte (Nat.min_le_left _ _)
  omega

end Instruction
