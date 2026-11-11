/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Std

namespace GB300Math.TensorDecompression

def expand (width : Nat) (x : BitVec 128) : BitVec 128 :=
  (List.range 16).foldl (fun out j =>
    out ||| (((x >>> (width*j)) &&& BitVec.ofNat 128 (2^width-1)) <<< (8*j+6-width))) 0

def compact (width : Nat) (x : BitVec 128) : BitVec 128 :=
  (List.range 16).foldl (fun out j =>
    out ||| (((x >>> (8*j+6-width)) &&& BitVec.ofNat 128 (2^width-1)) <<< (width*j))) 0

def sourceChunk (shape row col : Nat) : Option Nat :=
  if row ≥ 128 ∨ col ≥ 8 then none else
  match shape with
  | 0 => some (4*row+512*(col/4))
  | 1 => if row%32 = 0 then some (4*(row/32)+512*(col/4)) else none
  | 2 => if col < 4 then some (4*row) else none
  | 3 => if col < 4 then some (4*(row%64)) else none
  | 4 => if col < 4 then some (4*(row%32+32*(row/64))) else none
  | 5 => if col < 4 then some (4*(row%32)) else none
  | _ => none

def word (shape width : Nat) (input : Array Nat) (before row col : Nat) : Nat :=
  match sourceChunk shape row col with
  | none => before
  | some base =>
    if width = 8 then input[base+col%4]!
    else
      let chunk := (List.range 4).foldl (fun x j => x + input[base+j]! * 2^(32*j)) 0
      (expand width (BitVec.ofNat 128 chunk)).toNat / 2^(32*(col%4)) % 2^32

end GB300Math.TensorDecompression
