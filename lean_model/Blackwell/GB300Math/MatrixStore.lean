/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import GB300Math.MatrixLoad

namespace GB300Math.MatrixStore

def sourceByte (shape transpose offset : Nat) : Nat :=
  let matrix := offset/128
  let q := offset%128
  if shape = 16 then
    let lane := q/32+4*(q%8)
    let component := q%32/16+2*(q%16/8)
    lane*16+matrix*4+component
  else if transpose = 1 then
    let row := q/16
    let col := q%16/2
    let lane := col*4+row/2
    lane*16+matrix*4+2*(row%2)+q%2
  else (q/4)*16+matrix*4+q%4

def word (shape transpose matrices : Nat) (words : Array Nat) (cell : Nat) : Nat :=
  if cell >= matrices*32 then 0xdeadbeef
  else (List.range 4).foldl (fun acc b =>
    acc + MatrixLoad.byte words (sourceByte shape transpose (cell*4+b))*2^(8*b)) 0

def targetByte (shape transpose lane component : Nat) : Nat :=
  if shape = 16 then
    lane%4*32+lane/4+(component%2)*16+(component/2)*8
  else if transpose = 1 then
    (2*(lane%4)+component/2)*16+(lane/4)*2+component%2
  else lane*4+component

theorem byte_layout_inverse : ∀ q : Fin 128,
    let source := sourceByte 16 1 q
    targetByte 16 1 (source/16) (source%4) = q := by decide +kernel

theorem half_transpose_inverse : ∀ q : Fin 128,
    let source := sourceByte 8 1 q
    targetByte 8 1 (source/16) (source%4) = q := by decide +kernel

theorem half_row_inverse : ∀ q : Fin 128,
    let source := sourceByte 8 0 q
    targetByte 8 0 (source/16) (source%4) = q := by decide +kernel

theorem byte_layout_in_bounds : ∀ q : Fin 128, sourceByte 16 1 q < 512 := by decide +kernel

theorem half_transpose_in_bounds : ∀ q : Fin 128, sourceByte 8 1 q < 512 := by decide +kernel

#print axioms byte_layout_inverse
#print axioms half_transpose_inverse
#print axioms half_row_inverse
#print axioms byte_layout_in_bounds
#print axioms half_transpose_in_bounds
end GB300Math.MatrixStore
