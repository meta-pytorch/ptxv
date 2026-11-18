/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Aeneas

set_option autoImplicit false
set_option debug.skipKernelTC false

namespace PtxFrontend

@[ext]
structure Dimensions where
  x : Nat
  y : Nat
  z : Nat
  deriving DecidableEq

def Dimensions.size (shape : Dimensions) : Nat := shape.x * shape.y * shape.z

def Dimensions.positive (shape : Dimensions) : Prop :=
  0 < shape.x ∧ 0 < shape.y ∧ 0 < shape.z

def Dimensions.contains (shape point : Dimensions) : Prop :=
  point.x < shape.x ∧ point.y < shape.y ∧ point.z < shape.z

def coordinates (shape : Dimensions) (index : Nat) : Dimensions :=
  ⟨index % shape.x, index / shape.x % shape.y, index / shape.x / shape.y⟩

def linearIndex (shape point : Dimensions) : Nat :=
  point.x + shape.x * (point.y + shape.y * point.z)

theorem coordinate_linear_index (shape : Dimensions) (index : Nat) :
    linearIndex shape (coordinates shape index) = index := by
  have first := Nat.mod_add_div index shape.x
  have second := Nat.mod_add_div (index / shape.x) shape.y
  simp only [linearIndex, coordinates]
  nlinarith

theorem linear_coordinates (shape point : Dimensions) (inside : shape.contains point) :
    coordinates shape (linearIndex shape point) = point := by
  rcases shape with ⟨x, y, z⟩
  rcases point with ⟨a, b, c⟩
  rcases inside with ⟨aBound, bBound, cBound⟩
  dsimp [Dimensions.contains] at aBound bBound cBound
  have positiveX : 0 < x := by omega
  have positiveY : 0 < y := by omega
  have divX : (a + x * (b + y * c)) / x = b + y * c := by
    rw [Nat.add_mul_div_left _ _ positiveX, Nat.div_eq_of_lt aBound]
    simp
  have modX : (a + x * (b + y * c)) % x = a := by
    rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt aBound]
  have divY : (b + y * c) / y = c := by
    rw [Nat.add_mul_div_left _ _ positiveY, Nat.div_eq_of_lt bBound]
    simp
  have modY : (b + y * c) % y = b := by
    rw [Nat.add_mul_mod_self_left, Nat.mod_eq_of_lt bBound]
  apply Dimensions.ext <;> simp only [coordinates, linearIndex, divX, modX, divY, modY]

theorem coordinates_in_bounds (shape : Dimensions) (index : Nat)
    (positive : shape.positive) (bound : index < shape.size) :
    shape.contains (coordinates shape index) := by
  rcases positive with ⟨positiveX, positiveY, positiveZ⟩
  refine ⟨Nat.mod_lt _ positiveX, Nat.mod_lt _ positiveY, ?_⟩
  change index / shape.x / shape.y < shape.z
  rw [Nat.div_div_eq_div_mul]
  exact (Nat.div_lt_iff_lt_mul (by positivity)).2 (by simpa [Dimensions.size, Nat.mul_comm,
    Nat.mul_left_comm, Nat.mul_assoc] using bound)

theorem linear_index_in_bounds (shape point : Dimensions) (inside : shape.contains point) :
    linearIndex shape point < shape.size := by
  rcases inside with ⟨xBound, yBound, zBound⟩
  have middle : point.y + shape.y * point.z < shape.y * shape.z := by
    calc
      point.y + shape.y * point.z < shape.y + shape.y * point.z := Nat.add_lt_add_right yBound _
      _ = shape.y * (point.z + 1) := by ring
      _ ≤ shape.y * shape.z := Nat.mul_le_mul_left _ (by omega)
  dsimp [linearIndex, Dimensions.size]
  calc
    point.x + shape.x * (point.y + shape.y * point.z) <
        shape.x + shape.x * (point.y + shape.y * point.z) := Nat.add_lt_add_right xBound _
    _ = shape.x * (point.y + shape.y * point.z + 1) := by ring
    _ ≤ shape.x * (shape.y * shape.z) := Nat.mul_le_mul_left _ (by omega)
    _ = shape.x * shape.y * shape.z := by ring

structure ActorIds where
  tid : Dimensions
  ctaid : Dimensions
  ntid : Dimensions
  nctaid : Dimensions
  lane : Nat
  deriving DecidableEq

def actorAt (grid block : Dimensions) (index : Nat) : ActorIds :=
  ⟨coordinates block (index % block.size), coordinates grid (index / block.size),
    block, grid, index % block.size % 32⟩

def actorInventory (grid block : Dimensions) : List ActorIds :=
  (List.range (grid.size * block.size)).map (actorAt grid block)

theorem inventory_size (grid block : Dimensions) :
    (actorInventory grid block).length = grid.size * block.size := by
  simp [actorInventory]

theorem inventory_actor (grid block : Dimensions) (index : Nat)
    (bound : index < grid.size * block.size) :
    (actorInventory grid block)[index]? = some (actorAt grid block index) := by
  simp [actorInventory, bound]

theorem inventory_every_thread (grid block ctaid tid : Dimensions)
    (_positiveGrid : grid.positive) (positiveBlock : block.positive)
    (ctaInside : grid.contains ctaid) (tidInside : block.contains tid) :
    ∃ index, index < grid.size * block.size ∧
      (actorAt grid block index).ctaid = ctaid ∧ (actorAt grid block index).tid = tid ∧
      (actorAt grid block index).lane = linearIndex block tid % 32 := by
  let cta := linearIndex grid ctaid
  let thread := linearIndex block tid
  have ctaBound : cta < grid.size := linear_index_in_bounds grid ctaid ctaInside
  have threadBound : thread < block.size := linear_index_in_bounds block tid tidInside
  have blockPositive : 0 < block.size := by
    rcases positiveBlock with ⟨x, y, z⟩
    dsimp [Dimensions.size]
    positivity
  let index := cta * block.size + thread
  have quotient : index / block.size = cta := by
    change (cta * block.size + thread) / block.size = cta
    rw [Nat.mul_comm cta block.size, Nat.mul_add_div blockPositive, Nat.div_eq_of_lt threadBound]
    simp
  have remainder : index % block.size = thread := by
    simp [index, Nat.add_mod, Nat.mod_eq_of_lt threadBound]
  refine ⟨index, ?_, ?_, ?_, ?_⟩
  · dsimp [index]
    nlinarith
  · simpa [actorAt, quotient, cta] using linear_coordinates grid ctaid ctaInside
  · simpa [actorAt, remainder, thread] using linear_coordinates block tid tidInside
  · simp [actorAt, remainder, thread]

end PtxFrontend
