/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import AlternateFloatSpec

namespace GB300Math.AlternateFloat

def gridInteger : Value → Option Int
  | .finite negative magnitude => some (signed negative magnitude)
  | _ => none

def pairBound (op dst af bf : String) (a b : Nat) : Bool :=
  match decode af a, decode bf b with
  | some x, some y =>
    let ideal := if op = "add" then add x y else if op = "sub" then add x (negate y) else mul x y
    match gridInteger ideal, decode dst (encode dst false ideal) with
    | some z, some value =>
      match gridInteger value with
      | some rounded =>
        let factor := if dst = "e4m3" then 17 else 9
        let floor := if dst = "e4m3" then 17*2^502 else 9*2^495
        decide (factor * (rounded - z).natAbs ≤ max z.natAbs floor)
      | _ => true
    | _, _ => true
  | _, _ => true

end GB300Math.AlternateFloat
