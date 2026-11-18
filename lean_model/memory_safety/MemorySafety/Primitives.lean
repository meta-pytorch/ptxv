/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

module
public import Aeneas
@[expose] public section

namespace MemorySafety.Primitives

/-- The existing UTF-8 String abstraction maps Rust's empty constructor to the empty Lean string. -/
def stringNew : String := ""

open Aeneas Aeneas.Std

def stringBytes (value : String) : Result (Slice U8) :=
  let bytes := value.toUTF8.data.toList.map (fun byte => U8.ofNat byte.toNat (by
    have bound := byte.toNat_lt_size
    change byte.toNat < 256 at bound
    change byte.toNat ≤ 255
    omega))
  if bound : bytes.length ≤ Usize.max then .ok (Slice.from bytes bound)
  else .fail .integerOverflow

end MemorySafety.Primitives
