/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.BarrierBatchModel
import MemoryWrite.Code.Funs

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false

namespace MemoryWrite.BarrierBatchBudget

def actualReadDecision (pending count : U32) : Result Bool := do
  let batch ← barrier.arrival_batch pending
  let batch ← barrier.record_arrival_batch batch
    ⟨.Add 0#i64, pending, count, false, false⟩
  barrier.arrival_batch_writes batch

theorem actual_two_of_four_is_read : actualReadDecision 4#u32 2#u32 = .ok false := by
  simp (config := { decide := true }) only [actualReadDecision, barrier.arrival_batch, barrier.record_arrival_batch,
    barrier.invalidate_arrival_batch, barrier.arrival_batch_writes, U32.checked_sub,
    U32.checked_add, core.num.checked_sub_UScalar, core.num.checked_add_UScalar,
    HSub.hSub, HAdd.hAdd, HDiv.hDiv, UScalar.sub, UScalar.add, UScalar.tryMk,
    UScalar.div, Option.ofResult, lift, Sub.sub, Add.add,
    UScalar.tryMkOpt, UScalar.check_bounds, UScalar.ofNatCore, bind_ok,
    Result.match.ok, Result.ofOption, UScalar.val, ite_true, ite_false, _root_.dite_true, decide_false]

theorem actual_three_of_four_remains_write : actualReadDecision 4#u32 3#u32 = .ok true := by
  simp (config := { decide := true }) only [actualReadDecision, barrier.arrival_batch, barrier.record_arrival_batch,
    barrier.invalidate_arrival_batch, barrier.arrival_batch_writes, U32.checked_sub,
    U32.checked_add, core.num.checked_sub_UScalar, core.num.checked_add_UScalar,
    HSub.hSub, HAdd.hAdd, HDiv.hDiv, UScalar.sub, UScalar.add, UScalar.tryMk,
    UScalar.div, Option.ofResult, lift, Sub.sub, Add.add,
    UScalar.tryMkOpt, UScalar.check_bounds, UScalar.ofNatCore, bind_ok,
    Result.match.ok, Result.ofOption, UScalar.val, ite_true, ite_false, _root_.dite_true, decide_true]

end MemoryWrite.BarrierBatchBudget
