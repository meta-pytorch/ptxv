/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafetyExecution.Native.Core

open Aeneas Aeneas.Std Result PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false
set_option debug.skipKernelTC false

namespace MemorySafetyExecution.Native

theorem load_value_access_fields (state : CoreState) (left right : execution_core.ResolvedAccess) (signed : Bool)
    (allocation : left.allocation=right.allocation) (offset : left.offset=right.offset) (width : left.width=right.width) :
    execution_core.load_value state left signed=execution_core.load_value state right signed := by
  unfold execution_core.load_value
  rw [allocation,offset,width]

theorem store_value_access_fields (state : CoreState) (left right : execution_core.ResolvedAccess) (value : U64)
    (allocation : left.allocation=right.allocation) (offset : left.offset=right.offset) (width : left.width=right.width) :
    execution_core.store_value state left value=execution_core.store_value state right value := by
  unfold execution_core.store_value
  rw [allocation,offset,width]

end MemorySafetyExecution.Native
