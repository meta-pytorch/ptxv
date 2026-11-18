/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemorySafety.Code.Funs

open Aeneas Aeneas.Std PtxvMemorySafety.ptxv_memory_write

namespace ValueFlowDependencies.Source

abbrev Metadata := symbolic.Metadata
abbrev Registers := value_flow_dependencies.Registers
abbrev State := value_flow_dependencies.State

deriving instance DecidableEq for symbolic.Metadata,
  value_flow_dependencies.Registers, value_flow_dependencies.State

end ValueFlowDependencies.Source
