/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Lake
open Lake DSL

require ptxv_memory_write from "../memory_write"
require ptxv_instruction_effects from "../instruction_effects"

package ptxv_subsystem_interop

@[default_target]
lean_lib Interop where
  moreLeanArgs := #["-j4"]
