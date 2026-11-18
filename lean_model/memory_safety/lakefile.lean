/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Lake
open Lake DSL
private def aeneasPath : System.FilePath := run_io do
  let path ← IO.FS.readFile (__dir__ / ".aeneas-path")
  return System.FilePath.mk path.trimAscii.toString

require aeneas from aeneasPath
package ptxv_memory_safety_joint where
  moreLeanArgs := #["-j4"]
lean_lib MemoryWrite where
  globs := #[.submodules `MemoryWrite]
lean_lib MemorySafety where
  globs := #[.submodules `MemorySafety]
@[default_target]
lean_lib MemorySafetyExecution where
  globs := #[.submodules `MemorySafetyExecution]
@[default_target]
lean_lib ValueFlow where
  globs := #[.submodules `ValueFlow]
@[default_target]
lean_lib MemorySafetyIR where
  globs := #[.submodules `MemorySafetyIR]

lean_lib Frontend where
  globs := #[.submodules `Frontend]
@[default_target]
lean_lib FrontendProgram where
  globs := #[.submodules `FrontendProgram]

@[default_target]
lean_lib ValueFlowDependencies where
  globs := #[.submodules `ValueFlowDependencies]
