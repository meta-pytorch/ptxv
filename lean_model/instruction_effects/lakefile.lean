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

package ptxv_instruction_effects

@[default_target]
lean_lib Instruction where
  moreLeanArgs := #["-j4"]
