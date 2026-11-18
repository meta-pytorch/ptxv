/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.SourceAssembly
import FrontendProgram.GlobalLoops

open Aeneas Aeneas.Std Result PtxvMemorySafety
set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
namespace PtxFrontendProgram

theorem source_parse_source (source entry : Slice U8) (program : strict_memory_frontend_parse.SourceProgram)
    (computed : strict_memory_frontend_parse.parse_source source entry=.ok (some program)) :
    ModuleMeaning source.val entry.val program :=
  source_parse_source_of_globals source_parse_globals source entry program computed

theorem source_parse_source_global_names (source entry : Slice U8) (program : strict_memory_frontend_parse.SourceProgram)
    (computed : strict_memory_frontend_parse.parse_source source entry=.ok (some program)) :
    ∀ (i j : Nat) (left right : Global),program.scope.globals.val[i]?=some left → program.scope.globals.val[j]?=some right →
      nameBytes source.val left.«name»=nameBytes source.val right.«name» → i=j :=
  module_global_names_injective source.val entry.val program (source_parse_source source entry program computed)

end PtxFrontendProgram
