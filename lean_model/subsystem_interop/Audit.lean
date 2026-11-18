/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Interop
import Lean.Util.CollectAxioms

open Lean Elab Command

run_cmd do
  let environment ← getEnv
  let allowed : List Name := [``propext, ``Classical.choice, ``Quot.sound]
  let mut checked : Nat := 0
  for index in [:environment.header.moduleNames.size] do
    if environment.header.moduleNames[index]!.toString.startsWith "Interop" then
      for declarationInfo in environment.header.moduleData[index]!.constants do
        let declarationName := declarationInfo.name
        match declarationInfo with
        | .thmInfo _ =>
          let axioms ← collectAxioms declarationName
          for axiomName in axioms do
            unless axiomName ∈ allowed do
              throwError "Unexpected assumption {axiomName} in {declarationName}"
          checked := checked + 1
          logInfo m!"audit {declarationName}: {axioms}"
        | .axiomInfo _ => throwError "Project assumption {declarationName} is not permitted"
        | _ => pure ()
  unless checked > 0 do throwError "No project theorems were audited"
  logInfo m!"Audited {checked} project theorems"
