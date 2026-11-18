/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.SourceModel
import ValueFlow.InputTransfer

open Aeneas Aeneas.Std Result
open PtxvMemorySafety.ptxv_memory_write

set_option autoImplicit false

namespace ValueFlow.Source

abbrev DependencyRegisters := value_flow_dependencies.Registers
abbrev Metadata := symbolic.Metadata
abbrev DependencyTransfer := value_flow_dependencies.Transfer

def metadataDependency (values : List Metadata) (index : Nat) : Option Bool := do
  let value ← values[index]?
  if value.defined then some value.tainted else none

def dependenciesModel (registers : DependencyRegisters) : Dependencies :=
  ⟨metadataDependency registers.virtuals.val, metadataDependency registers.predicates.val⟩

def transferTaint : DependencyTransfer → Bool
  | .None => false
  | .Virtual _ metadata | .Predicate _ metadata | .Store metadata => metadata.tainted

def TransferDefined : DependencyTransfer → Prop
  | .None => True
  | .Virtual _ metadata | .Predicate _ metadata | .Store metadata => metadata.defined = true

def TransferMatches : V.Action → DependencyTransfer → Prop
  | .Next, .None | .Jump _, .None | .Halt, .None => True
  | .WriteVirtual destination _, .Virtual other _
  | .Load destination _ _ _ _, .Virtual other _
  | .WritePredicate destination _, .Predicate other _ => destination = other
  | .Store _ _ _, .Store _ => True
  | _, _ => False

end ValueFlow.Source
