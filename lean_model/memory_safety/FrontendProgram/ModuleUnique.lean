/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import FrontendProgram.CodeInventory
import FrontendProgram.CodeUnique
import FrontendProgram.DeclarationUniqueness
import FrontendProgram.GlobalUnique

open Aeneas Aeneas.Std PtxvMemorySafety
set_option autoImplicit false
set_option debug.skipKernelTC false
set_option Elab.async false
set_option maxHeartbeats 12000000
set_option maxRecDepth 8192

namespace PtxFrontendProgram

theorem scope_fields_unique (left right : Scope)
    (declarations : left.declarations=right.declarations) (globals : left.globals=right.globals)
    (labels : left.labels=right.labels) (virtualCount : left.virtual_count=right.virtual_count)
    (predicateCount : left.predicate_count=right.predicate_count) : left=right := by
  cases left
  cases right
  cases declarations
  cases globals
  cases labels
  cases virtualCount
  cases predicateCount
  rfl

theorem program_fields_unique (left right : strict_memory_frontend_parse.SourceProgram)
    (entry : left.entry=right.entry) (scope : left.scope=right.scope) (code : left.code=right.code) : left=right := by
  cases left
  cases right
  cases entry
  cases scope
  cases code
  rfl

theorem module_meaning_unique (source entry : List U8)
    (left right : strict_memory_frontend_parse.SourceProgram)
    (first : ModuleMeaning source entry left) (second : ModuleMeaning source entry right) : left=right := by
  obtain ⟨leftTokens,leftGlobalsEnd,leftDeclarationsStart,leftInstructionsStart,leftFinish,
    leftLexed,leftHeader,leftGlobals,leftEntry,leftDeclarations,leftCode,leftConsumed,leftNonempty,
    leftLayout,leftSeparate,leftGlobalsSeparate,leftLabelsSeparate,leftSymbols⟩:=first
  obtain ⟨rightTokens,rightGlobalsEnd,rightDeclarationsStart,rightInstructionsStart,rightFinish,
    rightLexed,rightHeader,rightGlobals,rightEntry,rightDeclarations,rightCode,rightConsumed,rightNonempty,
    rightLayout,rightSeparate,rightGlobalsSeparate,rightLabelsSeparate,rightSymbols⟩:=second
  have firstModule : ModuleMeaning source entry left :=
    ⟨leftTokens,leftGlobalsEnd,leftDeclarationsStart,leftInstructionsStart,leftFinish,
      leftLexed,leftHeader,leftGlobals,leftEntry,leftDeclarations,leftCode,leftConsumed,leftNonempty,
      leftLayout,leftSeparate,leftGlobalsSeparate,leftLabelsSeparate,leftSymbols⟩
  have secondModule : ModuleMeaning source entry right :=
    ⟨rightTokens,rightGlobalsEnd,rightDeclarationsStart,rightInstructionsStart,rightFinish,
      rightLexed,rightHeader,rightGlobals,rightEntry,rightDeclarations,rightCode,rightConsumed,rightNonempty,
      rightLayout,rightSeparate,rightGlobalsSeparate,rightLabelsSeparate,rightSymbols⟩
  have sameTokens:=source_tokens_lexed_unique source leftTokens rightTokens leftLexed rightLexed
  subst rightTokens
  obtain ⟨sameGlobals,sameGlobalsEnd⟩:=globals_before_entry_unique source entry leftTokens 6 leftGlobalsEnd rightGlobalsEnd
    leftDeclarationsStart rightDeclarationsStart left.scope.globals.val right.scope.globals.val left.entry right.entry
    leftGlobals rightGlobals leftEntry rightEntry
  subst rightGlobalsEnd
  obtain ⟨sameDeclarationsStart,sameEntry⟩:=entry_syntax_unique source entry leftTokens leftGlobalsEnd
    leftDeclarationsStart rightDeclarationsStart left.entry right.entry leftEntry rightEntry
  subst rightDeclarationsStart
  obtain ⟨sameInstructionsStart,sameDeclarations,sameVirtualCount,samePredicateCount⟩:=declarations_then_code_unique
    source leftTokens left.scope right.scope leftDeclarationsStart leftInstructionsStart rightInstructionsStart leftFinish rightFinish
    left.code.val right.code.val left.scope.labels.val right.scope.labels.val leftDeclarations rightDeclarations
    leftCode rightCode leftLayout rightLayout
  subst rightInstructionsStart
  obtain ⟨_,sameLabels,_⟩:=code_inventory_unique source entry left right firstModule secondModule leftTokens
    leftInstructionsStart 0 leftFinish rightFinish left.code.val right.code.val left.scope.labels.val right.scope.labels.val leftCode rightCode
  have sameScope:=scope_fields_unique left.scope right.scope sameDeclarations
    ((alloc.vec.Vec.eq_iff _ _).mpr sameGlobals) ((alloc.vec.Vec.eq_iff _ _).mpr sameLabels) sameVirtualCount samePredicateCount
  rw [←sameScope] at rightCode
  obtain ⟨_,_,sameCode⟩:=code_syntax_unique source entry left firstModule leftTokens leftInstructionsStart 0 leftFinish rightFinish
    left.code.val right.code.val left.scope.labels.val left.scope.labels.val leftCode rightCode
  exact program_fields_unique left right sameEntry sameScope ((alloc.vec.Vec.eq_iff _ _).mpr sameCode)

theorem parsed_module_unique (source entry : Slice U8) (parsed other : strict_memory_frontend_parse.SourceProgram)
    (computed : strict_memory_frontend_parse.parse_source source entry=.ok (some parsed))
    (meaning : ModuleMeaning source.val entry.val other) : parsed=other := by
  exact module_meaning_unique source.val entry.val parsed other (source_parse_source source entry parsed computed) meaning

end PtxFrontendProgram
