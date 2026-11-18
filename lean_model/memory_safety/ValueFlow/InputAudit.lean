/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import ValueFlow.SourceDependencyApply

#print axioms ValueFlow.evaluate_inputs
#print axioms ValueFlow.Source.source_evaluate
#print axioms ValueFlow.Source.source_dependency_transfer
#print axioms ValueFlow.Source.source_transfer_success
#print axioms ValueFlow.Source.source_evaluate_inputs
#print axioms ValueFlow.Source.source_dependency_store_loop
#print axioms ValueFlow.Source.source_dependency_store
#print axioms ValueFlow.Source.source_dependency_read_loop
#print axioms ValueFlow.Source.source_dependency_read_memory
#print axioms ValueFlow.Source.source_loaded_memory_metadata
#print axioms ValueFlow.Source.dependencies_model_virtual_set
#print axioms ValueFlow.Source.dependencies_model_predicate_set
#print axioms ValueFlow.Source.source_apply_transfer_virtual
#print axioms ValueFlow.Source.source_apply_transfer_predicate
#print axioms ValueFlow.Source.source_apply_transfer_none
#print axioms ValueFlow.Source.source_apply_transfer_store_success
#print axioms ValueFlow.Source.source_transfer_load_computation
