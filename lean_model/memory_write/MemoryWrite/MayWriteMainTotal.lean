/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import MemoryWrite.MayWriteMainProgress
import MemoryWrite.MayWriteMainLoop

open Aeneas Aeneas.Std Result PtxvMemoryWrite

set_option autoImplicit false
set_option maxHeartbeats 3000000

namespace MemoryWrite.MayWrite

@[step]
theorem rust_main_loop_total (graph : cache_graph.MayWriteGraph) (bit : U32) (words : Usize)
    (sets : cache_graph.WorkSets) (work : alloc.vec.Vec Usize) (pending : Usize)
    (bounded : GraphBounded graph) (wordCountValue : words.val = wordCount graph.register_count.val)
    (initialStorage : StorageInvariant sets work pending graph.reach.val.length words.val)
    (initialSettled : WorkInvariant (graphView graph bit) (workView sets work pending))
    (space : rank graph.reach.val.length (words.val * 64) (workSolution sets) pending.val < Usize.max) :
    cache_graph.may_write_sets_loop0 graph.reach graph.succ graph.writes bit sets words work pending
      ⦃ result => MainPost graph bit words (workSolution sets) result ⦄ := by
  obtain ⟨result, executed, _⟩ := MemoryWrite.spec_returns _ _
    (rust_main_loop_returns graph bit words sets work pending bounded wordCountValue initialStorage space)
  simp only [executed, WP.spec_ok]
  exact successful_main_loop graph bit words sets work pending bounded wordCountValue initialStorage
    initialSettled result executed

end MemoryWrite.MayWrite
