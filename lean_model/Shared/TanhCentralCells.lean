/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import TanhCentralData
namespace GB300Math.TanhCentralCells
set_option maxRecDepth 1000000
set_option maxHeartbeats 0
theorem certificate : ∀ e:Fin 11,∀u:Fin 512,u.val<count e.val → Valid e.val u.val := by decide +kernel
#print axioms certificate
end GB300Math.TanhCentralCells
