/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Approx

namespace GB300Math.DivisionPtx

open Approx

set_option maxRecDepth 100000
set_option maxHeartbeats 0

theorem full_ftz_large_equal : evaluate "div_full" true 0x7f000000 0x7f000000 =
    some 0x3f800000 := by decide +kernel

theorem approximate_ftz_large_equal : evaluate "div" true 0x7f000000 0x7f000000 =
    some 0 := by decide +kernel

theorem full_ftz_negative_large_equal : evaluate "div_full" true 0xff000000 0x7f000000 =
    some 0xbf800000 := by decide +kernel

theorem full_ftz_negative_zero : evaluate "div_full" true 0x80000000 0x7f000000 =
    some 0x80000000 := by decide +kernel

theorem full_ftz_infinity_over_infinity : evaluate "div_full" true 0x7f800000 0x7f800000 =
    some 0x7fffffff := by decide +kernel

theorem flush_before_subnormal_rounding : mulFtz 0x00800000 0x3f7fffff = 0 := by
  decide +kernel

theorem rounding_then_flushing_differs : flush (mul 0x00800000 0x3f7fffff) =
    0x00800000 := by decide +kernel

theorem approximate_ftz_boundary : evaluate "div" true 0x3f7fffff 0x7e800000 =
    some 0 := by decide +kernel

theorem approximate_gradual_boundary : evaluate "div" false 0x3f7fffff 0x7e800000 =
    some 0x00800000 := by decide +kernel

#print axioms full_ftz_large_equal
#print axioms approximate_ftz_large_equal
#print axioms full_ftz_negative_large_equal
#print axioms full_ftz_negative_zero
#print axioms full_ftz_infinity_over_infinity
#print axioms flush_before_subnormal_rounding
#print axioms rounding_then_flushing_differs
#print axioms approximate_ftz_boundary
#print axioms approximate_gradual_boundary

end GB300Math.DivisionPtx
