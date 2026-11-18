/-
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree.
-/

import Aeneas

set_option autoImplicit false

namespace ValueFlow

def integerSigned (bits value : Nat) : Int := Int.bmod value (2 ^ bits)

end ValueFlow
