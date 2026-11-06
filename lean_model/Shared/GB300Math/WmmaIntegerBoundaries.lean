import Std
import GB300Math.WmmaArithmetic

namespace GB300Math.WmmaIntegerBoundaries
open WmmaArithmetic

theorem clamp_range (x : Int) :
    -(2^31) ≤ max (-(2^31)) (min (2^31-1) x) ∧
    max (-(2^31)) (min (2^31-1) x) ≤ 2^31-1 := by omega

theorem wrap_range (x : Int) : 0 ≤ x % 2^32 ∧ x % 2^32 < 2^32 := by omega

theorem positive_saturates :
    word 1 1 1 "s8" "s32" "satfinite" #[127,127,0x7fffffff] 0 = 0x7fffffff := by decide +kernel

theorem positive_wraps :
    word 1 1 1 "s8" "s32" "wrap" #[127,127,0x7fffffff] 0 = 0x80003f00 := by decide +kernel

theorem negative_saturates :
    word 1 1 1 "s8" "s32" "satfinite" #[0xffffff80,127,0x80000000] 0 = 0x80000000 := by decide +kernel

theorem negative_wraps :
    word 1 1 1 "s8" "s32" "wrap" #[0xffffff80,127,0x80000000] 0 = 0x7fffc080 := by decide +kernel

end GB300Math.WmmaIntegerBoundaries
