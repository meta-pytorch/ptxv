import MufuRcp64Output
namespace GB300Math.Rcp64Bounds
set_option maxRecDepth 8000

theorem high_word_witness :
    Fp64Seed.evaluate false 0x3fffe8bf00000000 = 0x3fe00ba800000000 := by decide +kernel

theorem full_word_witness :
    Fp64Seed.evaluate false 0x3ff04884ffffffff = 0x3fef717e00000000 := by decide +kernel

theorem high_word_tighter_bound_false :
    ¬ |((1051560 : ℝ)/2^21)-1/((2^20+1042623)/2^20)| /
      |1/((2^20+1042623)/2^20)| ≤ 1/(2^21 : ℝ) := by norm_num

theorem ignored_low_word_changes_ideal_error :
    ¬ |((2060670 : ℝ)/2^21)-1/(((2^20+18564)*2^32+4294967295)/2^52)| /
      |1/(((2^20+18564)*2^32+4294967295)/2^52)| ≤ 1/(2^20 : ℝ) := by norm_num

#print axioms high_word_witness
#print axioms full_word_witness
#print axioms high_word_tighter_bound_false
#print axioms ignored_low_word_changes_ideal_error

open FloatLib.Floats.Formats.BinaryInterchange

theorem high_word_witness_decode :
    (Model.ofNatBits 0x3fffe8bf00000000 : Model FloatFormat.binary64).toRat? =
      some ((2^20+1042623 : Rat)/2^20) ∧
    (Model.ofNatBits 0x3fe00ba800000000 : Model FloatFormat.binary64).toRat? =
      some ((1051560 : Rat)/2^21) := by decide +kernel

theorem full_word_witness_decode :
    (Model.ofNatBits 0x3ff04884ffffffff : Model FloatFormat.binary64).toRat? =
      some (((2^20+18564 : Rat)*2^32+4294967295)/2^52) ∧
    (Model.ofNatBits 0x3fef717e00000000 : Model FloatFormat.binary64).toRat? =
      some ((2060670 : Rat)/2^21) := by decide +kernel

#print axioms high_word_witness_decode
#print axioms full_word_witness_decode
end GB300Math.Rcp64Bounds
