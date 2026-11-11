import MufuRcp32Classification
namespace GB300Math.Rcp64Bounds
open FloatLib.Floats.Formats.BinaryInterchange

theorem normal_high_fields (e : Fin 2046) (f : Fin (2^20)) :
    let w := ((1+e.val)*2^20+f.val)*2^32
    let x : Model FloatFormat.binary64 := Model.ofNatBits w
    x.expField = 1+e.val ∧ x.fracField = f.val*2^32 ∧ x.signBit = false := by
  have he := e.isLt
  have hf := f.isLt
  have hw : ((1+e.val)*2^20+f.val)*2^32 < 2^64 := by omega
  have hn : ((1+e.val)*2^20+f.val)*2^32 < 2^63 := by omega
  constructor
  · change (((BitVec.ofNat 64 (((1+e.val)*2^20+f.val)*2^32) >>> 52) &&& (2047 : BitVec 64)).toNat) = _
    simp only [BitVec.toNat_and, BitVec.toNat_ushiftRight, BitVec.toNat_ofNat]
    rw [Nat.mod_eq_of_lt hw, Nat.shiftRight_eq_div_pow]
    change (((((1+e.val)*2^20+f.val)*2^32) / 2^52) &&& (2^11-1)) = _
    rw [Nat.and_two_pow_sub_one_eq_mod]
    omega
  · constructor
    · change ((BitVec.ofNat 64 (((1+e.val)*2^20+f.val)*2^32) &&& (4503599627370495 : BitVec 64)).toNat) = _
      simp only [BitVec.toNat_and, BitVec.toNat_ofNat]
      rw [Nat.mod_eq_of_lt hw]
      change ((((1+e.val)*2^20+f.val)*2^32) &&& (2^52-1)) = _
      rw [Nat.and_two_pow_sub_one_eq_mod]
      omega
    · rw [Model.signBit_eq_msb]
      change (BitVec.ofNat 64 (((1+e.val)*2^20+f.val)*2^32)).msb = false
      rw [BitVec.msb_eq_decide]
      simp only [BitVec.toNat_ofNat,Nat.mod_eq_of_lt hw]
      simpa using hn

theorem decode_normal_high_raw (e : Fin 2046) (f : Fin (2^20)) :
    (Model.ofNatBits (((1+e.val)*2^20+f.val)*2^32) : Model FloatFormat.binary64).toRat? =
      some (((2^52+f.val*2^32 : Nat) : Rat)*(2 : Rat)^((e.val : Int)-1074)) := by
  have fields := normal_high_fields e f
  dsimp only at fields
  have hieee : FloatFormat.binary64.isIEEE = true := by decide
  have hen : (1+e.val == FloatFormat.binary64.expAllOnesNat) = false := by
    change (1+e.val == 2047) = false
    simp only [beq_eq_false_iff_ne]
    have h := e.isLt
    omega
  have hez : (1+e.val == 0) = false := by simp
  simp only [Model.toRat?,Model.toDyadic?,hieee,ite_true,
    Model.ieeeToDyadic?,Model.IEEE.isNaN,Model.IEEE.isInf,
    fields.1,fields.2.1,fields.2.2,hen,hez]
  norm_num [FloatLib.Numerics.Dyadic.toRat,FloatLib.Numerics.Dyadic.signedSignificand,
    Model.pow2,FloatFormat.binary64,FloatFormat.ieeeNormalMantissaExpOffset,FloatFormat.bias]
  ring_nf
  simp

theorem decode_normal_high (e : Fin 2046) (f : Fin (2^20)) :
    (Model.ofNatBits (((1+e.val)*2^20+f.val)*2^32) : Model FloatFormat.binary64).toRat? =
      some (((2^20+f.val : Nat) : Rat)*(2 : Rat)^((e.val : Int)-1042)) := by
  rw [decode_normal_high_raw]
  apply congrArg some
  have hp : (2 : Rat)^32*(2 : Rat)^((e.val : Int)-1074) = (2 : Rat)^((e.val : Int)-1042) := by
    have hh : (32 : Int)+((e.val : Int)-1074) = (e.val : Int)-1042 := by ring
    rw [show (2 : Rat)^32 = (2 : Rat)^(32 : Int) by norm_num,
      ← zpow_add₀ (by norm_num : (2 : Rat) ≠ 0),hh]
  calc
    _ = ((2^20+f.val : Nat) : Rat)*((2 : Rat)^32*(2 : Rat)^((e.val : Int)-1074)) := by
      norm_num
      ring
    _ = _ := by rw [hp]

#print axioms normal_high_fields
#print axioms decode_normal_high_raw
#print axioms decode_normal_high
end GB300Math.Rcp64Bounds
