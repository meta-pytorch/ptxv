import MufuRcp32Signed
namespace GB300Math.FixedNormal
open FloatLib.Floats.Formats.BinaryInterchange
set_option maxRecDepth 12000
set_option maxHeartbeats 15000000

theorem decode (lead mant : Nat) (scale : Int)
    (he : 1 ≤ (lead:Int)-scale+127 ∧ (lead:Int)-scale+127 ≤ 254)
    (hm : 2^23 ≤ mant ∧ mant < 2^24) :
    (Model.ofNatBits (((lead:Int)-scale+127).toNat*2^23+mant%2^23) :
      Model FloatFormat.binary32).toRat? = some ((mant:Rat)*(2:Rat)^((lead:Int)-scale-23)) := by
  let e : Fin 254 := ⟨((lead:Int)-scale+126).toNat,by omega⟩
  let f : Fin (2^23) := ⟨mant%2^23,by omega⟩
  have h := MufuRcp32Bounds.decode_normal e f
  have heq : (e.val:Int)-149=(lead:Int)-scale-23 := by dsimp [e];omega
  have hf : 2^23+f.val=mant := by dsimp [f];omega
  have hw : (1+e.val)*2^23+f.val=((lead:Int)-scale+127).toNat*2^23+mant%2^23 := by dsimp [e,f];omega
  rwa [heq,hf,hw] at h

theorem mantissa_range (mag lead : Nat) (hm : 2^lead ≤ mag ∧ mag < 2^(lead+1))
    (hl : lead ≤ 64) :
    let mant := if lead ≥ 23 then mag/2^(lead-23) else mag*2^(23-lead)
    2^23 ≤ mant ∧ mant < 2^24 := by
  dsimp
  interval_cases lead <;> norm_num at * <;> omega

theorem fixed_subunit (mag scale : Nat) (hs : 58 ≤ scale ∧ scale ≤ 65)
    (hm : 0 < mag ∧ mag < 2^scale) :
    ∃ y : Rat,
      (Model.ofNatBits (Mufu.fixedToWord mag scale) : Model FloatFormat.binary32).toRat? = some y ∧
      0 ≤ (mag:Rat)/(2:Rat)^scale-y ∧ (mag:Rat)/(2:Rat)^scale-y ≤ 1/2^24 := by
  have hn : mag ≠ 0 := by omega
  have ha := Nat.log2_self_le hn
  have hb := Nat.lt_log2_self (n:=mag)
  have hhigh : mag.log2 < scale := (Nat.log2_lt hn).mpr hm.2
  generalize hlead : mag.log2 = lead at *
  have hl : lead ≤ 64 := by omega
  let mant := if lead ≥ 23 then mag/2^(lead-23) else mag*2^(23-lead)
  have hmnt : 2^23 ≤ mant ∧ mant < 2^24 := mantissa_range mag lead ⟨ha,hb⟩ hl
  have hexp : 1 ≤ (lead:Int)-(scale:Int)+127 ∧ (lead:Int)-(scale:Int)+127 ≤ 254 := by omega
  refine ⟨(mant:Rat)*(2:Rat)^((lead:Int)-(scale:Int)-23),?_,?_⟩
  · unfold Mufu.fixedToWord
    simp only [hlead]
    exact decode lead mant scale hexp hmnt
  have hdiv0 := Nat.div_mul_le_self mag (2^(lead-23))
  have hdiv1 : mag < mag/2^(lead-23)*2^(lead-23)+2^(lead-23) := by
    have hmod := Nat.mod_lt mag (by positivity : 0 < 2^(lead-23))
    have hid := Nat.mod_add_div mag (2^(lead-23))
    nlinarith only [hmod,hid]
  have hd0 : (((mag/2^(lead-23):Nat):Rat)*(2^(lead-23):Nat)) ≤ mag := by exact_mod_cast hdiv0
  have hd1 : (mag:Rat) < ((mag/2^(lead-23):Nat):Rat)*(2^(lead-23):Nat)+(2^(lead-23):Nat) := by exact_mod_cast hdiv1
  rcases hs with ⟨hs0,hs1⟩
  dsimp [mant]
  interval_cases scale <;> interval_cases lead <;> norm_num at * <;> constructor <;> linarith

#print axioms decode
#print axioms mantissa_range
#print axioms fixed_subunit
end GB300Math.FixedNormal
