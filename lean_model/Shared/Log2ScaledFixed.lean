import Log2Fixed
namespace GB300Math.Log2ScaledFixed
open FloatLib.Floats.Formats.BinaryInterchange
set_option maxRecDepth 12000
set_option maxHeartbeats 5000000
theorem fixed_error (mag scale : Nat) (hs : 1≤ scale ∧ scale≤ 127)
    (hm : 0< mag ∧ mag<(scale+1)*2^36) :
    ∃ y : Rat,
      (Model.ofNatBits (Mufu.fixedToWord mag 36) : Model FloatFormat.binary32).toRat? = some y ∧
      0 ≤ (mag:Rat)/2^36-y ∧ (mag:Rat)/2^36-y ≤(scale:Rat)/2^23 := by
  have hmax : mag< 2^43 := by omega
  have hn : mag ≠ 0 := by omega
  have ha := Nat.log2_self_le hn
  have hb := Nat.lt_log2_self (n:=mag)
  have hhigh : mag.log2 ≤ 42 := by have h := (Nat.log2_lt hn).mpr hmax;omega
  generalize hlead : mag.log2 = lead at *
  have hexp : ((lead:Int)-36+127).toNat = lead+91 := by omega
  let mant := if lead ≥ 23 then mag/2^(lead-23) else mag*2^(23-lead)
  have hmnt : 2^23 ≤ mant ∧ mant < 2^24 := by
    dsimp [mant]
    interval_cases lead <;> norm_num at * <;> omega
  refine ⟨(mant:Rat)*(2:Rat)^((lead:Int)-59),?_,?_⟩
  · unfold Mufu.fixedToWord
    simp only [hlead]
    rw [hexp]
    exact Log2Fixed.decode lead mant hhigh hmnt
  have hdiv0 := Nat.div_mul_le_self mag (2^(lead-23))
  have hdiv1 : mag < mag/2^(lead-23)*2^(lead-23)+2^(lead-23) := by
    have hpos : 0 < 2^(lead-23) := by positivity
    have hmod := Nat.mod_lt mag hpos
    have hid := Nat.mod_add_div mag (2^(lead-23))
    nlinarith only [hmod,hid]
  have hd0 : (((mag/2^(lead-23):Nat):Rat)*(2^(lead-23):Nat)) ≤ mag := by exact_mod_cast hdiv0
  have hd1 : (mag:Rat) < ((mag/2^(lead-23):Nat):Rat)*(2^(lead-23):Nat)+(2^(lead-23):Nat) := by exact_mod_cast hdiv1
  have hstep : 2^(lead-36)≤ scale := by
    by_cases h : 36≤ lead
    · have hid : (2:Nat)^lead=2^(lead-36)*2^36 := by rw [←pow_add];congr 1;omega
      rw [hid] at ha
      omega
    · have hzero : lead-36=0 := by omega
      simpa [hzero] using hs.1
  have hstepr : ((2^(lead-36):Nat):Rat)≤ scale := by exact_mod_cast hstep
  dsimp [mant]
  interval_cases lead <;> norm_num at hd0 hd1 ⊢ <;>
    norm_num only [Nat.reduceSub,Nat.reducePow,Nat.cast_ofNat] at hstepr <;> constructor <;> linarith

#print axioms fixed_error
end GB300Math.Log2ScaledFixed
