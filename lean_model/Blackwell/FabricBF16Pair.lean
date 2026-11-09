import FabricBF16V2Bounds

namespace FabricBF16Pair
open FabricBF16Block FabricBF16V2 GB300Math.TensorAlignmentLoss

def exponent (word : Nat) : Nat := word%32768/128
def magnitude (word : Nat) : Nat := unit (word%32768)
def value (word : Nat) : Int := if word≥32768 then -(magnitude word:Int) else magnitude word

theorem magnitude_formula (word : Nat) : magnitude word =
    (word%128 + if exponent word=0 then 0 else 128)*2^(exponent word-1) := by
  simp [magnitude,unit,exponent]

theorem magnitude_upper (word : Nat) : magnitude word < 2^(exponent word+7) := by
  rw [magnitude_formula]
  have hf : word%128<128 := Nat.mod_lt _ (by decide)
  by_cases h : exponent word=0
  · simpa [h] using hf
  · rw [ite_eq_right h]
    calc
      _ < 256*2^(exponent word-1) := Nat.mul_lt_mul_of_pos_right (by omega) (Nat.two_pow_pos _)
      _ = 2^(exponent word+7) := by
        rw [show 256=2^8 by decide,←Nat.pow_add]
        congr 1
        omega

theorem magnitude_lower (word : Nat) (h : 0<exponent word) :
    2^(exponent word+6)≤magnitude word := by
  rw [magnitude_formula,ite_eq_right (show exponent word≠0 by omega)]
  calc
    _ = 128*2^(exponent word-1) := by
      rw [show 128=2^7 by decide,←Nat.pow_add]
      congr 1
      omega
    _ ≤ _ := Nat.mul_le_mul_right _ (by omega)

theorem magnitude_dvd (word shift : Nat) (h : shift≤exponent word-1) :
    2^shift ∣ magnitude word := by
  unfold magnitude unit
  exact dvd_mul_of_dvd_right (Nat.pow_dvd_pow 2 h) _

theorem value_abs (word : Nat) : (value word).natAbs=magnitude word := by
  unfold value
  split <;> simp

theorem truncate_abs_le (x : Int) (step : Nat) : (truncateGrid x step).natAbs≤x.natAbs := by
  unfold truncateGrid
  split <;> (simp only [Int.natAbs_neg,Int.natAbs_natCast]; exact Nat.div_mul_le_self x.natAbs step)

theorem truncate_exact (x : Int) (step : Nat) (h : step ∣ x.natAbs) :
    truncateGrid x step=x := by
  have hd := Nat.div_mul_cancel h
  unfold truncateGrid
  rw [hd]
  cases x with
  | ofNat n => simp
  | negSucc n => simp;omega

theorem units_value (w : Nat) (hw : w<65536) (hf : w%32768<32640) :
    units w=some (value w) := by
  simpa [value,magnitude] using FabricBF16Decode.units_canonical w hw hf

theorem shift_bounds (k : Nat) (h : 2≤k) :
    16*(k/16)-26≤k-2 ∧ 16*(k/16)-17≤k-2 := by omega

theorem aligned_ordered (a b : Nat) (horder : exponent b≤exponent a)
    (nonzero : truncateGrid (value a) (2^(16*(exponent a/16)-26))+
      truncateGrid (value b) (2^(16*(exponent a/16)-26))≠0) :
    2^(16*(exponent a/16)-17) ≤
      (truncateGrid (value a) (2^(16*(exponent a/16)-26))+
       truncateGrid (value b) (2^(16*(exponent a/16)-26))).natAbs := by
  let k:=exponent a
  let f:=16*(k/16)-26
  let c:=16*(k/16)-17
  change 2^c≤(truncateGrid (value a) (2^f)+truncateGrid (value b) (2^f)).natAbs
  by_cases hk : k<2
  · have hc : c=0 := by dsimp [c];omega
    rw [hc]
    exact Int.natAbs_pos.mpr nonzero
  · have hk2 : 2≤k := by omega
    obtain ⟨hf,hc⟩ := shift_bounds k hk2
    have hfa : f≤exponent a-1 := by dsimp [f,k];omega
    have ha : truncateGrid (value a) (2^f)=value a := by
      apply truncate_exact
      rw [value_abs]
      exact magnitude_dvd a f hfa
    rw [ha] at nonzero ⊢
    by_cases hb : k-1≤exponent b
    · have hfb : f≤exponent b-1 := by dsimp [f];omega
      have htb : truncateGrid (value b) (2^f)=value b := by
        apply truncate_exact
        rw [value_abs]
        exact magnitude_dvd b f hfb
      rw [htb] at nonzero ⊢
      have hda : (2^c:Int) ∣ value a := by
        apply Int.natCast_dvd.mpr
        rw [value_abs]
        exact magnitude_dvd a c (by dsimp [c,k];omega)
      have hdb : (2^c:Int) ∣ value b := by
        apply Int.natCast_dvd.mpr
        rw [value_abs]
        exact magnitude_dvd b c (by dsimp [c];omega)
      exact Nat.le_of_dvd (Int.natAbs_pos.mpr nonzero) (Int.natCast_dvd.mp (dvd_add hda hdb))
    · have hsmall := magnitude_upper b
      have hpow : 2^(exponent b+7)≤2^(k+5) := Nat.pow_le_pow_right (by decide) (by omega)
      have hbig := magnitude_lower a (by dsimp [k] at hk2;omega)
      have htr := truncate_abs_le (value b) (2^f)
      rw [value_abs] at htr
      have htriangle := Int.natAbs_add_le
        (value a+truncateGrid (value b) (2^f)) (-truncateGrid (value b) (2^f))
      simp only [Int.add_neg_cancel_right,Int.natAbs_neg,value_abs] at htriangle
      have hp : 2^c≤2^(k+5) := Nat.pow_le_pow_right (by decide) (by dsimp [c];omega)
      have htwice : 2^(k+6)=2*2^(k+5) := by rw [show k+6=k+5+1 by omega,Nat.pow_succ];omega
      change 2^(k+6)≤magnitude a at hbig
      rw [htwice] at hbig
      omega

theorem pair_nonzero (a b : Nat)
    (h : aligned [a,b] [value a,value b]≠0) :
    2^(coarseShift [a,b])≤(aligned [a,b] [value a,value b]).natAbs := by
  rcases Nat.le_total (exponent b) (exponent a) with hab | hba
  · have ho : b%32768/128≤a%32768/128 := hab
    simpa [aligned,latticeShift,coarseShift,maxExponent,truncatedSum,exponent,Nat.max_eq_left ho] using
      aligned_ordered a b hab (by simpa [aligned,latticeShift,maxExponent,truncatedSum,exponent,Nat.max_eq_left ho] using h)
  · have ho : a%32768/128≤b%32768/128 := hba
    simpa [aligned,latticeShift,coarseShift,maxExponent,truncatedSum,exponent,Nat.max_eq_right ho,Int.add_comm] using
      aligned_ordered b a hba (by simpa [aligned,latticeShift,maxExponent,truncatedSum,exponent,Nat.max_eq_right ho,Nat.max_eq_left ho,Int.add_comm] using h)

theorem zeros_padding (a b : Nat) :
    coarseShift [a,b,0,0]=coarseShift [a,b] ∧
    aligned [a,b,0,0] [value a,value b,0,0]=aligned [a,b] [value a,value b] := by
  simp [coarseShift,aligned,latticeShift,maxExponent,truncatedSum,truncateGrid]

theorem zero_output (states : Array Accum) (lane : Nat) (sr : Bool)
    (h : states[lane]! =.finite 0) : FabricBF16Block.output states lane sr=0 := by
  cases sr <;> simp [FabricBF16Block.output,h,FabricBF16V2Bounds.round_zero,unit]

theorem two_active_output (a b : Nat) (states : Array Accum) (lane : Nat) (sr : Bool)
    (ha : a<65536 ∧ a%32768<32640) (hb : b<65536 ∧ b%32768<32640)
    (hs : states[lane]! =accumulate [a,b,0,0]) :
    FabricBF16V2.output states lane (coarseShift [a,b,0,0]) sr =
      FabricBF16Block.output states lane sr := by
  have hf : ∀w∈[a,b,0,0],w%32768<32640 := by intro w hw;simp only [List.mem_cons,List.not_mem_nil,or_false] at hw;rcases hw with rfl|rfl|rfl|rfl <;> simp_all
  have hc := FabricBF16RoundBounds.accumulate_finite [a,b,0,0] hf
  have hva := units_value a ha.1 ha.2
  have hvb := units_value b hb.1 hb.2
  have hv : states[lane]! =.finite (aligned [a,b] [value a,value b]) := by
    rw [hs,hc]
    simp only [List.map_cons,List.map_nil,hva,hvb,Option.getD_some]
    change Accum.finite (aligned [a,b,0,0] [value a,value b,0,0])= _
    rw [(zeros_padding a b).2]
  rw [(zeros_padding a b).1]
  by_cases hz : aligned [a,b] [value a,value b]=0
  · have ho := zero_output states lane sr (by simpa [hz] using hv)
    simp [FabricBF16V2.output,hv,hz,ho]
  · have hn := pair_nonzero a b hz
    simp [FabricBF16V2.output,hv,show ¬(aligned [a,b] [value a,value b]).natAbs<2^(coarseShift [a,b]) by omega]

#print axioms pair_nonzero
#print axioms zeros_padding
#print axioms zero_output
#print axioms two_active_output
#print axioms shift_bounds
#print axioms aligned_ordered
#print axioms magnitude_upper
#print axioms magnitude_lower
#print axioms magnitude_dvd
#print axioms value_abs
#print axioms truncate_abs_le
#print axioms truncate_exact
#print axioms units_value
end FabricBF16Pair
