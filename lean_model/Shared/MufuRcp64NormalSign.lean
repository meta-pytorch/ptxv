import MufuRcp64ScaleAccuracy
import MufuRcp32Signed
set_option maxRecDepth 32768
namespace GB300Math.Rcp64Bounds

theorem normal_high_sign_fields (w e f n : Nat)
    (hw : w < 2^31) (he : w/2^20%2048 = e) (hf : w%2^20 = f)
    (elo : 1 ≤ e) (ehi : e < 2047) (hn : Fp64Seed.interpolate false (f*8) = n) :
    Fp64Seed.highWord false (2^31+w) = 2^31+Fp64Seed.highWord false w := by
  have hnw : 2^31+w < 2^32 := by omega
  have hpw : w < 2^32 := by omega
  have hne : (2^31+w)/2^20%2048 = e := by omega
  have hnf : (2^31+w)%2^20 = f := by omega
  have hps : w &&& 0x80000000 = 0 := RcpSigned.sign_zero w hw
  have hns : (2^31+w) &&& 0x80000000 = 2^31 := RcpSigned.sign_set w hw
  unfold Fp64Seed.highWord
  rw [Nat.mod_eq_of_lt hnw,Nat.mod_eq_of_lt hpw]
  dsimp only
  rw [he,hf,hne,hnf,hps,hns]
  have henz : ¬e=0 := by omega
  have heni : ¬e=2047 := by omega
  simp only [heni,henz,Bool.false_eq_true,false_and,ite_false,Nat.zero_mul,Nat.zero_add]
  rw [hn]
  simp only [Nat.zero_or]
  clear hn
  have upper : 2046-(e:Int)-(if n=2^24 then 0 else 1) ≤ 2045 := by split_ifs <;> omega
  generalize hk : 2046-(e:Int)-(if n=2^24 then 0 else 1) = k at *
  by_cases h : k ≤ 0
  · rw [ite_eq_left h,ite_eq_left h,Nat.add_zero]
  · rw [ite_eq_right h,ite_eq_right h]
    have hh : k.toNat ≤ 2045 := by omega
    have hf' : (n%2^23)/8 < 2^20 := by omega
    have hor : k.toNat*2^20 ||| (n%2^23)/8 = k.toNat*2^20+(n%2^23)/8 := by
      simpa only [Nat.mul_comm] using (Nat.two_pow_add_eq_or_of_lt hf' k.toNat).symm
    rw [Nat.or_assoc,hor]
    have bound : k.toNat*2^20+(n%2^23)/8 < 2^31 := by omega
    simpa using (Nat.two_pow_add_eq_or_of_lt bound 1).symm

theorem normal_high_sign (e : Fin 2046) (f : Fin (2^20)) :
    Fp64Seed.highWord false (2^31+(1+e.val)*2^20+f.val) =
      2^31+Fp64Seed.highWord false ((1+e.val)*2^20+f.val) := by
  have he:=e.isLt
  have hf:=f.isLt
  have hinput : 2^31+(1+e.val)*2^20+f.val = 2^31+((1+e.val)*2^20+f.val) := by omega
  rw [hinput]
  exact normal_high_sign_fields ((1+e.val)*2^20+f.val) (1+e.val) f.val _
    (by omega) (by omega) (by omega) (by omega) (by omega) rfl

theorem full_rcp_normal_sign (e : Fin 2046) (f : Fin (2^20)) (low : Fin (2^32)) :
    let w := ((1+e.val)*2^20+f.val)*2^32+low.val
    Fp64Seed.evaluate false (w+2^63) = Fp64Seed.evaluate false w+2^63 := by
  dsimp only
  have hl:=low.isLt
  have hp : (((1+e.val)*2^20+f.val)*2^32+low.val)/2^32 = (1+e.val)*2^20+f.val := by omega
  have addquot (v : Nat) : (v+2^63)/2^32 = v/2^32+2^31 := by
    change (v+2^31*2^32)/2^32 = _
    exact Nat.add_mul_div_right v (2^31) (by decide)
  have hn : (((1+e.val)*2^20+f.val)*2^32+low.val+2^63)/2^32 = 2^31+(1+e.val)*2^20+f.val := by
    rw [addquot,hp]
    omega
  unfold Fp64Seed.evaluate
  rw [hp,hn,normal_high_sign]
  ring

#print axioms normal_high_sign_fields
#print axioms normal_high_sign
#print axioms full_rcp_normal_sign
end GB300Math.Rcp64Bounds
