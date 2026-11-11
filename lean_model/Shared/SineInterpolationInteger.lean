import SqrtBounds
namespace GB300Math.SineInterpolationInteger
set_option maxRecDepth 100000
set_option maxHeartbeats 3000000
abbrev RangeValid (a b c : Int) : Prop :=
  0 ≤ a ∧ a ≤ 2^25 ∧ 0 ≤ b ∧ b ≤ 51474 ∧ -1265 ≤ c ∧ c ≤ 0 ∧
  0 ≤ a*2^31+b*131071*2^18+c*131071^2
set_option maxHeartbeats 0 in
theorem ranges_valid : ∀ segment : Fin 64,
    let (a,b,c) := MufuTables.sin.coefficients[segment.val]!
    RangeValid a b c := by decide +kernel

theorem floor_join (a q : Int) : (a+q/128)/4096 = (a*128+q)/524288 := by omega

theorem polynomial_nonnegative (a b c k : Int) (hr : RangeValid a b c)
    (hk : 0 ≤ k ∧ k ≤ 131071) : 0 ≤ a*2^31+b*k*2^18+c*k*k := by
  rcases hr with ⟨ha,ha',hb,hb',hc,hc',hend⟩
  have hconc := mul_nonpos_of_nonpos_of_nonneg hc' (mul_nonneg hk.1 (by linarith : 0 ≤ 131071-k))
  have hleft := mul_nonneg (show 0 ≤ a*2^31 by positivity) (show 0 ≤ 131071-k by linarith)
  have hright := mul_nonneg hend hk.1
  nlinarith only [hconc,hleft,hright]

theorem interpolation_facts (index : Nat) (hi : index < 2^23) :
    let (a,b,c) := MufuTables.sin.coefficients[index/131072]!
    let k : Int := (index%131072 : Nat)
    let n : Int := a*2^31+b*k*2^18+c*k*k
    let z := Mufu.interpolate MufuTables.sin index
    0 ≤ z ∧ z < 2^38 ∧ -(2^19: Int) ≤ z*2^19-n ∧ z*2^19-n ≤ 1265*4718593 := by
  let segment : Fin 64 := ⟨index/131072,by omega⟩
  rcases ht : MufuTables.sin.coefficients[segment.val]! with ⟨a,b,c⟩
  have hr := ranges_valid segment
  rw [ht] at hr
  have hc : -1265 ≤ c ∧ c ≤ 0 := ⟨hr.2.2.2.2.1,hr.2.2.2.2.2.1⟩
  let k := index%131072
  let square := Fp64Seed.squareTruncated k 19
  let drop := k*k-square
  have hs : square ≤ k*k := by unfold square Fp64Seed.squareTruncated; exact Nat.sub_le _ _
  have hd : 0 ≤ (drop:Int) ∧ (drop:Int) ≤ 4718593 := by
    exact ⟨Int.natCast_nonneg _,by exact_mod_cast SqrtBounds.square_drop_bound k⟩
  have hsquare : (square:Int) = (k:Int)*(k:Int)-(drop:Int) := by
    have h : drop+square=k*k := Nat.sub_add_cancel hs
    have hc : (drop:Int)+(square:Int)=(k:Int)*(k:Int) := by exact_mod_cast h
    omega
  have hk : 0 ≤ (k:Int) ∧ (k:Int) ≤ 131071 := by dsimp [k];omega
  have hn := polynomial_nonnegative a b c k hr hk
  have hdrop := mul_nonneg (show 0 ≤ -c by omega) hd.1
  have hdropmax := mul_le_mul (show -c ≤ 1265 by omega) hd.2 hd.1 (by norm_num : (0:Int) ≤ 1265)
  let n := a*2^31+b*(k:Int)*2^18+c*(k:Int)*(k:Int)
  let full := n-c*(drop:Int)
  have hfull : 0 ≤ full := by dsimp [full,n]; nlinarith only [hn,hdrop]
  have hz : Mufu.interpolate MufuTables.sin index = full/2^19 := by
    change ((MufuTables.sin.coefficients[index/131072]!).1*16777216+0+
      (MufuTables.sin.coefficients[index/131072]!).2.1*(k:Int)*2048+
      ((MufuTables.sin.coefficients[index/131072]!).2.2*(square:Int))/128)/4096 = full/524288
    rw [show index/131072=segment.val by rfl,ht]
    dsimp only
    rw [floor_join,hsquare]
    congr 1
    dsimp [full,n]
    ring
  have hrem : 0 ≤ full%2^19 ∧ full%2^19 ≤ 2^19-1 := by omega
  have hid : full/2^19*2^19 = full-full%2^19 := by omega
  have hbprod := mul_le_mul hr.2.2.2.1 hk.2 hk.1 (by norm_num : (0:Int) ≤ 51474)
  have hsnonneg : 0 ≤ (square:Int) := Int.natCast_nonneg _
  have hcprod := mul_nonpos_of_nonpos_of_nonneg hc.2 hsnonneg
  have hupper : full < 2^57 := by
    have ha := hr.2.1
    dsimp [full,n] at *
    rw [hsquare] at hcprod
    nlinarith only [ha,hbprod,hcprod]
  change 0 ≤ _ ∧ _ < _ ∧ _ ≤ _ ∧ _ ≤ _
  rw [hz]
  change 0 ≤ full/2^19 ∧ full/2^19 < 2^38 ∧ -2^19 ≤ full/2^19*2^19-n ∧ full/2^19*2^19-n ≤ 1265*4718593
  dsimp [full] at *
  constructor
  · exact Int.ediv_nonneg hfull (by norm_num)
  constructor
  · omega
  constructor <;> nlinarith only [hid,hrem.1,hrem.2,hdrop,hdropmax]

#print axioms ranges_valid
#print axioms interpolation_facts
end GB300Math.SineInterpolationInteger
