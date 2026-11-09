import SqrtBounds
namespace GB300Math.Exp2Interpolation
set_option maxRecDepth 100000
set_option maxHeartbeats 3000000
abbrev RangeValid (a b c : Int) : Prop :=
  2^25≤a ∧ 0≤b ∧ 0≤c ∧ c≤980 ∧
  a*2^32+3205496832+b*131071*2^19+c*131071^2<2^58
set_option maxHeartbeats 0 in
theorem ranges_valid : ∀segment:Fin 64,
    let (a,b,c):=MufuTables.ex2.coefficients[segment.val]!
    RangeValid a b c := by decide +kernel

theorem interpolation_facts (index : Nat) (hi : index<2^23) :
    let (a,b,c):=MufuTables.ex2.coefficients[index/131072]!
    let k:Int:=(index%131072:Nat)
    let n:Int:=a*2^32+3205496832+b*k*2^19+c*k*k
    let z:=Mufu.interpolate MufuTables.ex2 index
    2^23≤z ∧ z<2^24 ∧ -(980*4718593+2^34:Int)≤z*2^34-n ∧ z*2^34-n≤0 := by
  let segment:Fin 64:=⟨index/131072,by omega⟩
  rcases ht:MufuTables.ex2.coefficients[segment.val]! with ⟨a,b,c⟩
  have hr:=ranges_valid segment
  rw [ht] at hr
  let k:=index%131072
  let square:=Fp64Seed.squareTruncated k 19
  let drop:=k*k-square
  have hs:square≤k*k:=by unfold square Fp64Seed.squareTruncated;exact Nat.sub_le _ _
  have hd:0≤(drop:Int) ∧ (drop:Int)≤4718593:=by
    exact ⟨Int.natCast_nonneg _,by exact_mod_cast SqrtBounds.square_drop_bound k⟩
  have hsquare:(square:Int)=(k:Int)*(k:Int)-(drop:Int):=by
    have h:drop+square=k*k:=Nat.sub_add_cancel hs
    have hc:(drop:Int)+(square:Int)=(k:Int)*(k:Int):=by exact_mod_cast h
    omega
  have hk:0≤(k:Int) ∧ (k:Int)≤131071:=by dsimp [k];omega
  let n:=a*2^32+3205496832+b*(k:Int)*2^19+c*(k:Int)*(k:Int)
  let full:=n-c*(drop:Int)
  have hdrop:0≤c*(drop:Int):=mul_nonneg hr.2.2.1 hd.1
  have hdropmax:c*(drop:Int)≤980*4718593:=mul_le_mul hr.2.2.2.1 hd.2 hd.1 (by norm_num)
  have hbase:a*2^32≥2^57:=by nlinarith [hr.1]
  have hb:0≤b*(k:Int)*2^19:=mul_nonneg (mul_nonneg hr.2.1 hk.1) (by norm_num)
  have hc:0≤c*(square:Int):=mul_nonneg hr.2.2.1 (Int.natCast_nonneg _)
  have hfull:2^57≤full:=by
    dsimp [full,n]
    rw [hsquare] at hc
    nlinarith only [hbase,hb,hc]
  have hbupper:=mul_le_mul_of_nonneg_left hk.2 hr.2.1
  have hksq:(k:Int)*(k:Int)≤131071^2:=by nlinarith only [hk.1,hk.2]
  have hcupper:=mul_le_mul_of_nonneg_left hksq hr.2.2.1
  have hnupper:n<2^58:=by
    dsimp [n]
    nlinarith only [hr.2.2.2.2,hbupper,hcupper]
  have hz:Mufu.interpolate MufuTables.ex2 index=full/2^34:=by
    simp only [Mufu.interpolate, MufuTables.ex2, ge_iff_le, le_refl, ite_true, Int.toNat_zero, pow_zero, mul_one]
    change ((MufuTables.ex2.coefficients[index/131072]!).1*4294967296+3205496832+
      (MufuTables.ex2.coefficients[index/131072]!).2.1*(k:Int)*524288+
      ((MufuTables.ex2.coefficients[index/131072]!).2.2*(square:Int)))/17179869184=full/17179869184
    rw [show index/131072=segment.val from rfl,ht]
    dsimp only
    rw [hsquare]
    congr 1
    dsimp [full,n]
    ring
  have hrem:0≤full%2^34 ∧ full%2^34<2^34:=by omega
  have hid:full/2^34*2^34=full-full%2^34:=by omega
  change 2^23≤_ ∧ _<2^24 ∧ _≤_ ∧ _≤0
  rw [hz]
  change 2^23≤full/2^34 ∧ full/2^34<2^24 ∧ -(980*4718593+2^34:Int)≤full/2^34*2^34-n ∧ full/2^34*2^34-n≤0
  dsimp [full] at *
  constructor
  · omega
  constructor
  · omega
  constructor <;> nlinarith only [hid,hrem.1,hrem.2,hdrop,hdropmax]

#print axioms ranges_valid
#print axioms interpolation_facts
end GB300Math.Exp2Interpolation
