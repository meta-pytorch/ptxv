import SqrtBounds
namespace GB300Math.Log2Interpolation
set_option maxRecDepth 100000
set_option maxHeartbeats 4000000
abbrev RangeValid (a b c : Int) : Prop :=
  7≤a ∧ a≤2^26 ∧ 0≤b ∧ b≤47272 ∧ -11632≤c ∧ c≤0 ∧
  0≤b*2^22+2*c*131071 ∧
  a*2^34+857088*4096+b*131071*2^22+c*131071^2-c*4718593<2^60
set_option maxHeartbeats 0 in
theorem ranges_valid : ∀ segment : Fin 64,
    let (a,b,c) := MufuTables.lg2.coefficients[segment.val]!
    RangeValid a b c := by decide +kernel

theorem floor_join (a q : Int) : (a+q/4096)/4096=(a*4096+q)/16777216 := by omega

theorem polynomial_bounds (a b c k : Int) (hr : RangeValid a b c)
    (hk : 0≤k ∧ k≤131071) :
    a*2^34+857088*4096≤a*2^34+857088*4096+b*k*2^22+c*k*k ∧
    a*2^34+857088*4096+b*k*2^22+c*k*k≤
      a*2^34+857088*4096+b*131071*2^22+c*131071^2 := by
  rcases hr with ⟨ha,ha',hb,hb',hc,hc',hs,hend⟩
  have hck := mul_nonpos_of_nonpos_of_nonneg hc' (show 0≤2*131071-k by omega)
  have hcoef : 0≤b*2^22+c*k := by nlinarith only [hck,hs]
  have hlo := mul_nonneg hcoef hk.1
  have hck' := mul_nonpos_of_nonpos_of_nonneg hc' (show 0≤131071-k by omega)
  have hcoef' : 0≤b*2^22+c*(131071+k) := by nlinarith only [hck',hs]
  have hhi := mul_nonneg hcoef' (show 0≤131071-k by omega)
  constructor <;> nlinarith only [hlo,hhi]

theorem interpolation_facts (index : Nat) (hi : index<2^23) :
    let (a,b,c) := MufuTables.lg2.coefficients[index/131072]!
    let k : Int := (index%131072:Nat)
    let n : Int := a*2^34+857088*4096+b*k*2^22+c*k*k
    let z := Mufu.interpolate MufuTables.lg2 index
    0<z ∧ z<2^36 ∧ -(2^24:Int)≤z*2^24-n ∧ z*2^24-n≤11632*4718593 := by
  let segment : Fin 64 := ⟨index/131072,by omega⟩
  rcases ht : MufuTables.lg2.coefficients[segment.val]! with ⟨a,b,c⟩
  have hr := ranges_valid segment
  rw [ht] at hr
  have hc : -11632≤c ∧ c≤0 := ⟨hr.2.2.2.2.1,hr.2.2.2.2.2.1⟩
  let k := index%131072
  let square := Fp64Seed.squareTruncated k 19
  let drop := k*k-square
  have hs : square≤k*k := by unfold square Fp64Seed.squareTruncated;exact Nat.sub_le _ _
  have hd : 0≤(drop:Int) ∧ (drop:Int)≤4718593 :=
    ⟨Int.natCast_nonneg _,by exact_mod_cast SqrtBounds.square_drop_bound k⟩
  have hsquare : (square:Int)=(k:Int)*(k:Int)-(drop:Int) := by
    have h : drop+square=k*k := Nat.sub_add_cancel hs
    have hc : (drop:Int)+(square:Int)=(k:Int)*(k:Int) := by exact_mod_cast h
    omega
  have hk : 0≤(k:Int) ∧ (k:Int)≤131071 := by dsimp [k];omega
  have hn := polynomial_bounds a b c k hr hk
  have hdrop := mul_nonneg (show 0≤-c by omega) hd.1
  have hdropmax := mul_le_mul (show -c≤11632 by omega) hd.2 hd.1 (by norm_num : (0:Int)≤11632)
  let n := a*2^34+857088*4096+b*(k:Int)*2^22+c*(k:Int)*(k:Int)
  let full := n-c*(drop:Int)
  have hfull : 2^24≤full := by
    have ha := hr.1
    dsimp [full,n]
    nlinarith only [hn.1,hdrop,ha]
  have hz : Mufu.interpolate MufuTables.lg2 index=full/2^24 := by
    change ((MufuTables.lg2.coefficients[index/131072]!).1*4194304+857088+
      (MufuTables.lg2.coefficients[index/131072]!).2.1*(k:Int)*1024+
      ((MufuTables.lg2.coefficients[index/131072]!).2.2*(square:Int))/4096)/4096=full/16777216
    rw [show index/131072=segment.val by rfl,ht]
    dsimp only
    rw [floor_join,hsquare]
    congr 1
    dsimp [full,n]
    ring
  have hrem : 0≤full%2^24 ∧ full%2^24≤2^24-1 := by omega
  have hid : full/2^24*2^24=full-full%2^24 := by omega
  have hdropc := mul_le_mul_of_nonneg_left hd.2 (show 0≤-c by omega)
  have hupper : full<2^60 := by
    have hend := hr.2.2.2.2.2.2.2
    dsimp [full,n]
    nlinarith only [hn.2,hdropc,hend]
  change 0<_ ∧ _<_ ∧ _≤_ ∧ _≤_
  rw [hz]
  change 0<full/2^24 ∧ full/2^24<2^36 ∧ -2^24≤full/2^24*2^24-n ∧ full/2^24*2^24-n≤11632*4718593
  dsimp [full] at *
  constructor
  · omega
  constructor
  · omega
  constructor <;> nlinarith only [hid,hrem.1,hrem.2,hdrop,hdropmax]
#print axioms ranges_valid
#print axioms interpolation_facts
end GB300Math.Log2Interpolation
