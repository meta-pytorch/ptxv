import Log2Polynomial
import Log2Interpolation
namespace GB300Math.Log2Table
open FloatLib.Floats.Formats.BinaryInterchange
set_option maxRecDepth 12000
set_option maxHeartbeats 6000000

theorem table_error (index : Nat) (hi : index<2^23) :
    |(Mufu.interpolate MufuTables.lg2 index:ℝ)/2^36-Real.log (1+(index:ℝ)/2^23)/Real.log 2| ≤ 1/5000000 := by
  let j : Fin 512 := ⟨index/16384,by omega⟩
  let t : ℝ := (index%16384:Nat)/16384
  have ht : 0 ≤ t ∧ t ≤ 1 := by
    have hr : ((index%16384:Nat):ℝ) ≤ 16384 := by exact_mod_cast (show index%16384≤16384 by omega)
    dsimp [t]
    constructor
    · positivity
    · linarith
  have hp := Log2Polynomial.cell_error j t ht
  have hg : j.val/8=index/131072 := by dsimp [j];omega
  have hk : (j.val%8)*16384+index%16384=index%131072 := by dsimp [j];omega
  have hindex : j.val*16384+index%16384=index := by
    dsimp only [j]
    simpa only [Nat.mul_comm,Nat.add_comm] using Nat.mod_add_div index 16384
  have hindexr : (j.val:ℝ)*16384+((index%16384:Nat):ℝ)=index := by exact_mod_cast hindex
  have hangle : 1+(j.val+t)/512=1+(index:ℝ)/2^23 := by
    dsimp [t]
    have hmul := congrArg (fun x:ℝ => x) hindexr
    nlinarith only [hmul]
  rw [hangle] at hp
  rcases hcoeff : MufuTables.lg2.coefficients[index/131072]! with ⟨a,b,c⟩
  let k : Int := (index%131072:Nat)
  let n : Int := a*2^34+857088*4096+b*k*2^22+c*k*k
  have hpoly : Log2Polynomial.nativePoly j.val t=(n:ℝ)/2^60 := by
    unfold Log2Polynomial.nativePoly Log2Cells.nativeCoefficients
    rw [hg,hcoeff]
    dsimp only
    push_cast
    have hkr : ((j.val%8:Nat):ℝ)*16384+((index%16384:Nat):ℝ)=(index%131072:Nat) := by exact_mod_cast hk
    dsimp only [t,n,k]
    simp only [Int.cast_add,Int.cast_mul,Int.cast_pow,Int.cast_ofNat,Int.cast_natCast]
    rw [←hkr]
    ring
  rw [hpoly] at hp
  have h := Log2Interpolation.interpolation_facts index hi
  rw [hcoeff] at h
  dsimp only at h
  have hlo : -(2^24:ℝ) ≤ (Mufu.interpolate MufuTables.lg2 index:ℝ)*2^24-(n:ℝ) := by exact_mod_cast h.2.2.1
  have hhi : (Mufu.interpolate MufuTables.lg2 index:ℝ)*2^24-(n:ℝ) ≤ (11632*4718593:ℝ) := by exact_mod_cast h.2.2.2
  have herror : |(Mufu.interpolate MufuTables.lg2 index:ℝ)/2^36-(n:ℝ)/2^60| ≤ 48/1000000000 := by
    apply abs_le.mpr
    constructor <;> nlinarith only [hlo,hhi]
  have ht := abs_sub_le ((Mufu.interpolate MufuTables.lg2 index:ℝ)/2^36) ((n:ℝ)/2^60) (Real.log (1+(index:ℝ)/2^23)/Real.log 2)
  linarith

#print axioms table_error
end GB300Math.Log2Table
