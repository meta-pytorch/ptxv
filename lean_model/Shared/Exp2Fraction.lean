import Exp2Local
import Exp2Interpolation
namespace GB300Math.Exp2Fraction
open Exp2Cells Exp2Local Exp2Interpolation
set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

def cell (index : Nat) (hi : index<2^23) : Fin 256 := ⟨index/32768,by omega⟩
noncomputable def parameter (index : Nat) : ℝ := ((index%32768:Nat):ℝ)/32768

theorem parameter_range (index : Nat) : 0≤parameter index ∧ parameter index≤1 := by
  have hi : index%32768≤32768 := by omega
  have hc : ((index%32768:Nat):ℝ)≤32768 := by exact_mod_cast hi
  unfold parameter
  constructor
  · positivity
  · linarith

theorem coordinate (index : Nat) (hi : index<2^23) :
    (((cell index hi).val:ℝ)+parameter index)/256=(index:ℝ)/2^23 := by
  have h : index%32768+32768*(index/32768)=index := Nat.mod_add_div _ _
  have hc : ((index%32768:Nat):ℝ)+32768*((index/32768:Nat):ℝ)=(index:ℝ) := by exact_mod_cast h
  dsimp [cell,parameter]
  linarith

theorem native_identity (index : Nat) (hi : index<2^23) :
    let (a,b,c):=MufuTables.ex2.coefficients[index/131072]!
    let k:ℝ:=((index%131072:Nat):ℝ)
    nativePoly (cell index hi).val (parameter index)=
      ((a:ℝ)*2^32+3205496832+(b:ℝ)*k*2^19+(c:ℝ)*k*k)/2^57 := by
  have hs : index/32768/4=index/131072 := by omega
  have hk : index%131072=(index/32768%4)*32768+index%32768 := by omega
  have hkr : ((index%131072:Nat):ℝ)=((index/32768%4:Nat):ℝ)*32768+((index%32768:Nat):ℝ) := by exact_mod_cast hk
  unfold nativePoly nativeCoefficients cell parameter
  dsimp only
  rw [hs]
  rcases MufuTables.ex2.coefficients[index/131072]! with ⟨a,b,c⟩
  dsimp only
  push_cast
  rw [hkr]
  ring

theorem interpolation_error (index : Nat) (hi : index<2^23) :
    -(152/10^9:ℝ)≤(Mufu.interpolate MufuTables.ex2 index:ℝ)/2^23-
      nativePoly (cell index hi).val (parameter index) ∧
    (Mufu.interpolate MufuTables.ex2 index:ℝ)/2^23-
      nativePoly (cell index hi).val (parameter index)≤0 := by
  have h:=interpolation_facts index hi
  have hid:=native_identity index hi
  rcases hcoef:MufuTables.ex2.coefficients[index/131072]! with ⟨a,b,c⟩
  rw [hcoef] at h hid
  dsimp only at h hid
  have lo : -(980*4718593+2^34:ℝ)≤(Mufu.interpolate MufuTables.ex2 index:ℝ)*2^34-
    ((a:ℝ)*2^32+3205496832+(b:ℝ)*((index%131072:Nat):ℝ)*2^19+(c:ℝ)*((index%131072:Nat):ℝ)*((index%131072:Nat):ℝ)) := by
    have hh := (Int.cast_le (R:=ℝ)).mpr h.2.2.1
    simp only [Int.cast_sub,Int.cast_add,Int.cast_mul,Int.cast_neg,Int.cast_pow,Int.cast_natCast,Int.cast_ofNat] at hh
    norm_num at hh ⊢
    exact hh
  have up : (Mufu.interpolate MufuTables.ex2 index:ℝ)*2^34-
    ((a:ℝ)*2^32+3205496832+(b:ℝ)*((index%131072:Nat):ℝ)*2^19+(c:ℝ)*((index%131072:Nat):ℝ)*((index%131072:Nat):ℝ))≤0 := by
    have hh := (Int.cast_le (R:=ℝ)).mpr h.2.2.2
    simp only [Int.cast_sub,Int.cast_add,Int.cast_mul,Int.cast_pow,Int.cast_natCast,Int.cast_ofNat] at hh
    norm_num at hh ⊢
    exact hh
  rw [hid]
  constructor <;> linarith

theorem fractional_error (index : Nat) (hi : index<2^23) :
    |(Mufu.interpolate MufuTables.ex2 index:ℝ)/2^23-
      Real.exp (Real.log 2*((index:ℝ)/2^23))|≤1/5000000 := by
  have hp:=polynomial_ideal (cell index hi) (parameter index) (parameter_range index)
  have hcoord:=coordinate index hi
  have hexp : Real.log 2*((((cell index hi).val:ℝ)+parameter index)/256)=Real.log 2*((index:ℝ)/2^23) := by rw [hcoord]
  have heq : Real.log 2*((((cell index hi).val:ℝ)+parameter index))/256=Real.log 2*((index:ℝ)/2^23) := by nlinarith only [hexp]
  rw [heq] at hp
  have hn:=interpolation_error index hi
  exact abs_le.mpr ⟨by linarith,by linarith⟩

theorem fractional_signed_error (index : Nat) (hi : index<2^23) :
    -(123/10^9:ℝ)≤(Mufu.interpolate MufuTables.ex2 index:ℝ)/2^23-
      Real.exp (Real.log 2*((index:ℝ)/2^23)) ∧
    (Mufu.interpolate MufuTables.ex2 index:ℝ)/2^23-
      Real.exp (Real.log 2*((index:ℝ)/2^23))≤196/10^9 := by
  have hp:=polynomial_ideal (cell index hi) (parameter index) (parameter_range index)
  have hcoord:=coordinate index hi
  have heq : Real.log 2*((((cell index hi).val:ℝ)+parameter index))/256=Real.log 2*((index:ℝ)/2^23) := by rw [←hcoord];ring
  rw [heq] at hp
  have hn:=interpolation_error index hi
  constructor <;> linarith

#print axioms fractional_signed_error
#print axioms parameter_range
#print axioms coordinate
#print axioms native_identity
#print axioms interpolation_error
#print axioms fractional_error
end GB300Math.Exp2Fraction
