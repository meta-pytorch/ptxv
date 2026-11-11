import TanhCentralIndex
import TanhCentralCells
import TanhRange
namespace GB300Math.TanhCentralModel
open TanhCentralCells TanhCentralInteger TanhCentralIndex TanhCentralPolynomial
set_option maxRecDepth 100000
set_option maxHeartbeats 5000000

def index (e f:Nat) := (f/2^(discard e))/2^17
def argument (e f:Nat) := (f/2^(discard e))%2^17
def value (e f:Nat) : Int :=
  let (a,b,c):=coeff e (index e f)
  a*2^34+b*(argument e f:Int)*2^17+c*(Fp64Seed.squareTruncated (argument e f) 19:Int)

theorem index_valid (e:Fin 11) (f:Nat) (hf:f<2^23) :
    index e.val f<16 ∧ index e.val f<(TanhTables.binades[e.val]!).2.size := by
  fin_cases e <;> norm_num [index,TanhCentralIndex.discard,TanhTables.binades] <;> omega

def Positive (e j:Nat) : Prop :=
  let (a,b,c):=coeff e j
  0<a+c ∧ 0≤b ∧ c≤0
noncomputable instance (e j:Nat) : Decidable (Positive e j) := inferInstanceAs (Decidable (_ ∧ _))
theorem positive_coeff : ∀e:Fin 11,∀j:Fin 16,j.val<(TanhTables.binades[e.val]!).2.size → Positive e.val j.val := by decide +kernel

theorem value_range (e:Fin 11) (f:Nat) (hf:f<2^23) :
    0<value e.val f ∧ value e.val f<(2:Int)^(scale e.val) := by
  let j:Fin 16:=⟨index e.val f,(index_valid e f hf).1⟩
  have hv:=positive_coeff e j (index_valid e f hf).2
  have hk : argument e.val f<2^17 := Nat.mod_lt _ (by decide)
  have hsq : Fp64Seed.squareTruncated (argument e.val f) 19≤argument e.val f*argument e.val f := by
    unfold Fp64Seed.squareTruncated;exact Nat.sub_le _ _
  have hkr : (0:Int)≤argument e.val f ∧ (argument e.val f:Int)≤131071 := by omega
  have hsq' : (Fp64Seed.squareTruncated (argument e.val f) 19:Int)≤(2:Int)^34 := by
    have hc : (Fp64Seed.squareTruncated (argument e.val f) 19:Int)≤(argument e.val f:Int)*(argument e.val f:Int) := by exact_mod_cast hsq
    nlinarith
  have hu:=TanhRange.polynomial_bound (coeff e.val j.val).1 (coeff e.val j.val).2.1 (coeff e.val j.val).2.2
    (scale e.val) (argument e.val f) (TanhRange.certificate e j) hk
  unfold Positive at hv
  rcases hc:coeff e.val j.val with ⟨a,b,c⟩
  rw [hc] at hv hu
  dsimp only at hv hu
  constructor
  · have h1:=mul_le_mul_of_nonpos_left hsq' hv.2.2
    have h2:=mul_nonneg hv.2.1 hkr.1
    change 0<value e.val f
    unfold value
    change 0<(coeff e.val j.val).1*2^34+(coeff e.val j.val).2.1*(argument e.val f:Int)*2^17+(coeff e.val j.val).2.2*(Fp64Seed.squareTruncated (argument e.val f) 19:Int)
    rw [hc];dsimp only
    nlinarith
  · unfold value
    change (coeff e.val j.val).1*2^34+(coeff e.val j.val).2.1*(argument e.val f:Int)*2^17+(coeff e.val j.val).2.2*(Fp64Seed.squareTruncated (argument e.val f) 19:Int)<_
    rw [hc]
    exact hu

theorem unrounded_error (e:Fin 11) (f:Nat) (hf:f<2^23) :
    |(value e.val f:ℝ)/2^(scale e.val)-Real.tanh ((2^23+(f:ℝ))*(2:ℝ)^((e.val:Int)-31))|≤145/10000000 := by
  let u:Fin 512:=⟨f/stride e.val,lt_of_lt_of_le (geometry e f hf).2.1 (geometry e f hf).2.2.1⟩
  let j:Fin 16:=⟨index e.val f,(index_valid e f hf).1⟩
  have ht:=position_range e f hf
  have hp:=ideal_error e u (certificate e u (geometry e f hf).2.1) (position e.val f) ht
  have hi:=input_identity e f hf
  rw [hi] at hp
  have hk : argument e.val f<2^17 := Nat.mod_lt _ (by decide)
  have herr:=interpolation_error (coeff e.val j.val).1 (coeff e.val j.val).2.1 (coeff e.val j.val).2.2
    (scale e.val) (argument e.val f) hk (remainder e.val f) (remainder_range e.val f)
  have hcost: (|( (coeff e.val j.val).2.1:ℝ)| *2^17+|( (coeff e.val j.val).2.2:ℝ)| *(2*131071+1+4718593))/2^(scale e.val)≤1/400000 := by
    have h:=(Rat.cast_le (K:=ℝ)).mpr (costs e j)
    simpa [cost] using h
  have hn:=native_identity e f hf
  dsimp only at hn
  have herr' : |(value e.val f:ℝ)/2^(scale e.val)-nativePoly e.val u.val (position e.val f)|≤1/400000 := by
    rw [hn]
    exact herr.trans hcost
  have htriangle:=abs_sub_le ((value e.val f:ℝ)/2^(scale e.val))
    (nativePoly e.val u.val (position e.val f)) (Real.tanh ((2^23+(f:ℝ))*(2:ℝ)^((e.val:Int)-31)))
  linarith

#print axioms index_valid
#print axioms positive_coeff
#print axioms value_range
#print axioms unrounded_error
end GB300Math.TanhCentralModel
