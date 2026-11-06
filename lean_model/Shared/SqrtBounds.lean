import RcpBounds
set_option maxRecDepth 1000000
set_option maxHeartbeats 0
namespace GB300Math.SqrtBounds
open GB300Math.RcpBounds

theorem square_drop_bound (k : Nat) : k*k-Fp64Seed.squareTruncated k 19 ≤ 4718593 := by
  have h0 : k &&& 262142 ≤ 262142 := Nat.and_le_right
  have h1 : k &&& 131068 ≤ 131068 := Nat.and_le_right
  have h2 : k &&& 65528 ≤ 65528 := Nat.and_le_right
  have h3 : k &&& 32752 ≤ 32752 := Nat.and_le_right
  have h4 : k &&& 16352 ≤ 16352 := Nat.and_le_right
  have h5 : k &&& 8128 ≤ 8128 := Nat.and_le_right
  have h6 : k &&& 3968 ≤ 3968 := Nat.and_le_right
  have h7 : k &&& 1792 ≤ 1792 := Nat.and_le_right
  have h8 : k &&& 512 ≤ 512 := Nat.and_le_right
  simp only [Fp64Seed.squareTruncated, Std.Legacy.Range.forIn_eq_forIn_range']
  simp [Std.Legacy.Range.size, List.range', pure_yield_ite, Id.run]
  change k*k ≤ 4718593+(k*k-_)
  simp only [accumulate_three_ite, accumulate_ite]
  have b0 : (if k % 2 = 1 then 1 + (k &&& 262142) * 2 else 0) ≤ 524285 := by split <;> omega
  have b1 : (if k >>> 1 % 2 = 1 then 4 + (k &&& 131068) * 4 else 0) ≤ 524276 := by split <;> omega
  have b2 : (if k >>> 2 % 2 = 1 then 16 + (k &&& 65528) * 8 else 0) ≤ 524240 := by split <;> omega
  have b3 : (if k >>> 3 % 2 = 1 then 64 + (k &&& 32752) * 16 else 0) ≤ 524096 := by split <;> omega
  have b4 : (if k >>> 4 % 2 = 1 then 256 + (k &&& 16352) * 32 else 0) ≤ 523520 := by split <;> omega
  have b5 : (if k >>> 5 % 2 = 1 then 1024 + (k &&& 8128) * 64 else 0) ≤ 521216 := by split <;> omega
  have b6 : (if k >>> 6 % 2 = 1 then 4096 + (k &&& 3968) * 128 else 0) ≤ 512000 := by split <;> omega
  have b7 : (if k >>> 7 % 2 = 1 then 16384 + (k &&& 1792) * 256 else 0) ≤ 475136 := by split <;> omega
  have b8 : (if k >>> 8 % 2 = 1 then 65536 + (k &&& 512) * 512 else 0) ≤ 327680 := by split <;> omega
  have b9 : (if k >>> 9 % 2 = 1 then 262144 else 0) ≤ 262144 := by split <;> omega
  omega

structure Parameters where
  (a b c m step : Int)

def parameters (segment : Nat) : Parameters :=
  let (a,b,c) := MufuTables.sqrt.coefficients[segment]!
  ⟨a*2^34+17179344896,b*2^19,c*2,
   (2^23+(segment%64 : Nat)*131072)*2^(23+segment/64),2^(23+segment/64)⟩

def qmax (p : Parameters) : Int := p.a+p.b*131071+p.c*131071^2

def taylor (p : Parameters) (left : Int) : Int × Int × Int × Int × Int :=
  let p0 := p.a*p.a-p.m*2^72
  let p1 := 2*p.a*p.b-p.step*2^72
  let p2 := p.b*p.b+2*p.a*p.c
  let p3 := 2*p.b*p.c
  let p4 := p.c*p.c
  (p0+p1*left+p2*left^2+p3*left^3+p4*left^4,
   p1+2*p2*left+3*p3*left^2+4*p4*left^3,
   p2+3*p3*left+6*p4*left^2,p3+4*p4*left,p4)

def lower (q0 q1 q2 q3 q4 d : Int) := q0+boxLo q1 d+boxLo q2 (d^2)+boxLo q3 (d^3)+boxLo q4 (d^4)
def upper (q0 q1 q2 q3 q4 d : Int) := q0+boxHi q1 d+boxHi q2 (d^2)+boxHi q3 (d^3)+boxHi q4 (d^4)

theorem polynomial_box (q0 q1 q2 q3 q4 t d : Int) (ht : 0 ≤ t) (hd : t ≤ d) :
    lower q0 q1 q2 q3 q4 d ≤ q0+q1*t+q2*t^2+q3*t^3+q4*t^4 ∧
    q0+q1*t+q2*t^2+q3*t^3+q4*t^4 ≤ upper q0 q1 q2 q3 q4 d := by
  have h1 := term_box q1 t d ht hd
  have h2 := term_box q2 (t^2) (d^2) (pow_nonneg ht 2) (pow_le_pow_left₀ ht hd 2)
  have h3 := term_box q3 (t^3) (d^3) (pow_nonneg ht 3) (pow_le_pow_left₀ ht hd 3)
  have h4 := term_box q4 (t^4) (d^4) (pow_nonneg ht 4) (pow_le_pow_left₀ ht hd 4)
  simp only [lower,upper]
  constructor <;> omega

abbrev CellBound (p : Parameters) (left : Int) : Prop :=
  let (q0,q1,q2,q3,q4) := taylor p left
  let e := -p.c*4718593
  p.c ≤ 0 ∧ 0 ≤ p.b+262142*p.c ∧ 2^59 ≤ p.a ∧
  qmax p+e < 2^60 ∧
  -2^97+2*qmax p*2^36 ≤ lower q0 q1 q2 q3 q4 8191 ∧
  upper q0 q1 q2 q3 q4 8191+2*qmax p*e+e^2 ≤ 2^97

set_option maxHeartbeats 0 in
theorem cells_valid : ∀ segment : Fin 128, ∀ cell : Fin 16,
    CellBound (parameters segment.val) ((cell.val : Int)*8192) := by decide +kernel

theorem numerator_range (p : Parameters) (k : Int)
    (hc : p.c ≤ 0) (hb : 0 ≤ p.b+262142*p.c) (hk : 0 ≤ k ∧ k ≤ 131071) :
    p.a ≤ p.a+p.b*k+p.c*k*k ∧ p.a+p.b*k+p.c*k*k ≤ qmax p := by
  have hck := mul_le_mul_of_nonpos_left hk.2 hc
  have hc2 := mul_le_mul_of_nonpos_left (show k+131071 ≤ 262142 by omega) hc
  have hlo := mul_nonneg (show 0 ≤ p.b+p.c*k by nlinarith) hk.1
  have hhi := mul_nonneg (show 0 ≤ 131071-k by omega)
    (show 0 ≤ p.b+p.c*(131071+k) by nlinarith)
  dsimp [qmax]
  constructor <;> nlinarith

theorem cell_error (p : Parameters) (left k dropped : Int)
    (hcell : CellBound p left) (hk : 0 ≤ k ∧ k ≤ 131071)
    (hwindow : 0 ≤ k-left ∧ k-left ≤ 8191)
    (hd : 0 ≤ dropped ∧ dropped ≤ 4718593) :
    let z := (p.a+p.b*k+p.c*(k*k-dropped))/2^36
    2^23 ≤ z ∧ z < 2^24 ∧
    -2^25 ≤ z*z-(p.m+p.step*k) ∧ z*z-(p.m+p.step*k) ≤ 2^25 := by
  rcases ht : taylor p left with ⟨q0,q1,q2,q3,q4⟩
  simp only [CellBound,ht] at hcell
  rcases hcell with ⟨hc,hb,ha,hmax,hlo,hhi⟩
  have hq := numerator_range p k hc hb hk
  have hbox := polynomial_box q0 q1 q2 q3 q4 (k-left) 8191 hwindow.1 hwindow.2
  have heq : q0+q1*(k-left)+q2*(k-left)^2+q3*(k-left)^3+q4*(k-left)^4 =
      (p.a+p.b*k+p.c*k*k)^2-(p.m+p.step*k)*2^72 := by
    have vals := ht
    simp only [taylor,Prod.mk.injEq] at vals
    rcases vals with ⟨rfl,rfl,rfl,rfl,rfl⟩
    ring
  rw [heq] at hbox
  let q := p.a+p.b*k+p.c*k*k
  let n := p.a+p.b*k+p.c*(k*k-dropped)
  let z := n/2^36
  let delta := -p.c*dropped-n%2^36
  let e := -p.c*4718593
  have hr : 0 ≤ n%2^36 ∧ n%2^36 ≤ 2^36-1 := by omega
  have hd0 := mul_nonneg (show 0 ≤ -p.c by omega) hd.1
  have hd1 := mul_le_mul_of_nonneg_left hd.2 (show 0 ≤ -p.c by omega)
  have he0 : 0 ≤ e := by dsimp [e]; omega
  have hdelta : -2^36 ≤ delta ∧ delta ≤ e := by dsimp [delta,e]; omega
  have hn : z*2^36 = q+delta := by
    have hv : n = z*2^36+n%2^36 := by dsimp [z]; omega
    dsimp [q,delta,n] at *
    nlinarith only [hv]
  have hzn : 2^23 ≤ z ∧ z < 2^24 := by
    dsimp [q] at hn
    constructor <;> omega
  have hloss := mul_nonneg (show 0 ≤ 2*q by dsimp [q]; omega)
    (show 0 ≤ delta+2^36 by omega)
  have hqmax := mul_nonneg (show 0 ≤ 2*(qmax p-q) by dsimp [q]; omega)
    (show (0 : Int) ≤ 2^36 by norm_num)
  have hlower : -2*qmax p*2^36 ≤ (q+delta)^2-q^2 := by nlinarith [sq_nonneg delta]
  have hstep := mul_nonneg (show 0 ≤ e-delta by omega)
    (show 0 ≤ 2*q+e+delta by dsimp [q]; omega)
  have hu := mul_nonneg (show 0 ≤ 2*(qmax p-q) by dsimp [q]; omega) he0
  have hupper : (q+delta)^2-q^2 ≤ 2*qmax p*e+e^2 := by nlinarith
  have hid : (z*z-(p.m+p.step*k))*2^72 =
      (q^2-(p.m+p.step*k)*2^72)+((q+delta)^2-q^2) := by
    rw [← hn]
    ring
  change 2^23 ≤ z ∧ z < 2^24 ∧ _
  refine ⟨hzn.1,hzn.2,?_,?_⟩
  · dsimp [q,e] at hid hlower hupper
    nlinarith only [hid,hlower,hbox.1,hlo]
  · dsimp [q,e] at hid hlower hupper
    nlinarith only [hid,hupper,hbox.2,hhi]

theorem interpolate_error (index : Nat) (hi : index < 2^24) :
    let z := Mufu.interpolate MufuTables.sqrt index
    let target : Int := (2^23+index%2^23)*2^(23+index/2^23)
    2^23 ≤ z ∧ z < 2^24 ∧ -2^25 ≤ z*z-target ∧ z*z-target ≤ 2^25 := by
  let segment : Fin 128 := ⟨index/131072,by omega⟩
  let k := index%131072
  let cell : Fin 16 := ⟨k/8192,by dsimp [k];omega⟩
  let drop := k*k-Fp64Seed.squareTruncated k 19
  have hs : Fp64Seed.squareTruncated k 19 ≤ k*k := by
    unfold Fp64Seed.squareTruncated
    exact Nat.sub_le _ _
  have hd : 0 ≤ (drop : Int) ∧ (drop : Int) ≤ 4718593 := by
    constructor
    · exact Int.natCast_nonneg _
    · exact_mod_cast square_drop_bound k
  have hk : 0 ≤ (k : Int) ∧ (k : Int) ≤ 131071 := by dsimp [k];omega
  have hw : 0 ≤ (k : Int)-(cell.val : Int)*8192 ∧
      (k : Int)-(cell.val : Int)*8192 ≤ 8191 := by dsimp [cell];omega
  have h := cell_error (parameters segment.val) ((cell.val : Int)*8192) k drop
    (cells_valid segment cell) hk hw hd
  have hsquare : (k : Int)*(k : Int)-(drop : Int) = Fp64Seed.squareTruncated k 19 := by
    have hh := Nat.sub_add_cancel hs
    have hc : (drop : Int)+(Fp64Seed.squareTruncated k 19 : Int) = (k : Int)*(k : Int) := by
      exact_mod_cast hh
    omega
  have hpar : segment.val/64 = index/2^23 := by dsimp [segment];omega
  have hrem : (segment.val%64 : Int)*131072+(k : Int) = index%2^23 := by
    dsimp [segment,k];omega
  have hm : (parameters segment.val).m+(parameters segment.val).step*(k : Int) =
      (2^23+(index%2^23 : Nat))*2^(23+index/2^23) := by
    dsimp [parameters]
    rw [hpar]
    push_cast
    calc
      _ = (8388608+((segment.val%64 : Int)*131072+(k : Int)))*2^(23+index/2^23) := by ring
      _ = _ := by rw [hrem]; norm_num
  dsimp only at h ⊢
  rw [hm,hsquare] at h
  convert h using 1 <;>
    dsimp [Mufu.interpolate,MufuTables.sqrt,parameters,segment,k] <;> norm_num <;> ring_nf

noncomputable def normalizedInput (index : Nat) : ℝ :=
  ((2 : ℝ)^23+((index%2^23 : Nat) : ℝ))/2^23 * 2^(index/2^23)
noncomputable def normalizedOutput (index : Nat) : ℝ := (Mufu.interpolate MufuTables.sqrt index : ℝ)/2^23

theorem sqrt_absolute_error (index : Nat) (hi : index < 2^24) :
    |normalizedOutput index - Real.sqrt (normalizedInput index)| ≤ 1/2^22 := by
  have h := interpolate_error index hi
  dsimp only at h
  have hy : (1 : ℝ) ≤ normalizedOutput index := by
    dsimp [normalizedOutput]
    have hz : (2 : ℝ)^23 ≤ (Mufu.interpolate MufuTables.sqrt index : ℝ) := by exact_mod_cast h.1
    norm_num at hz ⊢
    linarith
  have hx : (1 : ℝ) ≤ normalizedInput index := by
    have hp : (1 : ℝ) ≤ (2 : ℝ)^(index/2^23) := one_le_pow₀ (by norm_num)
    have hn : (1 : ℝ) ≤ ((2 : ℝ)^23+((index%2^23 : Nat) : ℝ))/2^23 := by
      rw [le_div_iff₀ (by norm_num)]
      norm_num
    exact one_le_mul_of_one_le_of_one_le hn hp
  have hlo : -(2 : ℝ)^25 ≤ (Mufu.interpolate MufuTables.sqrt index : ℝ)^2-
      ((2 : ℝ)^23+((index%2^23 : Nat) : ℝ))*2^(23+index/2^23) := by
    have hh := (Int.cast_le (R := ℝ)).mpr h.2.2.1
    push_cast at hh
    have hm : (index : Int)%8388608 = ((index%8388608 : Nat) : Int) := by omega
    rw [hm] at hh
    simp only [Int.cast_natCast] at hh
    convert hh using 1 <;> norm_num <;> ring
  have hhi : (Mufu.interpolate MufuTables.sqrt index : ℝ)^2-
      ((2 : ℝ)^23+((index%2^23 : Nat) : ℝ))*2^(23+index/2^23) ≤ 2^25 := by
    have hh := (Int.cast_le (R := ℝ)).mpr h.2.2.2
    push_cast at hh
    have hm : (index : Int)%8388608 = ((index%8388608 : Nat) : Int) := by omega
    rw [hm] at hh
    simp only [Int.cast_natCast] at hh
    convert hh using 1 <;> norm_num <;> ring
  have he : normalizedOutput index ^ 2 - normalizedInput index =
      ((Mufu.interpolate MufuTables.sqrt index : ℝ)^2-
      ((2 : ℝ)^23+((index%2^23 : Nat) : ℝ))*2^(23+index/2^23))/2^46 := by
    dsimp [normalizedInput,normalizedOutput]
    rw [pow_add]
    ring
  have hdiff : -(1/(2 : ℝ)^21) ≤ normalizedOutput index ^ 2-normalizedInput index ∧
      normalizedOutput index ^ 2-normalizedInput index ≤ 1/(2 : ℝ)^21 := by
    rw [he]
    constructor <;> norm_num at * <;> linarith
  have hs := Real.sq_sqrt (show 0 ≤ normalizedInput index by linarith)
  have hs0 := Real.sqrt_nonneg (normalizedInput index)
  have hs1 : (1 : ℝ) ≤ Real.sqrt (normalizedInput index) := by nlinarith
  apply abs_le.mpr
  constructor
  · by_contra hn
    have hg : 0 < Real.sqrt (normalizedInput index)-normalizedOutput index-1/2^22 := by linarith
    have hp := mul_pos hg (show 0 < Real.sqrt (normalizedInput index)+normalizedOutput index by linarith)
    nlinarith
  · by_contra hn
    have hg : 0 < normalizedOutput index-Real.sqrt (normalizedInput index)-1/2^22 := by linarith
    have hp := mul_pos hg (show 0 < Real.sqrt (normalizedInput index)+normalizedOutput index by linarith)
    nlinarith

noncomputable def positiveNormalValue (word : Nat) : ℝ :=
  (RcpBounds.positiveNormalValue word : ℝ)

theorem input_decode (index : Nat) (hi : index < 2^24) :
    positiveNormalValue (0x3f800000+index) = normalizedInput index := by
  have he : Mufu.exponent (0x3f800000+index) = 127+index/2^23 := by dsimp [Mufu.exponent];omega
  have hf : Mufu.fraction (0x3f800000+index) = index%2^23 := by dsimp [Mufu.fraction];omega
  simp only [positiveNormalValue,RcpBounds.positiveNormalValue,he,hf]
  have hp : index/2^23 = 0 ∨ index/2^23 = 1 := by omega
  rcases hp with hp | hp <;> simp only [normalizedInput,hp] <;> norm_num <;> ring

theorem root_index (index : Nat) (hi : index < 2^24) :
    Mufu.rootIndex (0x3f800000+index) = (index,0) := by
  have he : Mufu.exponent (0x3f800000+index) = 127+index/2^23 := by dsimp [Mufu.exponent];omega
  have hf : Mufu.fraction (0x3f800000+index) = index%2^23 := by dsimp [Mufu.fraction];omega
  have hp : index/2^23 = 0 ∨ index/2^23 = 1 := by omega
  rcases hp with hp | hp
  · simp only [Mufu.rootIndex,he,hf,hp]
    norm_num
    omega
  · simp only [Mufu.rootIndex,he,hf,hp]
    norm_num
    omega

theorem output_word (index : Nat) (hi : index < 2^24) :
    Mufu.sqrt (0x3f800000+index) =
      0x3f800000+(Mufu.interpolate MufuTables.sqrt index).toNat%2^23 := by
  have he : Mufu.exponent (0x3f800000+index) = 127+index/2^23 := by dsimp [Mufu.exponent];omega
  have hs := RcpBounds.small_sign (0x3f800000+index) (by omega)
  have hmag : Mufu.magnitude (0x3f800000+index) ≤ 0x7f800000 := by dsimp [Mufu.magnitude];omega
  simp only [Mufu.sqrt,he,hs,root_index index hi,ite_eq_right (not_lt.mpr hmag)]
  simp [show 127+index/8388608 ≠ 255 by omega]

theorem output_decode (index : Nat) (hi : index < 2^24) :
    positiveNormalValue (Mufu.sqrt (0x3f800000+index)) = normalizedOutput index := by
  rw [output_word index hi]
  let z := (Mufu.interpolate MufuTables.sqrt index).toNat
  have hr := interpolate_error index hi
  have hb : 2^23 ≤ z ∧ z < 2^24 := by dsimp [z];omega
  have hz : (z : Int) = Mufu.interpolate MufuTables.sqrt index := by dsimp [z];omega
  have hm : z%2^23 = z-2^23 := by omega
  have he := RcpBounds.input_decode (z%2^23) (by omega)
  change (RcpBounds.positiveNormalValue (0x3f800000+z%2^23) : ℝ) = _
  rw [he]
  simp only [RcpBounds.normalizedInput,hm,normalizedOutput]
  have hsub : ((z-2^23 : Nat) : ℚ) = (z : ℚ)-2^23 := by rw [Nat.cast_sub hb.1];norm_num
  rw [hsub]
  have hzreal : (Mufu.interpolate MufuTables.sqrt index : ℝ) = z := by exact_mod_cast hz.symm
  rw [hzreal]
  push_cast
  ring

theorem sqrt_bits_absolute_error (index : Nat) (hi : index < 2^24) :
    |positiveNormalValue (Mufu.sqrt (0x3f800000+index))-
      Real.sqrt (positiveNormalValue (0x3f800000+index))| ≤ 1/2^22 := by
  rw [output_decode index hi,input_decode index hi]
  exact sqrt_absolute_error index hi

#print axioms sqrt_bits_absolute_error


def scaleExponent (exponent : Nat) : Int :=
  ((exponent : Int)-127-((exponent : Int)-127)%2)/2

def mantissaIndex (exponent fraction : Nat) : Nat :=
  (((exponent : Int)-127)%2).toNat*2^23+fraction

noncomputable def outputScale (exponent : Nat) : ℝ := (2 : ℝ)^scaleExponent exponent

theorem index_properties (exponent fraction : Nat) (hf : fraction < 2^23) :
    mantissaIndex exponent fraction < 2^24 ∧
    mantissaIndex exponent fraction%2^23 = fraction ∧
    mantissaIndex exponent fraction/2^23 = (((exponent : Int)-127)%2).toNat := by
  dsimp [mantissaIndex]
  omega

theorem input_decode_scaled (exponent fraction : Nat) (he : exponent < 256) (hf : fraction < 2^23) :
    positiveNormalValue (exponent*2^23+fraction) =
      normalizedInput (mantissaIndex exponent fraction)*outputScale exponent^2 := by
  have hp := index_properties exponent fraction hf
  unfold positiveNormalValue
  rw [RcpBounds.normal_decode exponent fraction he hf]
  push_cast
  dsimp [normalizedInput,outputScale]
  have hmod : mantissaIndex exponent fraction%8388608 = fraction := hp.2.1
  have hdiv : mantissaIndex exponent fraction/8388608 = (((exponent : Int)-127)%2).toNat := hp.2.2
  rw [hmod,hdiv]
  let p := ((exponent : Int)-127)%2
  have hp0 : 0 ≤ p := by dsimp [p];omega
  have hpt : (p.toNat : Int) = p := by omega
  have heq : (exponent : Int)-150 = -23+p+2*scaleExponent exponent := by
    dsimp [p,scaleExponent]
    omega
  have hpow : (2 : ℝ)^((exponent : Int)-150) =
      (2 : ℝ)^(-23 : Int)*(2 : ℝ)^p.toNat*((2 : ℝ)^scaleExponent exponent)^2 := by
    rw [← zpow_natCast (2 : ℝ) p.toNat, hpt, ← zpow_natCast ((2 : ℝ)^scaleExponent exponent) 2, ← zpow_mul, ← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0),
      ← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
    congr 1
    rw [heq]
    ring
  rw [hpow]
  norm_num
  ring

theorem output_word_scaled (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 254) (hf : fraction < 2^23) :
    Mufu.sqrt (exponent*2^23+fraction) =
      (127+scaleExponent exponent).toNat*2^23+
      (Mufu.interpolate MufuTables.sqrt (mantissaIndex exponent fraction)).toNat%2^23 := by
  have hex : Mufu.exponent (exponent*2^23+fraction) = exponent := by dsimp [Mufu.exponent];omega
  have hfr : Mufu.fraction (exponent*2^23+fraction) = fraction := by dsimp [Mufu.fraction];omega
  have hs := RcpBounds.small_sign (exponent*2^23+fraction) (by omega)
  have hmag : ¬Mufu.magnitude (exponent*2^23+fraction) > 0x7f800000 := by dsimp [Mufu.magnitude];omega
  have hr : Mufu.rootIndex (exponent*2^23+fraction) =
      (mantissaIndex exponent fraction,scaleExponent exponent) := by
    unfold Mufu.rootIndex
    rw [hex,hfr]
    rfl
  rw [Mufu.sqrt,ite_eq_right hmag]
  simp only [hex,hs]
  rw [ite_eq_right (show exponent ≠ 0 by omega),
    ite_eq_right (show ¬(0 : Nat) ≠ 0 by simp),ite_eq_right (show exponent ≠ 255 by omega)]
  rw [hr]

theorem output_decode_scaled (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 254) (hf : fraction < 2^23) :
    positiveNormalValue (Mufu.sqrt (exponent*2^23+fraction)) =
      normalizedOutput (mantissaIndex exponent fraction)*outputScale exponent := by
  have hp := index_properties exponent fraction hf
  have ht : -63 ≤ scaleExponent exponent ∧ scaleExponent exponent ≤ 63 := by dsimp [scaleExponent];omega
  have hec : (((127+scaleExponent exponent).toNat : Nat) : Int) = 127+scaleExponent exponent := by omega
  have hz := interpolate_error (mantissaIndex exponent fraction) hp.1
  generalize hv : Mufu.interpolate MufuTables.sqrt (mantissaIndex exponent fraction) = v at hz ⊢
  let z := v.toNat
  have hzb : 2^23 ≤ z ∧ z < 2^24 := by dsimp [z];omega
  have hzv : (z : Int) = v := by dsimp [z];omega
  rw [output_word_scaled exponent fraction he hf,hv]
  change positiveNormalValue ((127+scaleExponent exponent).toNat*2^23+z%2^23) = _
  unfold positiveNormalValue
  rw [RcpBounds.normal_decode _ _ (by omega) (by omega)]
  have hm : z%2^23 = z-2^23 := by omega
  have hsub : ((z-2^23 : Nat) : ℚ) = (z : ℚ)-2^23 := by rw [Nat.cast_sub hzb.1];norm_num
  rw [hm,hsub]
  push_cast
  rw [hec]
  have hpow : (2 : ℝ)^(127+scaleExponent exponent-150) =
      (2 : ℝ)^(-23 : Int)*(2 : ℝ)^scaleExponent exponent := by
    rw [← zpow_add₀ (by norm_num : (2 : ℝ) ≠ 0)]
    congr 1
    omega
  rw [hpow]
  have hzr : (v : ℝ) = z := by exact_mod_cast hzv.symm
  simp only [normalizedOutput,outputScale,hv,hzr]
  norm_num
  ring

theorem sqrt_bits_scaled_absolute_error (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 254) (hf : fraction < 2^23) :
    |positiveNormalValue (Mufu.sqrt (exponent*2^23+fraction))-
      Real.sqrt (positiveNormalValue (exponent*2^23+fraction))| ≤ outputScale exponent/2^22 := by
  have hp := index_properties exponent fraction hf
  have hs : 0 < outputScale exponent := by unfold outputScale; positivity
  rw [input_decode_scaled exponent fraction (by omega) hf,output_decode_scaled exponent fraction he hf,
    Real.sqrt_mul (show 0 ≤ normalizedInput (mantissaIndex exponent fraction) by unfold normalizedInput;positivity),
    Real.sqrt_sq (le_of_lt hs)]
  rw [← sub_mul,abs_mul,abs_of_pos hs]
  have h := mul_le_mul_of_nonneg_right (sqrt_absolute_error (mantissaIndex exponent fraction) hp.1) (le_of_lt hs)
  linarith

#print axioms sqrt_bits_scaled_absolute_error
end GB300Math.SqrtBounds
