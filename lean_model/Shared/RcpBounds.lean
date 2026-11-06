import GB300Math.Mufu
import Mathlib.Tactic

set_option maxRecDepth 10000

namespace GB300Math.RcpBounds

theorem pure_yield_ite (p : Prop) [Decidable p] (a b : Nat) :
    (if p then pure (ForInStep.yield a) else pure (ForInStep.yield b) : Id (ForInStep Nat)) =
    pure (ForInStep.yield (if p then a else b)) := by split <;> rfl

theorem accumulate_ite (p : Prop) [Decidable p] (a b : Nat) :
    (if p then a+b else a) = a+(if p then b else 0) := by split <;> simp_all

theorem accumulate_three_ite (p : Prop) [Decidable p] (a b c : Nat) :
    (if p then a+b+c else a) = a+(if p then b+c else 0) := by split <;> simp_all [Nat.add_assoc]

theorem square_drop_bound (k : Nat) : k*k - Fp64Seed.squareTruncated k 17 ≤ 1048577 := by
  have h0 : k &&& 65534 ≤ 65534 := Nat.and_le_right
  have h1 : k &&& 32764 ≤ 32764 := Nat.and_le_right
  have h2 : k &&& 16376 ≤ 16376 := Nat.and_le_right
  have h3 : k &&& 8176 ≤ 8176 := Nat.and_le_right
  have h4 : k &&& 4064 ≤ 4064 := Nat.and_le_right
  have h5 : k &&& 1984 ≤ 1984 := Nat.and_le_right
  have h6 : k &&& 896 ≤ 896 := Nat.and_le_right
  have h7 : k &&& 256 ≤ 256 := Nat.and_le_right
  simp only [Fp64Seed.squareTruncated, Std.Legacy.Range.forIn_eq_forIn_range']
  simp [Std.Legacy.Range.size, List.range', pure_yield_ite, Id.run]
  change k*k ≤ 1048577+(k*k-_)
  simp only [accumulate_three_ite, accumulate_ite]
  have b0 : (if k % 2 = 1 then 1 + (k &&& 65534) * 2 else 0) ≤ 131069 := by split <;> omega
  have b1 : (if k >>> 1 % 2 = 1 then 4 + (k &&& 32764) * 4 else 0) ≤ 131060 := by split <;> omega
  have b2 : (if k >>> 2 % 2 = 1 then 16 + (k &&& 16376) * 8 else 0) ≤ 131024 := by split <;> omega
  have b3 : (if k >>> 3 % 2 = 1 then 64 + (k &&& 8176) * 16 else 0) ≤ 130880 := by split <;> omega
  have b4 : (if k >>> 4 % 2 = 1 then 256 + (k &&& 4064) * 32 else 0) ≤ 130304 := by split <;> omega
  have b5 : (if k >>> 5 % 2 = 1 then 1024 + (k &&& 1984) * 64 else 0) ≤ 128000 := by split <;> omega
  have b6 : (if k >>> 6 % 2 = 1 then 4096 + (k &&& 896) * 128 else 0) ≤ 118784 := by split <;> omega
  have b7 : (if k >>> 7 % 2 = 1 then 16384 + (k &&& 256) * 256 else 0) ≤ 81920 := by split <;> omega
  have b8 : (if k >>> 8 % 2 = 1 then 65536 else 0) ≤ 65536 := by split <;> omega
  omega

def boxLo (c t : Int) := min 0 (c*t)
def boxHi (c t : Int) := max 0 (c*t)

theorem term_box (c x t : Int) (hx : 0 ≤ x) (ht : x ≤ t) :
    boxLo c t ≤ c*x ∧ c*x ≤ boxHi c t := by
  unfold boxLo boxHi
  by_cases hc : 0 ≤ c
  · constructor
    · exact le_trans (min_le_left _ _) (mul_nonneg hc hx)
    · exact le_trans (mul_le_mul_of_nonneg_left ht hc) (le_max_right _ _)
  · constructor
    · exact le_trans (min_le_right _ _) (mul_le_mul_of_nonpos_left ht (by omega))
    · exact le_trans (mul_nonpos_of_nonpos_of_nonneg (by omega) hx) (le_max_left _ _)

def lower (q0 q1 q2 q3 d : Int) := q0+boxLo q1 d+boxLo q2 (d*d)+boxLo q3 (d*d*d)
def upper (q0 q1 q2 q3 d : Int) := q0+boxHi q1 d+boxHi q2 (d*d)+boxHi q3 (d*d*d)

theorem polynomial_box (q0 q1 q2 q3 t d : Int) (ht : 0 ≤ t) (hd : t ≤ d) :
    lower q0 q1 q2 q3 d ≤ q0+q1*t+q2*t*t+q3*t*t*t ∧
    q0+q1*t+q2*t*t+q3*t*t*t ≤ upper q0 q1 q2 q3 d := by
  have h1 := term_box q1 t d ht hd
  have h2 := term_box q2 (t*t) (d*d) (mul_nonneg ht ht) (by nlinarith)
  have hcube : t*t*t ≤ d*d*d := by
    have hsq : t*t ≤ d*d := by nlinarith
    have hleft := mul_le_mul_of_nonneg_right hsq ht
    have hright := mul_le_mul_of_nonneg_left hd (mul_self_nonneg d)
    omega
  have h3 := term_box q3 (t*t*t) (d*d*d) (mul_nonneg (mul_nonneg ht ht) ht) hcube
  simp only [lower,upper]
  constructor <;> nlinarith

structure Parameters where
  (m a b c : Int)

def parameters (segment : Nat) : Parameters :=
  let (a,b,c) := MufuTables.rcp.coefficients[segment]!
  ⟨2^23+(segment : Int)*65536, a*2^32+1060634624, b*2^19, c*4⟩

def taylor (p : Parameters) (left : Int) : Int × Int × Int × Int :=
  let p0 := p.m*p.a - 2^47*2^34
  let p1 := p.a+p.m*p.b
  let p2 := p.b+p.m*p.c
  let p3 := p.c
  (p0+p1*left+p2*left^2+p3*left^3,
   p1+2*p2*left+3*p3*left^2,p2+3*p3*left,p3)

abbrev CellBound (p : Parameters) (left : Int) : Prop :=
  let (q0,q1,q2,q3) := taylor p left
  0 ≤ p.c ∧ 0 < p.m+left ∧
  -5*(p.m+left)*2^34+4*(p.m+left+4095)*(p.c*1048577+2^34-1) ≤ 4*lower q0 q1 q2 q3 4095 ∧
  4*upper q0 q1 q2 q3 4095 ≤ 5*(p.m+left)*2^34

abbrev CellValid (segment quarter : Nat) : Prop :=
  CellBound (parameters segment) ((quarter : Int)*4096)

set_option maxRecDepth 1000000 in
set_option maxHeartbeats 0 in
theorem cells_valid : ∀ segment : Fin 128, ∀ quarter : Fin 16,
    CellValid segment.val quarter.val := by decide +kernel

theorem cell_error (p : Parameters) (left k dropped : Int)
    (hcell : CellBound p left) (hk : 0 ≤ k-left ∧ k-left ≤ 4095)
    (hd : 0 ≤ dropped ∧ dropped ≤ 1048577) :
    let m := p.m+k
    let z := (p.a+p.b*k+p.c*(k*k-dropped))/2^34
    (-5*m ≤ 4*(m*z-2^47) ∧ 4*(m*z-2^47) ≤ 5*m) := by
  rcases ht : taylor p left with ⟨q0,q1,q2,q3⟩
  simp only [CellBound, ht] at hcell
  rcases hcell with ⟨hc, hm, hlo, hhi⟩
  have hbox := polynomial_box q0 q1 q2 q3 (k-left) 4095 hk.1 hk.2
  have heq : q0+q1*(k-left)+q2*(k-left)*(k-left)+q3*(k-left)*(k-left)*(k-left) =
      (p.m+k)*(p.a+p.b*k+p.c*k*k)-2^47*2^34 := by
    have vals := ht
    simp only [taylor, Prod.mk.injEq] at vals
    rcases vals with ⟨rfl,rfl,rfl,rfl⟩
    ring
  rw [heq] at hbox
  have hscaledLo := mul_le_mul_of_nonneg_left hbox.1 (by decide : (0 : Int) ≤ 4)
  have hscaledHi := mul_le_mul_of_nonneg_left hbox.2 (by decide : (0 : Int) ≤ 4)
  let n := p.a+p.b*k+p.c*(k*k-dropped)
  let z := n/2^34
  let r := n%2^34
  have hr : 0 ≤ r ∧ r ≤ 2^34-1 := by dsimp [r]; omega
  have hn : n = z*2^34+r := by dsimp [z,r]; omega
  have he : (p.m+k)*n = (p.m+k)*(p.a+p.b*k+p.c*k*k) - (p.m+k)*p.c*dropped := by
    dsimp [n]; ring
  have hdrop : 0 ≤ p.c*dropped+r ∧ p.c*dropped+r ≤ p.c*1048577+2^34-1 := by
    have hh := mul_le_mul_of_nonneg_left hd.2 hc
    have hl := mul_nonneg hc hd.1
    omega
  have hloss : 0 ≤ (p.m+k)*(p.c*dropped+r) := mul_nonneg (by omega) hdrop.1
  have hlossmax : (p.m+k)*(p.c*dropped+r) ≤
      (p.m+left+4095)*(p.c*1048577+2^34-1) := by
    apply mul_le_mul (by omega) hdrop.2 hdrop.1 (by omega)
  have hid : ((p.m+k)*z-2^47)*2^34 =
      (p.m+k)*(p.a+p.b*k+p.c*k*k)-2^47*2^34-(p.m+k)*(p.c*dropped+r) := by
    calc
      _ = (p.m+k)*(z*2^34+r)-2^47*2^34-(p.m+k)*r := by ring
      _ = (p.m+k)*n-2^47*2^34-(p.m+k)*r := by rw [hn]
      _ = _ := by rw [he]; ring
  change -5*(p.m+k) ≤ 4*((p.m+k)*z-2^47) ∧ 4*((p.m+k)*z-2^47) ≤ 5*(p.m+k)
  constructor <;> nlinarith only [hid, hscaledLo, hscaledHi, hloss, hlossmax, hlo, hhi, hk.1]

theorem interpolate_error (fraction : Nat) (hf : fraction < 2^23) :
    let m : Int := 2^23+fraction
    let z := Mufu.interpolate MufuTables.rcp fraction
    (-5*m ≤ 4*(m*z-2^47) ∧ 4*(m*z-2^47) ≤ 5*m) := by
  let segment : Fin 128 := ⟨fraction/65536, by omega⟩
  let k := fraction%65536
  let quarter : Fin 16 := ⟨k/4096, by dsimp [k]; omega⟩
  let drop := k*k-Fp64Seed.squareTruncated k 17
  have hs : Fp64Seed.squareTruncated k 17 ≤ k*k := by
    unfold Fp64Seed.squareTruncated
    exact Nat.sub_le _ _
  have hdrop : 0 ≤ (drop : Int) ∧ (drop : Int) ≤ 1048577 := by
    constructor
    · exact Int.natCast_nonneg _
    · exact_mod_cast square_drop_bound k
  have hwindow : 0 ≤ (k : Int)-(quarter.val : Int)*4096 ∧
      (k : Int)-(quarter.val : Int)*4096 ≤ 4095 := by
    dsimp [quarter]; omega
  have result := cell_error (parameters segment.val) ((quarter.val : Int)*4096)
    k drop (cells_valid segment quarter) hwindow hdrop
  have hm : (parameters segment.val).m+(k : Int) = 2^23+fraction := by
    dsimp [parameters,segment,k]
    omega
  have hsquare : (k : Int)*(k : Int)-(drop : Int) = Fp64Seed.squareTruncated k 17 := by
    have h := Nat.sub_add_cancel hs
    have hc : (drop : Int)+(Fp64Seed.squareTruncated k 17 : Int) = (k : Int)*(k : Int) := by
      exact_mod_cast h
    omega
  have hz : (parameters segment.val).a+(parameters segment.val).b*(k : Int)+
      (parameters segment.val).c*((k : Int)*(k : Int)-(drop : Int)) =
      (MufuTables.rcp.coefficients[fraction/65536]!).1*2^32+1060634624+
      (MufuTables.rcp.coefficients[fraction/65536]!).2.1*(k : Int)*2^19+
      (MufuTables.rcp.coefficients[fraction/65536]!).2.2*(Fp64Seed.squareTruncated k 17 : Int)*4 := by
    rw [hsquare]
    dsimp [parameters,segment]
    ring
  dsimp only at result ⊢
  rw [hm,hz] at result
  convert result using 1 <;>
    dsimp [Mufu.interpolate,MufuTables.rcp,k] <;> norm_num

abbrev RangeValid (p : Parameters) : Prop :=
  0 ≤ p.c ∧ p.b+131070*p.c ≤ 0 ∧
  2^23*2^34 ≤ p.a+p.b*65535+p.c*65535^2-p.c*1048577 ∧
  p.a < (2^24+1)*2^34

set_option maxRecDepth 1000000 in
set_option maxHeartbeats 0 in
theorem ranges_valid : ∀ segment : Fin 128, RangeValid (parameters segment.val) := by
  decide +kernel

theorem polynomial_range (p : Parameters) (k dropped : Int)
    (hp : RangeValid p) (hk : 0 ≤ k ∧ k ≤ 65535)
    (hd : 0 ≤ dropped ∧ dropped ≤ 1048577) :
    2^23 ≤ (p.a+p.b*k+p.c*(k*k-dropped))/2^34 ∧
    (p.a+p.b*k+p.c*(k*k-dropped))/2^34 ≤ 2^24 := by
  rcases hp with ⟨hc,hb,hlo,hhi⟩
  have hck := mul_le_mul_of_nonneg_left hk.2 hc
  have hc2 := mul_le_mul_of_nonneg_left (show k+65535 ≤ 131070 by omega) hc
  have hupper := mul_nonpos_of_nonpos_of_nonneg (show p.b+p.c*k ≤ 0 by nlinarith) hk.1
  have hlower := mul_nonneg (show 0 ≤ 65535-k by omega)
    (show 0 ≤ -p.b-p.c*(65535+k) by nlinarith)
  have hdrop0 := mul_nonneg hc hd.1
  have hdrop1 := mul_le_mul_of_nonneg_left hd.2 hc
  have hnlo : 2^23*2^34 ≤ p.a+p.b*k+p.c*(k*k-dropped) := by nlinarith
  have hnhi : p.a+p.b*k+p.c*(k*k-dropped) < (2^24+1)*2^34 := by nlinarith
  constructor <;> omega

theorem interpolate_range (fraction : Nat) (hf : fraction < 2^23) :
    2^23 ≤ Mufu.interpolate MufuTables.rcp fraction ∧
    Mufu.interpolate MufuTables.rcp fraction ≤ 2^24 := by
  let segment : Fin 128 := ⟨fraction/65536, by omega⟩
  let k := fraction%65536
  let drop := k*k-Fp64Seed.squareTruncated k 17
  have hs : Fp64Seed.squareTruncated k 17 ≤ k*k := by
    unfold Fp64Seed.squareTruncated
    exact Nat.sub_le _ _
  have hd : 0 ≤ (drop : Int) ∧ (drop : Int) ≤ 1048577 := by
    constructor
    · exact Int.natCast_nonneg _
    · exact_mod_cast square_drop_bound k
  have hk : 0 ≤ (k : Int) ∧ (k : Int) ≤ 65535 := by dsimp [k]; omega
  have h := polynomial_range (parameters segment.val) k drop (ranges_valid segment) hk hd
  have hsquare : (k : Int)*(k : Int)-(drop : Int) = Fp64Seed.squareTruncated k 17 := by
    have hh := Nat.sub_add_cancel hs
    have hc : (drop : Int)+(Fp64Seed.squareTruncated k 17 : Int) = (k : Int)*(k : Int) := by
      exact_mod_cast hh
    omega
  rw [hsquare] at h
  convert h using 1 <;>
    dsimp [Mufu.interpolate,MufuTables.rcp,parameters,segment,k] <;> norm_num <;> ring_nf

def normalizedInput (fraction : Nat) : ℚ := (2^23+(fraction : ℚ))/2^23
def normalizedOutput (fraction : Nat) : ℚ := (Mufu.interpolate MufuTables.rcp fraction : ℚ)/2^24

theorem reciprocal_absolute_error (fraction : Nat) (hf : fraction < 2^23) :
    |normalizedOutput fraction - 1/normalizedInput fraction| ≤ 5/2^26 := by
  have h := interpolate_error fraction hf
  dsimp only at h
  have hlo : -(5 : ℚ)*(2^23+fraction) ≤
      4*((2^23+(fraction : ℚ))*(Mufu.interpolate MufuTables.rcp fraction : ℚ)-2^47) := by
    exact_mod_cast h.1
  have hhi : 4*((2^23+(fraction : ℚ))*(Mufu.interpolate MufuTables.rcp fraction : ℚ)-2^47) ≤
      5*(2^23+(fraction : ℚ)) := by exact_mod_cast h.2
  have hm : (0 : ℚ) < 2^23+(fraction : ℚ) := by positivity
  have he : normalizedOutput fraction - 1/normalizedInput fraction =
      (4*((2^23+(fraction : ℚ))*(Mufu.interpolate MufuTables.rcp fraction : ℚ)-2^47))/
      ((2^23+(fraction : ℚ))*2^26) := by
    dsimp [normalizedOutput,normalizedInput]
    field_simp
    ring
  rw [he,abs_div,abs_of_pos (mul_pos hm (by norm_num))]
  calc
    _ ≤ (5*(2^23+(fraction : ℚ)))/((2^23+(fraction : ℚ))*2^26) :=
      div_le_div_of_nonneg_right (abs_le.mpr ⟨by linarith,hhi⟩) (by positivity)
    _ = _ := by field_simp

def positiveNormalValue (word : Nat) : ℚ :=
  (2^23+(Mufu.fraction word : ℚ))*(2 : ℚ)^((Mufu.exponent word : Int)-150)

theorem input_decode (fraction : Nat) (hf : fraction < 2^23) :
    positiveNormalValue (0x3f800000+fraction) = normalizedInput fraction := by
  have he : Mufu.exponent (0x3f800000+fraction) = 127 := by dsimp [Mufu.exponent]; omega
  have hm : Mufu.fraction (0x3f800000+fraction) = fraction := by dsimp [Mufu.fraction]; omega
  simp [positiveNormalValue,he,hm,normalizedInput]
  ring

theorem small_sign (word : Nat) (h : word < 2^31) : Mufu.sign word = 0 := by
  have hl : word &&& 2^31 ≤ word := Nat.and_le_left
  have hm := Nat.and_mod_two_pow (a := word) (b := 2^31) (n := 31)
  rw [Nat.mod_eq_of_lt (show word &&& 2^31 < 2^31 by omega)] at hm
  norm_num at hm
  exact hm

theorem output_word (fraction : Nat) (hf : fraction < 2^23) :
    Mufu.rcp (0x3f800000+fraction) =
      if (Mufu.interpolate MufuTables.rcp fraction).toNat = 2^24 then 0x3f800000
      else 126*2^23+(Mufu.interpolate MufuTables.rcp fraction).toNat % 2^23 := by
  have he : Mufu.exponent (0x3f800000+fraction) = 127 := by dsimp [Mufu.exponent]; omega
  have hm : Mufu.fraction (0x3f800000+fraction) = fraction := by dsimp [Mufu.fraction]; omega
  have hs := small_sign (0x3f800000+fraction) (by omega)
  have hmag : Mufu.magnitude (0x3f800000+fraction) ≤ 0x7f800000 := by
    dsimp [Mufu.magnitude]; omega
  by_cases h : (Mufu.interpolate MufuTables.rcp fraction).toNat = 2^24
  · simp [Mufu.rcp,he,hm,hs,not_lt.mpr hmag,h]
  · norm_num at h
    simp [Mufu.rcp,he,hm,hs,not_lt.mpr hmag,h]

theorem output_decode (fraction : Nat) (hf : fraction < 2^23) :
    positiveNormalValue (Mufu.rcp (0x3f800000+fraction)) = normalizedOutput fraction := by
  rw [output_word fraction hf]
  have hr := interpolate_range fraction hf
  let z := (Mufu.interpolate MufuTables.rcp fraction).toNat
  have hz : (z : Int) = Mufu.interpolate MufuTables.rcp fraction := by
    dsimp [z]; omega
  have hb : 2^23 ≤ z ∧ z ≤ 2^24 := by omega
  change positiveNormalValue (if z = 2^24 then _ else _) = _
  split
  · rename_i h
    have hi : Mufu.interpolate MufuTables.rcp fraction = 16777216 := by omega
    norm_num [positiveNormalValue,Mufu.exponent,Mufu.fraction,normalizedOutput,hi]
  · rename_i h
    have he : Mufu.exponent (126*2^23+z%2^23) = 126 := by dsimp [Mufu.exponent]; omega
    have hm : Mufu.fraction (126*2^23+z%2^23) = z-2^23 := by dsimp [Mufu.fraction]; omega
    have hzq : (Mufu.interpolate MufuTables.rcp fraction : ℚ) = z := by exact_mod_cast hz.symm
    have hsub : ((z-2^23 : Nat) : ℚ) = (z : ℚ)-2^23 := by
      rw [Nat.cast_sub hb.1]; norm_num
    change positiveNormalValue (126*2^23+z%2^23) = _
    unfold positiveNormalValue
    rw [he,hm]
    simp only [normalizedOutput,hzq,hsub]
    norm_num
    ring

theorem rcp_bits_absolute_error (fraction : Nat) (hf : fraction < 2^23) :
    |positiveNormalValue (Mufu.rcp (0x3f800000+fraction)) -
      1/positiveNormalValue (0x3f800000+fraction)| ≤ 5/2^26 := by
  rw [output_decode fraction hf,input_decode fraction hf]
  exact reciprocal_absolute_error fraction hf

def exponentScale (exponent : Nat) : ℚ := (2 : ℚ)^((exponent : Int)-127)

theorem normal_decode (exponent fraction : Nat) (he : exponent < 256) (hf : fraction < 2^23) :
    positiveNormalValue (exponent*2^23+fraction) =
    (2^23+(fraction : ℚ))*(2 : ℚ)^((exponent : Int)-150) := by
  have hh : Mufu.exponent (exponent*2^23+fraction) = exponent := by dsimp [Mufu.exponent]; omega
  have hm : Mufu.fraction (exponent*2^23+fraction) = fraction := by dsimp [Mufu.fraction];omega
  simp only [positiveNormalValue,hh,hm]

theorem input_decode_scaled (exponent fraction : Nat) (he : exponent < 256) (hf : fraction < 2^23) :
    positiveNormalValue (exponent*2^23+fraction) = normalizedInput fraction*exponentScale exponent := by
  rw [normal_decode exponent fraction he hf]
  dsimp [normalizedInput,exponentScale]
  have hpow : (2 : ℚ)^((exponent : Int)-150) =
      (2 : ℚ)^(-23 : Int)*(2 : ℚ)^((exponent : Int)-127) := by
    rw [← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0)]
    congr 1
    omega
  rw [hpow]
  norm_num
  ring

theorem output_word_scaled (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 252) (hf : fraction < 2^23) :
    Mufu.rcp (exponent*2^23+fraction) =
      if (Mufu.interpolate MufuTables.rcp fraction).toNat = 2^24
      then (254-exponent)*2^23
      else (253-exponent)*2^23+(Mufu.interpolate MufuTables.rcp fraction).toNat % 2^23 := by
  have hh : Mufu.exponent (exponent*2^23+fraction) = exponent := by dsimp [Mufu.exponent];omega
  have hm : Mufu.fraction (exponent*2^23+fraction) = fraction := by dsimp [Mufu.fraction];omega
  have hs := small_sign (exponent*2^23+fraction) (by omega)
  have hmag : Mufu.magnitude (exponent*2^23+fraction) ≤ 0x7f800000 := by dsimp [Mufu.magnitude];omega
  simp only [Mufu.rcp,hh,hm,hs,ite_eq_right (not_lt.mpr hmag)]
  by_cases h : (Mufu.interpolate MufuTables.rcp fraction).toNat = 2^24
  · have hpos : (0 : Int) < 254-exponent := by omega
    have hc : (254-(exponent : Int)).toNat = 254-exponent := by omega
    simp [h,show exponent ≠ 0 by omega,show exponent ≠ 255 by omega,not_le.mpr hpos,hc]
  · norm_num at h
    have hpos : (0 : Int) < 253-exponent := by omega
    have hc : (253-(exponent : Int)).toNat = 253-exponent := by omega
    simp [h,show exponent ≠ 0 by omega,show exponent ≠ 255 by omega,not_le.mpr hpos,hc]

theorem output_decode_scaled (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 252) (hf : fraction < 2^23) :
    positiveNormalValue (Mufu.rcp (exponent*2^23+fraction))*exponentScale exponent =
      normalizedOutput fraction := by
  rw [output_word_scaled exponent fraction he hf]
  have hr := interpolate_range fraction hf
  let z := (Mufu.interpolate MufuTables.rcp fraction).toNat
  have hz : (z : Int) = Mufu.interpolate MufuTables.rcp fraction := by dsimp [z];omega
  have hb : 2^23 ≤ z ∧ z ≤ 2^24 := by omega
  by_cases h : z = 2^24
  · have hh : (Mufu.interpolate MufuTables.rcp fraction).toNat = 2^24 := h
    rw [ite_eq_left hh]
    have hi : Mufu.interpolate MufuTables.rcp fraction = 16777216 := by omega
    have hh0 := normal_decode (254-exponent) 0 (by omega) (by norm_num)
    simp only [Nat.add_zero] at hh0
    rw [hh0]
    dsimp [normalizedOutput,exponentScale]
    rw [hi]
    have hexp : ((254-exponent : Nat) : Int)-150+((exponent : Int)-127) = -23 := by omega
    rw [mul_assoc,← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0),hexp]
    norm_num
  · have hh : (Mufu.interpolate MufuTables.rcp fraction).toNat ≠ 2^24 := h
    rw [ite_eq_right hh]
    change positiveNormalValue ((253-exponent)*2^23+z%2^23)*exponentScale exponent = _
    rw [normal_decode (253-exponent) (z%2^23) (by omega) (by omega)]
    have hm : z%2^23 = z-2^23 := by omega
    have hsub : ((z-2^23 : Nat) : ℚ) = (z : ℚ)-2^23 := by rw [Nat.cast_sub hb.1];norm_num
    have hzq : (Mufu.interpolate MufuTables.rcp fraction : ℚ) = z := by exact_mod_cast hz.symm
    rw [hm,hsub]
    dsimp [exponentScale,normalizedOutput]
    rw [hzq]
    have hexp : ((253-exponent : Nat) : Int)-150+((exponent : Int)-127) = -24 := by omega
    rw [mul_assoc,← zpow_add₀ (by norm_num : (2 : ℚ) ≠ 0),hexp]
    norm_num
    ring

theorem normalized_relative_error (fraction : Nat) (hf : fraction < 2^23) :
    |normalizedOutput fraction*normalizedInput fraction-1| ≤ 5/2^25 := by
  have habs := reciprocal_absolute_error fraction hf
  have hx : 1 ≤ normalizedInput fraction ∧ normalizedInput fraction < 2 := by
    dsimp [normalizedInput]
    have hfn : (0 : ℚ) ≤ fraction := by positivity
    have hfu : (fraction : ℚ) < 8388608 := by exact_mod_cast hf
    norm_num
    constructor <;> linarith
  have hne : normalizedInput fraction ≠ 0 := by linarith [hx.1]
  have he : normalizedOutput fraction*normalizedInput fraction-1 =
      (normalizedOutput fraction-1/normalizedInput fraction)*normalizedInput fraction := by
    field_simp
  rw [he,abs_mul,abs_of_pos (show 0 < normalizedInput fraction by linarith [hx.1])]
  have hm := mul_le_mul_of_nonneg_right habs (show 0 ≤ normalizedInput fraction by linarith [hx.1])
  norm_num at hm ⊢
  linarith [hx.2]

theorem rcp_bits_relative_error (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 252) (hf : fraction < 2^23) :
    |positiveNormalValue (Mufu.rcp (exponent*2^23+fraction))*
      positiveNormalValue (exponent*2^23+fraction)-1| ≤ 5/2^25 := by
  have ho := output_decode_scaled exponent fraction he hf
  rw [input_decode_scaled exponent fraction (by omega) hf]
  have hid : positiveNormalValue (Mufu.rcp (exponent*2^23+fraction))*
      (normalizedInput fraction*exponentScale exponent) =
      normalizedOutput fraction*normalizedInput fraction := by
    calc
      _ = (positiveNormalValue (Mufu.rcp (exponent*2^23+fraction))*exponentScale exponent)*normalizedInput fraction := by ring
      _ = _ := by rw [ho]
  rw [hid]
  exact normalized_relative_error fraction hf

theorem rcp_bits_scaled_absolute_error (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 252) (hf : fraction < 2^23) :
    |positiveNormalValue (Mufu.rcp (exponent*2^23+fraction))-
      1/positiveNormalValue (exponent*2^23+fraction)| ≤ (5/2^26)/exponentScale exponent := by
  have ho := output_decode_scaled exponent fraction he hf
  have hp : 0 < exponentScale exponent := by unfold exponentScale; positivity
  have hout := (eq_div_iff (ne_of_gt hp)).mpr ho
  rw [hout,input_decode_scaled exponent fraction (by omega) hf]
  have hid : normalizedOutput fraction/exponentScale exponent-
      1/(normalizedInput fraction*exponentScale exponent) =
      (normalizedOutput fraction-1/normalizedInput fraction)/exponentScale exponent := by ring
  rw [hid,abs_div,abs_of_pos hp]
  exact div_le_div_of_nonneg_right (reciprocal_absolute_error fraction hf) (le_of_lt hp)

def normalValue (word : Nat) : ℚ :=
  if word < 2^31 then positiveNormalValue word else -positiveNormalValue word

theorem value_sign (word : Nat) (hw : word < 2^31) :
    normalValue word = positiveNormalValue word ∧
    normalValue (2^31+word) = -positiveNormalValue word := by
  have he : Mufu.exponent (2^31+word) = Mufu.exponent word := by dsimp [Mufu.exponent];omega
  have hf : Mufu.fraction (2^31+word) = Mufu.fraction word := by dsimp [Mufu.fraction];omega
  simp only [normalValue,ite_eq_left hw,show ¬2^31+word < 2^31 by omega,ite_false]
  simp only [positiveNormalValue,he,hf,and_self]

theorem high_sign (word : Nat) (hw : word < 2^31) : Mufu.sign (2^31+word) = 2^31 := by
  have hl : (2^31+word) &&& 2^31 ≤ 2^31 := Nat.and_le_right
  have hh := Nat.and_div_two_pow (a := 2^31+word) (b := 2^31) (n := 31)
  have hd : (2^31+word)/2^31 = 1 := by omega
  rw [hd] at hh
  norm_num at hh
  dsimp [Mufu.sign]
  norm_num at hl
  omega

theorem reciprocal_sign (word : Nat) (hw : word ≤ 0x7f800000) :
    Mufu.rcp (2^31+word) = 2^31+Mufu.rcp word := by
  have hs0 := small_sign word (by omega)
  have hs1 := high_sign word (by omega)
  have he : Mufu.exponent (2^31+word) = Mufu.exponent word := by dsimp [Mufu.exponent];omega
  have hf : Mufu.fraction (2^31+word) = Mufu.fraction word := by dsimp [Mufu.fraction];omega
  have hm : Mufu.magnitude (2^31+word) = Mufu.magnitude word := by dsimp [Mufu.magnitude];omega
  have hm0 : ¬Mufu.magnitude word > 0x7f800000 := by dsimp [Mufu.magnitude];omega
  simp only [Mufu.rcp,he,hf,hm,hs0,hs1,ite_eq_right hm0]
  by_cases h0 : Mufu.exponent word = 0
  · simp [h0]
  · simp only [ite_eq_right h0,Nat.zero_add]
    by_cases h1 : Mufu.exponent word = 255
    · simp [h1]
    · simp only [ite_eq_right h1]
      split_ifs <;> simp_all [Nat.add_assoc]

theorem rcp_signed_relative_error (negative : Bool) (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 252) (hf : fraction < 2^23) :
    let word := (if negative then 2^31 else 0)+exponent*2^23+fraction
    |normalValue (Mufu.rcp word)*normalValue word-1| ≤ 5/2^25 := by
  let x := exponent*2^23+fraction
  have hx : x < 2^31 := by dsimp [x];omega
  have hy : Mufu.rcp x < 2^31 := by
    change Mufu.rcp (exponent*2^23+fraction) < 2^31
    rw [output_word_scaled exponent fraction he hf]
    generalize (Mufu.interpolate MufuTables.rcp fraction).toNat = z
    clear negative hx x hf fraction
    split <;> omega
  have vx := value_sign x hx
  have vy := value_sign (Mufu.rcp x) hy
  cases negative
  · simp only [Bool.false_eq_true,↓reduceIte,Nat.zero_add]
    change |normalValue (Mufu.rcp x)*normalValue x-1| ≤ _
    rw [vx.1,vy.1]
    exact rcp_bits_relative_error exponent fraction he hf
  · simp only [↓reduceIte]
    have hw : 2^31+exponent*2^23+fraction = 2^31+x := by dsimp [x];omega
    rw [hw,reciprocal_sign x (by dsimp [x];omega),vx.2,vy.2]
    simp only [neg_mul_neg]
    exact rcp_bits_relative_error exponent fraction he hf

theorem rcp_signed_scaled_absolute_error (negative : Bool) (exponent fraction : Nat)
    (he : 1 ≤ exponent ∧ exponent ≤ 252) (hf : fraction < 2^23) :
    let word := (if negative then 2^31 else 0)+exponent*2^23+fraction
    |normalValue (Mufu.rcp word)-1/normalValue word| ≤ (5/2^26)/exponentScale exponent := by
  let x := exponent*2^23+fraction
  have hx : x < 2^31 := by dsimp [x];omega
  have hy : Mufu.rcp x < 2^31 := by
    change Mufu.rcp (exponent*2^23+fraction) < 2^31
    rw [output_word_scaled exponent fraction he hf]
    generalize (Mufu.interpolate MufuTables.rcp fraction).toNat = z
    clear negative hx x hf fraction
    split <;> omega
  have vx := value_sign x hx
  have vy := value_sign (Mufu.rcp x) hy
  cases negative
  · simp only [Bool.false_eq_true,↓reduceIte,Nat.zero_add]
    change |normalValue (Mufu.rcp x)-1/normalValue x| ≤ _
    rw [vx.1,vy.1]
    exact rcp_bits_scaled_absolute_error exponent fraction he hf
  · simp only [↓reduceIte]
    have hw : 2^31+exponent*2^23+fraction = 2^31+x := by dsimp [x];omega
    rw [hw,reciprocal_sign x (by dsimp [x];omega),vx.2,vy.2]
    rw [one_div,inv_neg,sub_neg_eq_add]
    have hn : -positiveNormalValue (Mufu.rcp x)+(positiveNormalValue x)⁻¹ =
        -(positiveNormalValue (Mufu.rcp x)-(positiveNormalValue x)⁻¹) := by ring
    rw [hn,abs_neg]
    simpa only [one_div] using rcp_bits_scaled_absolute_error exponent fraction he hf

#print axioms rcp_bits_absolute_error
#print axioms rcp_bits_relative_error
#print axioms rcp_bits_scaled_absolute_error
#print axioms rcp_signed_relative_error
#print axioms rcp_signed_scaled_absolute_error
end GB300Math.RcpBounds
