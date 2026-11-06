import RcpBounds

set_option maxRecDepth 1000000
set_option maxHeartbeats 0
namespace GB300Math.RcpPtxBounds
open RcpBounds

def maskBound (block mask : Nat) := 512*(block &&& (mask/512))+mask%512

theorem mask_bound (k mask : Nat) : k &&& mask ≤ maskBound (k/512) mask := by
  have hdiv := Nat.and_div_two_pow (a := k) (b := mask) (n := 9)
  have hmod := Nat.and_mod_two_pow (a := k) (b := mask) (n := 9)
  have hlo : (k%512) &&& (mask%512) ≤ mask%512 := Nat.and_le_right
  norm_num at hdiv hmod
  unfold maskBound
  omega

def dropLimit (block : Nat) : Nat :=
  1 + maskBound block 65534*2 +
  4 + maskBound block 32764*4 +
  16 + maskBound block 16376*8 +
  64 + maskBound block 8176*16 +
  256 + maskBound block 4064*32 +
  1024 + maskBound block 1984*64 +
  4096 + maskBound block 896*128 +
  16384 + maskBound block 256*256 + 65536

theorem square_drop_local_bound (k : Nat) :
    k*k - Fp64Seed.squareTruncated k 17 ≤ dropLimit (k/512) := by
  have h0 := mask_bound k 65534
  have h1 := mask_bound k 32764
  have h2 := mask_bound k 16376
  have h3 := mask_bound k 8176
  have h4 := mask_bound k 4064
  have h5 := mask_bound k 1984
  have h6 := mask_bound k 896
  have h7 := mask_bound k 256
  simp only [Fp64Seed.squareTruncated, Std.Legacy.Range.forIn_eq_forIn_range']
  simp [Std.Legacy.Range.size, List.range', pure_yield_ite, Id.run]
  change k*k ≤ dropLimit (k/512)+(k*k-_)
  simp only [accumulate_three_ite, accumulate_ite]
  have b0 : (if k % 2 = 1 then 1 + (k &&& 65534) * 2 else 0) ≤ 1+maskBound (k/512) 65534*2 := by split <;> omega
  have b1 : (if k >>> 1 % 2 = 1 then 4 + (k &&& 32764) * 4 else 0) ≤ 4+maskBound (k/512) 32764*4 := by split <;> omega
  have b2 : (if k >>> 2 % 2 = 1 then 16 + (k &&& 16376) * 8 else 0) ≤ 16+maskBound (k/512) 16376*8 := by split <;> omega
  have b3 : (if k >>> 3 % 2 = 1 then 64 + (k &&& 8176) * 16 else 0) ≤ 64+maskBound (k/512) 8176*16 := by split <;> omega
  have b4 : (if k >>> 4 % 2 = 1 then 256 + (k &&& 4064) * 32 else 0) ≤ 256+maskBound (k/512) 4064*32 := by split <;> omega
  have b5 : (if k >>> 5 % 2 = 1 then 1024 + (k &&& 1984) * 64 else 0) ≤ 1024+maskBound (k/512) 1984*64 := by split <;> omega
  have b6 : (if k >>> 6 % 2 = 1 then 4096 + (k &&& 896) * 128 else 0) ≤ 4096+maskBound (k/512) 896*128 := by split <;> omega
  have b7 : (if k >>> 7 % 2 = 1 then 16384 + (k &&& 256) * 256 else 0) ≤ 16384+maskBound (k/512) 256*256 := by split <;> omega
  have b8 : (if k >>> 8 % 2 = 1 then 65536 else 0) ≤ 65536 := by split <;> omega
  unfold dropLimit
  omega
abbrev CellBound (p : Parameters) (left cap : Int) : Prop :=
  let (q0,q1,q2,q3) := taylor p left
  0 ≤ p.c ∧ 0 < p.m+left ∧
  -(p.m+left)*2^34+(p.m+left+511)*(p.c*cap+2^34-1) ≤ lower q0 q1 q2 q3 511 ∧
  upper q0 q1 q2 q3 511 ≤ (p.m+left)*2^34

abbrev CellValid (segment quarter : Nat) : Prop :=
  CellBound (parameters segment) ((quarter : Int)*512) (dropLimit quarter)

theorem cell_error (p : Parameters) (left k dropped cap : Int)
    (hcell : CellBound p left cap) (hk : 0 ≤ k-left ∧ k-left ≤ 511)
    (hd : 0 ≤ dropped ∧ dropped ≤ cap) :
    let m := p.m+k
    let z := (p.a+p.b*k+p.c*(k*k-dropped))/2^34
    (-m ≤ m*z-2^47 ∧ m*z-2^47 ≤ m) := by
  rcases ht : taylor p left with ⟨q0,q1,q2,q3⟩
  simp only [CellBound, ht] at hcell
  rcases hcell with ⟨hc, hm, hlo, hhi⟩
  have hbox := polynomial_box q0 q1 q2 q3 (k-left) 511 hk.1 hk.2
  have heq : q0+q1*(k-left)+q2*(k-left)*(k-left)+q3*(k-left)*(k-left)*(k-left) =
      (p.m+k)*(p.a+p.b*k+p.c*k*k)-2^47*2^34 := by
    have vals := ht
    simp only [taylor, Prod.mk.injEq] at vals
    rcases vals with ⟨rfl,rfl,rfl,rfl⟩
    ring
  rw [heq] at hbox
  have hscaledLo := hbox.1
  have hscaledHi := hbox.2
  let n := p.a+p.b*k+p.c*(k*k-dropped)
  let z := n/2^34
  let r := n%2^34
  have hr : 0 ≤ r ∧ r ≤ 2^34-1 := by dsimp [r]; omega
  have hn : n = z*2^34+r := by dsimp [z,r]; omega
  have he : (p.m+k)*n = (p.m+k)*(p.a+p.b*k+p.c*k*k) - (p.m+k)*p.c*dropped := by
    dsimp [n]; ring
  have hdrop : 0 ≤ p.c*dropped+r ∧ p.c*dropped+r ≤ p.c*cap+2^34-1 := by
    have hh := mul_le_mul_of_nonneg_left hd.2 hc
    have hl := mul_nonneg hc hd.1
    omega
  have hloss : 0 ≤ (p.m+k)*(p.c*dropped+r) := mul_nonneg (by omega) hdrop.1
  have hlossmax : (p.m+k)*(p.c*dropped+r) ≤
      (p.m+left+511)*(p.c*cap+2^34-1) := by
    apply mul_le_mul (by omega) hdrop.2 hdrop.1 (by omega)
  have hid : ((p.m+k)*z-2^47)*2^34 =
      (p.m+k)*(p.a+p.b*k+p.c*k*k)-2^47*2^34-(p.m+k)*(p.c*dropped+r) := by
    calc
      _ = (p.m+k)*(z*2^34+r)-2^47*2^34-(p.m+k)*r := by ring
      _ = (p.m+k)*n-2^47*2^34-(p.m+k)*r := by rw [hn]
      _ = _ := by rw [he]; ring
  change -(p.m+k) ≤ (p.m+k)*z-2^47 ∧ (p.m+k)*z-2^47 ≤ p.m+k
  constructor <;> nlinarith only [hid, hscaledLo, hscaledHi, hloss, hlossmax, hlo, hhi, hk.1]



abbrev ExactCell (segment quarter : Nat) : Prop := ∀ remainder : Fin 512,
  let f := segment*65536+quarter*512+remainder.val
  let m : Int := 2^23+f
  let z := Mufu.interpolate MufuTables.rcp f
  (-m ≤ m*z-2^47 ∧ m*z-2^47 ≤ m)

abbrev Certificate (segment quarter : Nat) : Prop :=
  CellValid segment quarter ∨ ExactCell segment quarter

 theorem cells_valid : ∀ segment : Fin 128, ∀ quarter : Fin 128,
    Certificate segment.val quarter.val := by decide +kernel

theorem interpolate_error (fraction : Nat) (hf : fraction < 2^23) :
    let m : Int := 2^23+fraction
    let z := Mufu.interpolate MufuTables.rcp fraction
    (-m ≤ m*z-2^47 ∧ m*z-2^47 ≤ m) := by
  let segment : Fin 128 := ⟨fraction/65536, by omega⟩
  let k := fraction%65536
  let quarter : Fin 128 := ⟨k/512, by dsimp [k]; omega⟩
  let drop := k*k-Fp64Seed.squareTruncated k 17
  have hs : Fp64Seed.squareTruncated k 17 ≤ k*k := by
    unfold Fp64Seed.squareTruncated
    exact Nat.sub_le _ _
  have hdrop : 0 ≤ (drop : Int) ∧ (drop : Int) ≤ dropLimit quarter.val := by
    constructor
    · exact Int.natCast_nonneg _
    · exact_mod_cast square_drop_local_bound k
  have hwindow : 0 ≤ (k : Int)-(quarter.val : Int)*512 ∧
      (k : Int)-(quarter.val : Int)*512 ≤ 511 := by
    dsimp [quarter]; omega
  rcases cells_valid segment quarter with hinterval | hexact
  swap
  · have h := hexact ⟨k%512, by omega⟩
    have hfdecomp : segment.val*65536+quarter.val*512+k%512 = fraction := by
      dsimp [segment,quarter,k]; omega
    simpa only [hfdecomp] using h
  have result := cell_error (parameters segment.val) ((quarter.val : Int)*512)
    k drop (dropLimit quarter.val) hinterval hwindow hdrop
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


#print axioms interpolate_error
end GB300Math.RcpPtxBounds
