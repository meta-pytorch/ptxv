import MufuRcp32Signed
namespace GB300Math.SineArgument
open FloatLib.Floats.Formats.BinaryInterchange
set_option maxRecDepth 12000
set_option maxHeartbeats 4000000

def rawPhase (e m : Nat) : Nat :=
  (if e=0 ∨ (e:Int)-125≥25 then 0 else if (e:Int)-125≥0 then m*2^((e:Int)-125).toNat
    else if -((e:Int)-125)<32 then m/2^(-((e:Int)-125)).toNat else 0)%2^25

def floorPhase (e m : Nat) : Nat := if 125 ≤ e then m*2^(e-125) else m/2^(125-e)

theorem phase_matches (e m : Nat) (he : 0<e) (hm : m<2^24) :
    rawPhase e m = floorPhase e m%2^25 := by
  by_cases hbig : 150≤e
  · have hs : ((e:Int)-125).toNat=e-125 := by omega
    have hp : e-125=25+(e-150) := by omega
    simp only [rawPhase,show (e:Int)-125≥25 by omega,or_true,ite_true,
      floorPhase,show 125≤e by omega]
    rw [hp,pow_add]
    simp [Nat.mul_mod]
  · by_cases hmid : 125≤e
    · have hs : ((e:Int)-125).toNat=e-125 := by omega
      simp only [rawPhase,show ¬(e=0 ∨ (e:Int)-125≥25) by omega,ite_false,
        show (e:Int)-125≥0 by omega,ite_true,hs,floorPhase,hmid]
    · have hs : (-((e:Int)-125)).toNat=125-e := by omega
      simp only [rawPhase,show ¬(e=0 ∨ (e:Int)-125≥25) by omega,ite_false,
        show ¬(e:Int)-125≥0 by omega,hs,floorPhase,hmid]
      split_ifs with hsmall
      · rfl
      · have hd : 2^24 ≤ 2^(125-e) := Nat.pow_le_pow_right (by decide) (by omega)
        rw [Nat.div_eq_of_lt (lt_of_lt_of_le hm hd)]

theorem quotient_reduction (m d : Nat) (hd : 0<d) :
    (m:Rat)/((d:Rat)*2^25) = ((m/d/2^25:Nat):Rat)+((m/d%2^25:Nat):Rat)/2^25+
      ((m%d:Nat):Rat)/((d:Rat)*2^25) := by
  have h1 := Nat.mod_add_div m d
  have h2 := Nat.mod_add_div (m/d) (2^25)
  have h1r : ((m%d:Nat):Rat)+(d:Rat)*((m/d:Nat):Rat)=m := by exact_mod_cast h1
  have h2r : ((m/d%2^25:Nat):Rat)+(2^25:Rat)*((m/d/2^25:Nat):Rat)=(m/d:Nat) := by exact_mod_cast h2
  have hd' : (d:Rat)≠0 := by exact_mod_cast (show d≠0 by omega)
  field_simp
  nlinarith only [h1r,h2r]

theorem remainder_bounds (m d : Nat) (hd : 0<d) :
    0 ≤ ((m%d:Nat):Rat)/((d:Rat)*2^25) ∧
    ((m%d:Nat):Rat)/((d:Rat)*2^25) ≤ 1/2^25 := by
  have hr := Nat.mod_lt m hd
  have hdr : 0<(d:Rat) := by exact_mod_cast hd
  have hrr : ((m%d:Nat):Rat) ≤ d := by exact_mod_cast (Nat.le_of_lt hr)
  constructor
  · positivity
  · apply (div_le_iff₀ (by positivity : 0<(d:Rat)*2^25)).mpr
    nlinarith

theorem positive_normal (e : Fin 254) (f : Fin (2^23)) :
    let word := (1+e.val)*2^23+f.val
    let q := rawPhase (1+e.val) (2^23+f.val)
    ∃ x r : Rat, ∃ n : Nat,
      (Model.ofNatBits word : Model FloatFormat.binary32).toRat?=some x ∧
      x=(n:Rat)+(q:Rat)/2^25+r ∧ 0≤r ∧ r≤1/2^25 := by
  have he := e.isLt
  have hf := f.isLt
  let m := 2^23+f.val
  have hm : m<2^24 := by dsimp [m];omega
  have hdecode := MufuRcp32Bounds.decode_normal e f
  rw [phase_matches (1+e.val) m (by omega) hm]
  change ∃ x r : Rat, ∃ n : Nat, _ ∧ x=(n:Rat)+(floorPhase (1+e.val) m%2^25:Nat)/2^25+r ∧ _
  by_cases hh : 124≤e.val
  · let q := m*2^(e.val-124)
    have hphase : floorPhase (1+e.val) m=q := by simp only [floorPhase,show 125≤1+e.val by omega,ite_true];dsimp [q];congr 2;omega
    rw [hphase]
    refine ⟨(q:Rat)/2^25,0,q/2^25,?_,?_,by norm_num,by norm_num⟩
    · have hp : (e.val:Int)-149=(e.val-124:Nat)-25 := by omega
      have hs : (2:Rat)^((e.val:Int)-149)=(2:Rat)^(e.val-124)/2^25 := by rw [hp,zpow_sub₀ (by norm_num),zpow_natCast];norm_num
      rw [hs] at hdecode
      convert hdecode using 1; dsimp [q,m]; push_cast; ring
    · have h := Nat.mod_add_div q (2^25)
      have hr : ((q%2^25:Nat):Rat)+(2^25:Rat)*((q/2^25:Nat):Rat)=q := by exact_mod_cast h
      nlinarith only [hr]
  · let d := 2^(124-e.val)
    have hd : 0<d := by dsimp [d];positivity
    have hphase : floorPhase (1+e.val) m=m/d := by simp only [floorPhase,show ¬125≤1+e.val by omega,ite_false];dsimp [d];congr 2;omega
    rw [hphase]
    refine ⟨(m:Rat)/((d:Rat)*2^25),((m%d:Nat):Rat)/((d:Rat)*2^25),m/d/2^25,?_,quotient_reduction m d hd,(remainder_bounds m d hd).1,(remainder_bounds m d hd).2⟩
    have hp : (e.val:Int)-149=-((124-e.val:Nat):Int)-25 := by omega
    have hs : (2:Rat)^((e.val:Int)-149)=1/((d:Rat)*2^25) := by
      rw [hp,zpow_sub₀ (by norm_num),zpow_neg,zpow_natCast]
      dsimp [d]
      push_cast
      norm_num
      ring
    rw [hs] at hdecode
    convert hdecode using 1; dsimp [m]; ring
#print axioms phase_matches
#print axioms quotient_reduction
#print axioms positive_normal
end GB300Math.SineArgument
