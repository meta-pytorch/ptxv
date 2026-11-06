import RcpPtxScale
import RcpPtxAdapter
set_option maxRecDepth 10000
namespace GB300Math.RcpPtxBounds
open RcpBounds

theorem rcp_normal_output_form (negative : Bool) (e f : Nat)
    (he : 1 ≤ e ∧ e ≤ 252) (hf : f < 2^23) :
    ∃ eo fo : Nat, (1 ≤ eo ∧ eo ≤ 254) ∧ fo < 2^23 ∧
      Mufu.rcp ((if negative then 2^31 else 0)+e*2^23+f) =
        (if negative then 2^31 else 0)+eo*2^23+fo := by
  let x := e*2^23+f
  let z := (Mufu.interpolate MufuTables.rcp f).toNat
  have hs : Mufu.rcp ((if negative then 2^31 else 0)+e*2^23+f) =
      (if negative then 2^31 else 0)+Mufu.rcp x := by
    cases negative
    · simp [x]
    · simp only [↓reduceIte]
      have hx : 2^31+e*2^23+f = 2^31+x := by dsimp [x]; omega
      rw [hx,reciprocal_sign x (by dsimp [x]; omega)]
  rw [hs]
  dsimp only [x]
  rw [output_word_scaled e f he hf]
  by_cases hz : z = 2^24
  · refine ⟨254-e,0,by omega,by norm_num,?_⟩
    simp [z] at hz
    simp [hz]
  · refine ⟨253-e,z%2^23,by omega,by omega,?_⟩
    simp [z] at hz ⊢
    simp [hz,Nat.add_assoc]

theorem rcp_nonftz_normal_adapter (negative : Bool) (e f : Nat)
    (he : 1 ≤ e ∧ e ≤ 252) (hf : f < 2^23) :
    let word := (if negative then 2^31 else 0)+e*2^23+f
    Approx.evaluate "rcp" false word = some (Mufu.rcp word) := by
  let word := (if negative then 2^31 else 0)+e*2^23+f
  have hsmall : ¬Mufu.magnitude word < 0x00800000 := by
    dsimp [word,Mufu.magnitude]
    cases negative <;> simp only [Bool.false_eq_true,↓reduceIte] <;> omega
  have hbig : ¬Mufu.magnitude word > 0x7e800000 := by
    dsimp [word,Mufu.magnitude]
    cases negative <;> simp only [Bool.false_eq_true,↓reduceIte] <;> omega
  have hinput := (mul_one_normal negative e f ⟨he.1,by omega⟩ hf).1
  obtain ⟨eo,fo,heo,hfo,hout⟩ := rcp_normal_output_form negative e f he hf
  have houtput : Approx.mul 0x3f800000 (Mufu.rcp word) = Mufu.rcp word := by
    rw [hout]
    exact (mul_one_normal negative eo fo heo hfo).2
  change Approx.evaluate "rcp" false word = some (Mufu.rcp word)
  change (let s := if Mufu.magnitude word > 0x7e800000 then 0x3e800000
                   else if Mufu.magnitude word < 0x00800000 then 0x4b800000 else 0x3f800000
          some (Approx.mul s (Mufu.rcp (Approx.mul word s)))) = some (Mufu.rcp word)
  dsimp only
  rw [ite_eq_right hbig,ite_eq_right hsmall]
  change Approx.mul word 0x3f800000 = word at hinput
  rw [hinput,houtput]

theorem rcp_nonftz_normal_one_ulp (negative : Bool) (e f output : Nat)
    (he : 1 ≤ e ∧ e ≤ 252) (hf : f < 2^23)
    (hout : Approx.evaluate "rcp" false
      ((if negative then 2^31 else 0)+e*2^23+f) = some output) :
    let word := (if negative then 2^31 else 0)+e*2^23+f
    |normalValue output-1/normalValue word| ≤ reciprocalUlp e f := by
  rw [rcp_nonftz_normal_adapter negative e f he hf] at hout
  cases Option.some.inj hout
  exact rcp_signed_one_ulp negative e f he hf

#print axioms rcp_nonftz_normal_adapter
#print axioms rcp_nonftz_normal_one_ulp
end GB300Math.RcpPtxBounds
